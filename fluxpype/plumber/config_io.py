"""
Read-mostly access to ``config.ini`` for the plumber, without the science stack.

* ``parse`` reads sections, raw values and inline comments (the pipeline strips
  everything after ``#`` before parsing, and so do we).
* ``expand_tasks`` mirrors ``pipe_helper.compute_configs`` and the runner's loop
  order (adapts, rotations, fluxon_count, flow_method).
* ``apply_overrides`` edits ``key = value`` lines in place and keeps comments.
  ``convert_value`` in pipe_helper calls ``eval`` on config values, so the
  override validation here is a security boundary; see ``validate_override``.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

TASK_KEYS = ("rotations", "fluxon_count", "adapts", "flow_method")

# keys a per-job override may never touch (paths, scripts, anything executed or resolved by the shell)
FORBIDDEN_KEYS = frozenset(
    {
        "run_script",
        "base_dir",
        "data_dir",
        "python_dir",
        "rc_path",
        "fl_prefix",
        "pl_prefix",
        "fl_mhdlib",
        "working_dir",
        "flux_perl_lib",
        "fl_perl_lib",
        "config_name",
        "jsoc_email",
    }
)

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALUE_RE = re.compile(r"^[\w.\-+\[\], /~()]*$")
_SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$")
_KV_RE = re.compile(r"^(\s*)([^\s=:#;\[][^=:#]*?)(\s*[=:]\s*)(.*)$")
_INTERP_RE = re.compile(r"\$\{([^}]+)\}")
BATCH_RE = re.compile(r"^[\w\-. ]+$")


class ConfigError(ValueError):
    """A config file or override is invalid."""


@dataclass
class Entry:
    key: str
    value: str
    comment: Optional[str]
    section: str  # where the value is defined: "DEFAULT" or a profile name
    overrides_default: bool = False


@dataclass
class ConfigFile:
    text: str
    defaults: Dict[str, Entry] = field(default_factory=dict)
    sections: Dict[str, Dict[str, Entry]] = field(default_factory=dict)

    @property
    def profiles(self) -> List[str]:
        return list(self.sections)

    def effective(self, section: str) -> Dict[str, Entry]:
        """DEFAULT merged with `section`; entries remember where they came from."""
        if section not in self.sections:
            raise ConfigError(f"unknown profile {section!r}")
        merged = dict(self.defaults)
        for key, entry in self.sections[section].items():
            entry = Entry(entry.key, entry.value, entry.comment, entry.section, key in self.defaults)
            merged[key] = entry
        return merged


def _split_comment(raw: str) -> Tuple[str, Optional[str]]:
    value, sep, comment = raw.partition("#")
    return value.strip(), (comment.strip() or None) if sep else None


def parse(text: str) -> ConfigFile:
    """Parse config text into sections of raw entries (comments kept separately)."""
    cf = ConfigFile(text=text)
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#;":
            continue
        m = _SECTION_RE.match(line)
        if m:
            current = m.group(1).strip()
            if current != "DEFAULT":
                cf.sections.setdefault(current, {})
            continue
        if current is None or line[0] in " \t":
            continue  # key outside a section, or a continuation line: the pipeline ignores both
        m = _KV_RE.match(line)
        if not m:
            continue
        key = m.group(2).strip().lower()
        value, comment = _split_comment(m.group(4))
        target = cf.defaults if current == "DEFAULT" else cf.sections[current]
        target[key] = Entry(key, value, comment, current)
    return cf


def read_file(path) -> ConfigFile:
    with open(os.path.expanduser(str(path)), "r", encoding="utf-8") as f:
        return parse(f.read())


def auto_base_dir(checkout: str) -> str:
    """What ``base_dir = auto`` resolves to: the folder that holds the checkout."""
    return os.path.dirname(os.path.abspath(checkout))


def resolve(effective: Dict[str, Entry], key: str, extra: Optional[Dict[str, str]] = None, _depth: int = 0) -> str:
    """Value of `key` with ``${other}`` placeholders substituted (display and probe use only)."""
    extra = extra or {}
    if key in extra:
        raw = extra[key]
    elif key in effective:
        raw = effective[key].value
    else:
        return ""
    if _depth > 8:
        return raw

    def sub(m):
        name = m.group(1).lower()
        return resolve(effective, name, extra, _depth + 1) if (name in effective or name in extra) else m.group(0)

    return _INTERP_RE.sub(sub, raw)


def resolved_path(effective: Dict[str, Entry], key: str, checkout: str) -> str:
    base = effective.get("base_dir").value if "base_dir" in effective else "auto"
    if base.strip() in ("", "auto"):
        base = auto_base_dir(checkout)
    return os.path.expanduser(resolve(effective, key, {"base_dir": base}))


# ---------------------------------------------------------------------------
# task expansion


def parse_list_or_range(value: str) -> List[str]:
    """Same semantics as pipe_helper.parse_list_or_range (rotations)."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return [item.strip() for item in value.strip("[]").split(",")]
    if value.startswith("(") and value.endswith(")"):
        start, stop, step = map(int, value.strip("()").split(","))
        return [str(x) for x in range(start, stop, step)]
    return [value.strip()]


def parse_list(value: str) -> List[str]:
    """Same semantics as pipe_helper.parse_list (casefolded items)."""
    parts = value.strip("[]").split(",") if value else []
    return [x.strip(" ").casefold() for x in parts]


@dataclass
class TaskPlan:
    tasks: List[Tuple[str, str, str, str]]  # (adapt, cr, nflux, method), runner order
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    axes: Dict[str, List[str]] = field(default_factory=dict)


def expand_tasks(effective: Dict[str, Entry]) -> TaskPlan:
    """Expand a profile's lists into tasks, flagging what the runner would choke on."""
    plan = TaskPlan(tasks=[])
    axes = {}
    for key in TASK_KEYS:
        if key not in effective:
            plan.errors.append(f"{key} is not set; the runner refuses to start without it")
            axes[key] = []
            continue
        raw = effective[key].value
        try:
            items = parse_list_or_range(raw) if key == "rotations" else parse_list(raw)
        except ValueError:
            plan.errors.append(f"{key} = {raw!r} is not a valid list or (start, stop, step) range")
            axes[key] = []
            continue
        kept = [i for i in items if i.strip()]
        if len(kept) != len(items):
            plan.warnings.append(f"{key} has an empty item (trailing comma?); it is ignored here")
        if not kept:
            plan.errors.append(f"{key} is empty; the runner would schedule zero jobs")
        axes[key] = kept
    plan.axes = axes
    for run_key in ("run_script",):
        if run_key not in effective:
            plan.errors.append(f"{run_key} is not set")
    if not plan.errors:
        plan.tasks = [
            (a, r, n, m)
            for a in axes["adapts"]
            for r in axes["rotations"]
            for n in axes["fluxon_count"]
            for m in axes["flow_method"]
        ]
    return plan


def task_instrument(adapt: str) -> str:
    """'adapt' or 'hmi', as the pipeline names the magnetogram source."""
    return "hmi" if str(adapt).strip().lower() in ("", "0", "false", "none") else "adapt"


# ---------------------------------------------------------------------------
# overrides


def validate_override(effective_keys, key: str, value: str) -> str:
    """Return the normalized value or raise ConfigError. Security boundary: see module docstring."""
    key = str(key).strip().lower()
    value = str(value).strip()
    if not _KEY_RE.match(key):
        raise ConfigError(f"bad key {key!r}")
    if key in FORBIDDEN_KEYS:
        raise ConfigError(f"{key} cannot be overridden per job")
    if key not in effective_keys:
        raise ConfigError(f"{key} is not an existing setting of this profile")
    if len(value) > 200:
        raise ConfigError(f"value for {key} is too long")
    if not _VALUE_RE.match(value):
        raise ConfigError(f"value for {key} contains characters that are not allowed")
    if "__" in value or ".." in value:
        raise ConfigError(f"value for {key} is not allowed")
    # config values are passed through eval(): allow literals, never calls or subscripts
    if re.search(r"[\w)\]]\s*[(\[]", value):
        raise ConfigError(f"value for {key} may not contain calls or subscripts")
    if key == "batch_name" and not BATCH_RE.match(value):
        raise ConfigError("batch_name may only contain letters, digits, space, '-', '_' and '.'")
    return value


def _section_bounds(lines: List[str], section: str) -> Tuple[int, int]:
    """(index of header, index one past the section's last line)."""
    start = None
    for i, line in enumerate(lines):
        m = _SECTION_RE.match(line)
        if m:
            if start is not None:
                return start, i
            if m.group(1).strip() == section:
                start = i
    if start is None:
        raise ConfigError(f"unknown profile {section!r}")
    return start, len(lines)


def set_value(text: str, section: str, key: str, value: str) -> str:
    """Set `key = value` inside `section` (including DEFAULT), keeping inline comments.
    No validation: for plumber-internal edits only. Use apply_overrides for user input."""
    key = key.lower()
    lines = text.split("\n")
    start, end = _section_bounds(lines, section)
    for i in range(start + 1, end):
        line = lines[i]
        s = line.strip()
        if not s or s[0] in "#;":
            continue
        m = _KV_RE.match(line)
        if m and m.group(2).strip().lower() == key:
            before, sep, comment = m.group(4).partition("#")
            tail = ""
            if sep:
                gap = before[len(before.rstrip()) :] or " "
                tail = gap + "#" + comment
            lines[i] = f"{m.group(1)}{m.group(2)}{m.group(3)}{value}{tail}"
            return "\n".join(lines)
    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, f"{key} = {value}")
    return "\n".join(lines)


def apply_overrides(text: str, section: str, overrides: Dict[str, str]) -> str:
    """Validate `overrides` against the profile and rewrite only those lines."""
    cf = parse(text)
    keys = set(cf.effective(section))
    cleaned = {}
    for key, value in (overrides or {}).items():
        cleaned[str(key).strip().lower()] = validate_override(keys, key, value)
    for key, value in cleaned.items():
        text = set_value(text, section, key, value)
    return text


def make_snapshot(text: str, section: str, overrides: Dict[str, str], checkout: str) -> str:
    """The text a job runs against: verbatim config, overrides applied, base_dir pinned."""
    out = apply_overrides(text, section, overrides)
    cf = parse(out)
    base = cf.defaults.get("base_dir")
    if base is not None and base.value.strip() in ("", "auto"):
        out = set_value(out, "DEFAULT", "base_dir", auto_base_dir(checkout))
    return out

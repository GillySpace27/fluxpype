"""
Job presets: the sections of ``config.ini`` (built-in, never written) plus presets the user made in
Plumber, kept in ``<home>/presets.ini``.

User presets are plain INI sections holding only their own keys. They inherit ``[DEFAULT]`` from
``config.ini`` exactly as a built-in section does, because a job's snapshot is ``config.ini`` with
``presets.ini`` appended. Every value is checked with the same rules as a per-run override
(``config_io.validate_override``): config values reach ``eval()`` in the pipeline.

Deleting a user preset moves it to ``<home>/presets.trash.ini``; hiding (any preset) only drops it
from the default list.
"""

import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import config_io

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]{0,63}$")
HEADER = "\n\n#############################################\n## Plumber presets ({path})\n"


class PresetError(ValueError):
    pass


def render(sections: Dict[str, Dict[str, str]], note: str = "") -> str:
    out = [note] if note else []
    for name, values in sections.items():
        out.append(f"[{name}]")
        out.extend(f"{k} = {v}" for k, v in values.items())
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def _write_atomic(path: Path, text: str) -> None:
    tmp = Path(f"{path}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class PresetStore:
    def __init__(self, home, config_path, store=None):
        self.home = Path(home)
        self.config_path = str(config_path)
        self.path = self.home / "presets.ini"
        self.trash_path = self.home / "presets.trash.ini"
        self.store = store
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ reading
    def base_text(self) -> str:
        with open(os.path.expanduser(self.config_path), "r", encoding="utf-8") as f:
            return f.read()

    def _read(self, path: Path) -> Dict[str, Dict[str, str]]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return {}
        cf = config_io.parse(text)
        return {name: {k: e.value for k, e in entries.items()} for name, entries in cf.sections.items()}

    def user_sections(self) -> Dict[str, Dict[str, str]]:
        return self._read(self.path)

    def trashed_sections(self) -> Dict[str, Dict[str, str]]:
        return self._read(self.trash_path)

    def combined_text(self) -> str:
        """What jobs and previews read: config.ini, then the user presets."""
        base = self.base_text()
        user = self.user_sections()
        if not user:
            return base
        builtin = set(config_io.parse(base).sections)
        kept = {n: v for n, v in user.items() if n not in builtin}  # a built-in name always wins
        return base.rstrip("\n") + HEADER.format(path=self.path) + "\n" + render(kept)

    def builtin_names(self) -> List[str]:
        return list(config_io.parse(self.base_text()).sections)

    def user_names(self) -> List[str]:
        builtin = set(self.builtin_names())
        return [n for n in self.user_sections() if n not in builtin]

    def hidden(self) -> List[str]:
        val = self.store.get_setting("hidden_presets") if self.store else None
        return list(val or [])

    def origin(self, name: str) -> str:
        return "builtin" if name in self.builtin_names() else "user"

    # ------------------------------------------------------------------ validation
    def _known_keys(self) -> set:
        cf = config_io.parse(self.base_text())
        keys = set(cf.defaults)
        for entries in cf.sections.values():
            keys |= set(entries)
        return keys

    def _check_name(self, name: str, allow_existing: Optional[str] = None) -> str:
        name = str(name).strip()
        if not NAME_RE.match(name):
            raise PresetError("names start with a letter and use letters, digits, '_', '-' or '.' (up to 64)")
        if name.upper() == "DEFAULT":
            raise PresetError("DEFAULT is reserved")
        taken = set(self.builtin_names()) | set(self.user_sections())
        if name in taken and name != allow_existing:
            raise PresetError(f"a preset named {name!r} already exists")
        return name

    def _clean(self, values: Dict[str, str]) -> Dict[str, str]:
        known = self._known_keys()
        out = {}
        for key, value in (values or {}).items():
            key = str(key).strip().lower()
            try:
                out[key] = config_io.validate_override(known, key, str(value))
            except config_io.ConfigError as e:
                raise PresetError(str(e)) from e
        return out

    # ------------------------------------------------------------------ writing
    def _save_user(self, sections: Dict[str, Dict[str, str]]) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        _write_atomic(
            self.path, render(sections, f"# Plumber job presets; edited by Plumber ({time.strftime('%Y-%m-%d')})")
        )

    def _own_values(self, name: str) -> Dict[str, str]:
        user = self.user_sections()
        if name in user:
            return dict(user[name])
        cf = config_io.parse(self.base_text())
        if name not in cf.sections:
            raise PresetError(f"unknown preset {name!r}")
        return {k: e.value for k, e in cf.sections[name].items() if k not in config_io.FORBIDDEN_KEYS}

    def check(self, name: str, base: Optional[str] = None, values: Optional[Dict[str, str]] = None):
        """`(name, base's own values, cleaned values)` for a new preset, or PresetError; writes nothing."""
        with self.lock:
            name = self._check_name(name)
            return name, (self._own_values(base) if base else {}), self._clean(values or {})

    def create(self, name: str, base: Optional[str] = None, values: Optional[Dict[str, str]] = None) -> str:
        with self.lock:
            name, start, clean = self.check(name, base, values)
            user = self.user_sections()
            user[name] = {**start, **clean}
            self._save_user(user)
            return name

    def save(self, name: str, values: Dict[str, str], remove: Optional[List[str]] = None) -> None:
        with self.lock:
            user = self.user_sections()
            if name not in user:
                raise PresetError("built-in presets come from config.ini and are read-only; duplicate it first")
            section = dict(user[name])
            section.update(self._clean(values))
            for key in remove or []:
                section.pop(str(key).strip().lower(), None)
            user[name] = section
            self._save_user(user)

    def rename(self, old: str, new: str) -> str:
        with self.lock:
            user = self.user_sections()
            if old not in user:
                raise PresetError("only your own presets can be renamed")
            new = self._check_name(new, allow_existing=old)
            user = {(new if k == old else k): v for k, v in user.items()}
            self._save_user(user)
            self._rename_hidden(old, new)
            return new

    def delete(self, name: str) -> None:
        with self.lock:
            user = self.user_sections()
            if name not in user:
                raise PresetError("built-in presets cannot be deleted; hide them instead")
            trash = self.trashed_sections()
            trash[name] = user.pop(name)
            self.home.mkdir(parents=True, exist_ok=True)
            _write_atomic(
                self.trash_path, render(trash, "# presets deleted in Plumber; restore them from the Presets page")
            )
            self._save_user(user)
            self._rename_hidden(name, None)

    def restore_deleted(self, name: str) -> str:
        with self.lock:
            trash = self.trashed_sections()
            if name not in trash:
                raise PresetError(f"{name!r} is not in the deleted presets")
            values = trash.pop(name)
            target = name
            n = 2
            while target in set(self.builtin_names()) | set(self.user_sections()):
                target = f"{name}_{n}"
                n += 1
            user = self.user_sections()
            user[target] = values
            self._save_user(user)
            _write_atomic(
                self.trash_path, render(trash, "# presets deleted in Plumber; restore them from the Presets page")
            )
            return target

    def set_hidden(self, name: str, hidden: bool) -> List[str]:
        if name not in set(self.builtin_names()) | set(self.user_sections()):
            raise PresetError(f"unknown preset {name!r}")
        cur = [n for n in self.hidden() if n != name] + ([name] if hidden else [])
        self.store.set_setting("hidden_presets", cur)
        return cur

    def _rename_hidden(self, old: str, new: Optional[str]) -> None:
        if not self.store:
            return
        cur = self.hidden()
        if old in cur:
            cur = [n for n in cur if n != old] + ([new] if new else [])
            self.store.set_setting("hidden_presets", cur)

    # ------------------------------------------------------------------ exchange
    def export(self, names: List[str]) -> str:
        sections = {n: self._own_values(n) for n in names}
        note = (
            f"# fluxpype job presets exported from Plumber on {time.strftime('%Y-%m-%d %H:%M')}\n"
            "# Each section holds only its own settings; the rest comes from [DEFAULT] in config.ini.\n"
            "# Paste into config.ini, or import back into Plumber.\n"
        )
        return render(sections, note)

    def import_text(self, text: str) -> List[str]:
        if len(text) > 512 * 1024:
            raise PresetError("that file is too large to be a preset file")
        cf = config_io.parse(text)
        if not cf.sections:
            raise PresetError("no [sections] found; expected INI text like an exported preset file")
        with self.lock:
            user = self.user_sections()
            taken = set(self.builtin_names()) | set(user)
            added = []
            for name, entries in cf.sections.items():
                values = self._clean({k: e.value for k, e in entries.items()})
                target, n = name, 2
                while target in taken:
                    target = f"{name}_{n}"
                    n += 1
                self._check_name(target)
                user[target] = values
                taken.add(target)
                added.append(target)
            self._save_user(user)
            return added

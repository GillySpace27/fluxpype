export type StageStatus = "pending" | "running" | "ok" | "failed" | "skipped";
export type Source = "event" | "log" | "file" | "none";
export type TaskStatus = "pending" | "running" | "ok" | "failed" | "timeout" | "cancelled" | "crashed";
export type JobStatus = "pending" | "running" | "ok" | "failed" | "cancelled" | "partial";

export interface Stage {
  id: string;
  label: string;
  status: StageStatus;
  source: Source;
  started: number | null;
  ended: number | null;
  elapsed_s: number | null;
  detail: string | null;
  verified: boolean | null;
}

export interface RelaxPoint {
  step: number;
  stiffness: number | null;
  fmax: number | null;
  f_av: number | null;
  fs_av: number | null;
  n_av: number | null;
  vertices: number | null;
  t: number | null;
  s: "event" | "log";
}

export interface RelaxCycle {
  cycle: number;
  threshold?: number | null;
  stiffness?: number | null;
  cum_time_s?: number | null;
  steps?: number | null;
  nc_curve?: number | null;
  nc_proxi?: number | null;
  broken?: number | null;
  t?: number | null;
  s: "event" | "log";
}

export interface Artifact {
  path: string;
  kind: string;
  t: number | null;
  source: Source;
}

export interface TaskState {
  id: string;
  job_id: string;
  profile: string;
  idx: number;
  cr: string;
  nflux: string;
  adapt: string;
  method: string;
  status: TaskStatus;
  started: number | null;
  ended: number | null;
  elapsed_s: number | null;
  exit_code: number | null;
  slots: number;
  attempt: number;
  mode: "events" | "degraded" | "log" | "none";
  degraded: boolean;
  degraded_reason: string | null;
  current_stage: string | null;
  stages: Stage[];
  relax: {
    n_points: number;
    latest: RelaxPoint | null;
    cycle: RelaxCycle | null;
    cycles: number;
    threshold: number | null;
    steps_per_s: number | null;
    source: Source;
  };
  metrics: { name: string; value: number | string | null; unit: string | null; source: Source }[];
  warnings: string[];
  artifacts: Artifact[];
  run: { start: Record<string, unknown> | null; end: { status: string; exit_code: number; elapsed_s: number } | null; log_done: boolean };
  sources: {
    events: { present: boolean; lines: number; bad_lines: number; last_change: number | null; age_s: number | null };
    log: { present: boolean; lines: number; bytes: number; last_change: number | null; age_s: number | null };
    files: { checked: boolean; ok: string[]; missing: string[] };
    process: { alive: boolean | null };
  };
  outcome: string | null;
  observed_at: number;
  spark?: number[];
}

export interface TaskRow {
  id: string;
  job_id: string;
  idx: number;
  cr: string;
  nflux: string;
  adapt: string;
  method: string;
  status: TaskStatus;
  exit_code: number | null;
  created: number;
  started: number | null;
  ended: number | null;
  slots: number;
  attempt: number;
  profile?: string;
}

export interface Job {
  id: string;
  profile: string;
  overrides: Record<string, string>;
  backend: string;
  status: JobStatus;
  created: number;
  started: number | null;
  ended: number | null;
  extra: Record<string, string>;
  tasks: TaskRow[];
  counts: Record<string, number>;
  states?: Record<string, TaskState>;
}

export interface QueueState {
  paused: boolean;
  max_parallel: number;
  cpu_budget: number;
  running: number;
  pending: number;
  slots_used: number;
  default_backend: string;
  active_backend?: string;
  backends: Record<string, { kind: string; slots: number; shared_fs: boolean; description: string }>;
  pending_tasks?: TaskRow[];
}

export interface RecentTask extends TaskRow {
  profile: string;
  thumb: string | null;
  stages_ok: number;
  stages: { id: string; status: StageStatus; source: Source }[];
  failure: string | null;
}

export interface Overview {
  queue: QueueState;
  running: TaskState[];
  pending: TaskRow[];
  recent: RecentTask[];
}

export interface StatusInfo {
  version: string;
  commit?: string | null;
  demo: boolean;
  plumber_home: string;
  config_path: string;
  data_dir: string;
  backends: { name: string; kind: string; slots: number; shared_fs: boolean; description: string; active: boolean }[];
  default_backend: string;
  active_backend: string;
  sandboxed: boolean;
  scenarios: string[];
  stages: { id: string; label: string }[];
  forbidden_keys: string[];
  /** the last MCP-agent request Plumber saw, or null if none since it started */
  agent?: { age_s: number; tool: string; count: number } | null;
  /** seconds since the MCP server's last heartbeat (alive) and since any agent was last seen, kept across restarts (seen) */
  agent_link?: { alive: number | null; seen: number | null };
}

export interface ProfileSummary {
  name: string;
  batch_name: string;
  task_count: number;
  axes: Record<string, string[]>;
  errors: string[];
  warnings: string[];
  own_keys: number;
  own: Record<string, string>;
  runs: { jobs: number; ok: number; last_status: string | null; last_at: number | null } | null;
  origin: "builtin" | "user";
  hidden: boolean;
  cache: PresetCache | null;
  is_default_profile: boolean;
}

export interface PresetCache {
  tasks: number;
  of: number;
  counts: { magnetogram: number; footpoints: number; pfss: number; world: number; relaxed_world: number };
}

export type Health = "ok" | "warn" | "fail" | "off" | "unknown";

export interface HealthCheck {
  id: string;
  label: string;
  status: Health;
  detail: string;
}

export interface BackendHealth {
  name: string;
  kind: string;
  label: string;
  description: string;
  slots: number;
  configured: boolean;
  active: boolean;
  sandboxed: boolean;
  data_dir: string | null;
  status: Health;
  checks: HealthCheck[];
}

export interface HealthReport {
  active_backend: string;
  backends: BackendHealth[];
  system: HealthCheck[];
  checked_at: number;
  overall: Health;
}

export interface ProfileEntry {
  key: string;
  value: string;
  resolved: string | null;
  source: "DEFAULT" | "profile";
  overrides_default: boolean;
  comment: string | null;
  editable: boolean;
}

export interface ProfileDetail {
  name: string;
  entries: ProfileEntry[];
  tasks: {
    cr: string;
    nflux: string;
    adapt: string;
    method: string;
    inst: string;
    cache: { magnetogram: boolean; footpoints: boolean; pfss: boolean; world: boolean; relaxed_world: boolean };
  }[];
  task_count: number;
  truncated: boolean;
  errors: string[];
  warnings: string[];
  axes: Record<string, string[]>;
  data_dir: string;
  slots: number;
  origin: "builtin" | "user";
  hidden: boolean;
  batch_name: string;
  backend: string;
  sandboxed: boolean;
}

export interface CacheRecord {
  id: string;
  path: string;
  name: string;
  size: number;
  mtime: number;
  kind: string;
  cr?: string;
  batch?: string;
  n?: string;
  inst?: string;
  reduce?: string;
  sel?: string;
  steps?: string;
}

export interface CacheData {
  data_dir: string;
  exists: boolean;
  total_count: number;
  total_size: number;
  tiles: { kind: string; count: number; size: number }[];
  scanned_at: number;
  trash: { count: number; size: number };
  records: CacheRecord[];
  truncated: boolean;
  presets: PresetScope[];
  sandboxed: boolean;
}

export interface PresetScope {
  name: string;
  batch: string;
  crs: string[];
  insts: string[];
  hidden: boolean;
}

export interface DataFolder {
  path: string;
  real: string;
  exists: boolean;
  magnetograms: number;
  pfss: number;
  batches: number;
  has_data: boolean;
  aliases?: string[];
}

export interface DataDirReport {
  current: DataFolder;
  source: "plumber" | "config";
  config_value: string;
  candidates: DataFolder[];
}

export interface TrashItem {
  id: string;
  folder: string;
  path: string;
  size: number;
  mtime: number;
  kind: string;
}

export interface RelaxSeries {
  n: number;
  points: RelaxPoint[];
  cycles: RelaxCycle[];
}

export interface LogChunk {
  offset: number;
  start?: number;
  size: number;
  text: string;
  file: string | null;
}

export interface FitsCard {
  key: string;
  value: string;
  comment: string;
}

export interface FilePreview {
  path: string;
  name: string;
  size: number;
  mtime: number;
  view: "image" | "pdf" | "fits" | "text" | "none";
  media_type?: string;
  hdus?: { name: string; dims: number[]; bitpix?: string; cards: FitsCard[]; incomplete?: boolean }[];
  renderable?: boolean;
  lines?: string[];
  truncated?: boolean;
  note?: string;
}

/** A parsed .flux world file (flat arrays; xyz is x,y,z repeated; fluxon i owns vertices offsets[i] to offsets[i+1]). */
export interface FluxWorld {
  frame: number | null;
  boundary: { type: string; args: number[] } | null;
  globals: string[];
  counts: { vertices: number; fluxons: number; concentrations: number; vneighbor_rows: number; pseudo_fluxons: number; bad_rows: number };
  concentrations: { id: number[]; xyz: number[]; flux: number[] };
  fluxons: { id: number[]; flux: number[]; fc0: number[]; fc1: number[]; offsets: number[]; xyz: number[] };
  /** set when the server kept only every `step`-th fluxon to bound the payload */
  decimated: { step: number; fluxons_kept: number; fluxons_total: number; vertices_total: number } | null;
}

/** A numeric .dat table parsed by the server (see plumber/datfile.py); `data` is one list per column, null for NaN. */
export interface DatPlotSpec {
  type: "scatter" | "lines";
  x: number;
  y: number;
  group?: number;
  color?: number;
  logy?: boolean;
}
export interface DatData {
  rows: number;
  bad_rows: number;
  header: boolean;
  columns: { name: string; min: number | null; max: number | null; mean: number | null; nonfinite: number; integer: boolean; sorted: boolean }[];
  data: (number | null)[][];
  decimated: { step: number; rows_kept: number; rows_total: number } | null;
  kind: { id: string; label: string; writer: string; columns: { name: string; unit: string }[] | null; plot: DatPlotSpec; notes: string[] } | null;
}

export interface FitsRender {
  stats: { hdu?: string; shape?: number[]; vmin?: number; vmax?: number; min?: number | null; max?: number | null; nan?: number };
  image: string;
}

/* ------------------------------------------------------------------ one run's results and settings */

export interface ResultFile extends CacheRecord {
  /** true: written while the task ran; false: older, reused from the cache; null: can't tell */
  made_here: boolean | null;
  /** made: written during the task; reused: older; later: rewritten after the task ended */
  origin: "made" | "reused" | "later" | "posthoc" | null;
  /** posthoc only: the plot run that made this file */
  plot_run?: string;
  /** plots only: the stage that writes it, guessed from its folder */
  stage?: string | null;
}

export interface ResultProduct {
  id: "magnetogram" | "footpoints" | "pfss" | "world" | "relaxed_world" | "wind" | "fluxlight";
  label: string;
  files: ResultFile[];
  more: number;
  /** status of the pipeline stage that writes this product, when known */
  stage_status?: StageStatus | null;
}

/** One plot the pipeline defines (see plumber/plotcatalog.py), with the files this task's folders hold for it. */
export interface CatalogPlot {
  id: string;
  stage: string;
  title: string;
  pattern: string;
  writer: string;
  call: string | null;
  gate: string | null;
  summary: string;
  /** defined in the code but not made by the current pipeline (call commented out, unreachable, or manual only) */
  legacy: boolean;
  /** can be remade alone from files already on disk (post-hoc plot run) */
  post_hoc: boolean;
  /** why it cannot be made alone (when post_hoc is false) */
  post_hoc_reason: string | null;
  command: string[] | null;
  command_ref: string | null;
  needs: Record<string, string>;
  group: string | null;
  overwrite_risk: string | null;
  /** all inputs exist (post-hoc plots only) */
  can_generate: boolean;
  /** data-folder-relative globs with no match */
  missing_inputs: string[];
  inputs: string[];
  /** inputs written after the task ended */
  inputs_later: string[];
  last_run: { id: string; status: string; detail: string | null } | null;
  /** which regenerate settings it takes ("light"), when it takes any */
  settings?: string;
  files: ResultFile[];
  n_files: number;
}

export interface PlotRunPlot {
  id: string;
  title: string;
  status: "pending" | "running" | "ok" | "failed" | "cancelled";
  exit_code: number | null;
  elapsed_s: number | null;
  detail: string | null;
  outputs: string[];
  replaced: string[];
}

/** One post-hoc plot run: plots remade from a finished task's files. It never changes the task's status. */
export interface PlotRun {
  id: string;
  task_id: string;
  job_id: string;
  status: "starting" | "running" | "ok" | "failed" | "cancelled" | "crashed";
  created: number | null;
  started: number | null;
  ended: number | null;
  exit_code: number | null;
  overwrite: boolean;
  /** settings changed for this run, as the user chose them */
  settings: Record<string, string>;
  plots: PlotRunPlot[];
  log_size: number;
}

/** One setting the wind or FLUXlight regenerate panel offers, with the value this job ran with. */
export interface RerunField {
  key: string;
  label: string;
  hint: string;
  kind: "choice" | "int" | "float";
  choices?: string[];
  min?: number;
  max?: number;
  value: string;
}
export interface RerunOptions {
  [group: string]: { fields: RerunField[]; can: boolean; missing: string[] };
}

export interface PlotPlanItem {
  id: string;
  title: string;
  argv: string[] | null;
  missing: string[];
  conflicts: string[];
  inputs_later: string[];
  overwrite_risk: string | null;
}

export interface TaskResults {
  task_id: string;
  job_id: string;
  data_dir: string | null;
  products: ResultProduct[];
  plots: ResultFile[];
  catalog: CatalogPlot[];
  headline: string | null;
}

export interface JobConfigEntry extends ProfileEntry {
  overridden: boolean;
  preset_value: string | null;
  differs_from_preset: boolean;
  set_by_plumber: boolean;
}

export interface JobConfig {
  job_id: string;
  profile: string;
  preset_exists: boolean;
  overrides: Record<string, string>;
  data_dir: string;
  batch_name: string;
  entries: JobConfigEntry[];
}

export interface JobFile extends ResultFile {
  stage: string;
  tasks: string[];
}

export interface JobFiles {
  job_id: string;
  data_dir: string | null;
  is_current_data_dir: boolean;
  files: JobFile[];
  missing: { task_id: string; cr: string; nflux: string; stage: string; label: string }[];
}

export interface UpdateCommit {
  sha: string;
  subject: string;
  author: string;
  time: number;
}

export interface UpdateStatus {
  supported: boolean;
  repo: string;
  checked_at: number;
  head?: string | null;
  branch?: string | null;
  upstream?: string | null;
  dirty?: string[];
  behind?: number;
  ahead?: number;
  commits?: UpdateCommit[];
  deps_changed?: string[];
  can_update?: boolean;
  reason?: string | null;
  fetched?: boolean | null;
  fetch_error?: string | null;
}

/** What an agent can do through the MCP server, and what it has done (GET /api/agent). */
export interface AgentTool {
  name: string;
  kind: "read" | "write" | "stop";
  what: string;
  calls: number;
  errors: number;
  last: number | null;
  last_status: number | null;
}
export interface AgentCall {
  call: string;
  ts: number;
  tool: string;
  /** the tool's arguments as JSON text */
  args: string;
  requests: number;
  ms: number;
  /** worst HTTP status among the call's requests */
  status: number;
  paths: string[];
  /** "mcp" for an agent, "ask" for the Ask bar's own reads */
  via: string;
  /** time of the call's last request */
  last: number;
}
/** What the Ask bar has spent today and the plan's usage windows as the CLI last reported them (GET /api/agent). */
export interface AskUsage {
  model: string | null;
  today: { questions: number; input: number; output: number; cache_read: number; cache_write: number; cost: number };
  limits: { ts: number; windows: Record<string, { utilization: number | null; resets_at: number | null }> } | null;
}
export interface AgentRecord {
  ask: AskUsage;
  tools: AgentTool[];
  calls: AgentCall[];
  total: number;
  since: number | null;
}

/** One reading of the machine (GET /api/system); null where the platform gives none. Rates are per second. */
export interface SysSample {
  t: number;
  cpu: number | null;
  mem: number | null;
  net_rx: number | null;
  net_tx: number | null;
  disk_r: number | null;
  disk_w: number | null;
  gpu: number | null;
}
export interface SysSnapshot {
  psutil: boolean;
  now: SysSample;
  history: SysSample[];
  cpu_count: number | null;
  load: number[];
  gpu: { name: string; pct: number; mem_used: number | null; mem_total: number | null } | null;
  storage: { label: string; path: string; total: number; used: number; free: number; pct: number }[];
  cores?: number[];
  memory?: { total: number; used: number; swap_used: number; swap_total: number };
  processes?: {
    top: { pid: number; name: string; cpu: number; rss: number; task: string | null }[];
    tasks: { label: string; cpu: number; rss: number; procs: number }[];
  };
}

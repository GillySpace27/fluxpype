import * as RDialog from "@radix-ui/react-dialog";
import * as RTabs from "@radix-ui/react-tabs";
import * as RTooltip from "@radix-ui/react-tooltip";
import { AlertTriangle, Check, CheckCircle2, CircleDashed, Loader2, MinusCircle, OctagonX, X, XCircle, Ban, Ghost } from "lucide-react";
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import { cn } from "../../lib/cn";

/* ------------------------------------------------------------------ tooltip */
export function Tip({ label, children, side = "top" }: { label: ReactNode; children: ReactNode; side?: "top" | "bottom" | "left" | "right" }) {
  return (
    <RTooltip.Root delayDuration={150}>
      <RTooltip.Trigger asChild>{children}</RTooltip.Trigger>
      <RTooltip.Portal>
        <RTooltip.Content
          side={side}
          sideOffset={6}
          className="z-[100] max-w-xs animate-fade-in rounded-lg border border-line bg-raised px-2.5 py-1.5 text-xs leading-snug text-ink shadow-pop"
        >
          {label}
        </RTooltip.Content>
      </RTooltip.Portal>
    </RTooltip.Root>
  );
}
export const TooltipProvider = ({ children }: { children: ReactNode }) => <RTooltip.Provider>{children}</RTooltip.Provider>;

/* ------------------------------------------------------------------ status */
const TONE: Record<string, { cls: string; label: string }> = {
  running: { cls: "text-run bg-run/10 border-run/25", label: "Running" },
  ok: { cls: "text-ok bg-ok/10 border-ok/25", label: "Succeeded" },
  failed: { cls: "text-fail bg-fail/10 border-fail/25", label: "Failed" },
  timeout: { cls: "text-fail bg-fail/10 border-fail/25", label: "Timed out" },
  crashed: { cls: "text-fail bg-fail/10 border-fail/25", label: "Crashed" },
  cancelled: { cls: "text-muted bg-skip/10 border-skip/25", label: "Cancelled" },
  pending: { cls: "text-muted bg-skip/10 border-skip/25", label: "Queued" },
  partial: { cls: "text-run bg-run/10 border-run/25", label: "Partial" },
  skipped: { cls: "text-muted bg-skip/10 border-skip/25", label: "Skipped" },
};

export function StatusIcon({ status, className }: { status: string; className?: string }) {
  const cls = cn("h-4 w-4 shrink-0", className);
  switch (status) {
    case "running":
      return <Loader2 className={cn(cls, "animate-spin text-run")} />;
    case "ok":
      return <CheckCircle2 className={cn(cls, "text-ok")} />;
    case "failed":
    case "crashed":
      return <XCircle className={cn(cls, "text-fail")} />;
    case "timeout":
      return <OctagonX className={cn(cls, "text-fail")} />;
    case "cancelled":
      return <Ban className={cn(cls, "text-muted")} />;
    case "skipped":
      return <MinusCircle className={cn(cls, "text-skip")} />;
    case "partial":
      return <AlertTriangle className={cn(cls, "text-run")} />;
    default:
      return <CircleDashed className={cn(cls, "text-faint")} />;
  }
}

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const t = TONE[status] ?? TONE.pending;
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11.5px] font-medium", t.cls, className)}>
      {status === "running" ? <span className="h-1.5 w-1.5 animate-pulse-ring rounded-full bg-run" /> : <span className="h-1.5 w-1.5 rounded-full bg-current opacity-80" />}
      {t.label}
    </span>
  );
}

export function Badge({ children, tone = "neutral", className }: { children: ReactNode; tone?: "neutral" | "accent" | "info" | "ok" | "fail" | "run"; className?: string }) {
  const tones = {
    neutral: "border-line bg-raised text-muted",
    accent: "border-accent/30 bg-accent/10 text-accent",
    info: "border-info/30 bg-info/10 text-info",
    ok: "border-ok/25 bg-ok/10 text-ok",
    fail: "border-fail/25 bg-fail/10 text-fail",
    run: "border-run/25 bg-run/10 text-run",
  };
  return <span className={cn("inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-none", tones[tone], className)}>{children}</span>;
}

/* ------------------------------------------------------------------ layout bits */
export function Card({ children, className, ...rest }: { children: ReactNode; className?: string } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("card", className)} {...rest}>
      {children}
    </div>
  );
}

export function SectionTitle({ title, hint, right, className }: { title: ReactNode; hint?: ReactNode; right?: ReactNode; className?: string }) {
  return (
    <div className={cn("mb-3 flex items-end justify-between gap-3", className)}>
      <div className="min-w-0">
        <h2 className="text-[15px] leading-tight">{title}</h2>
        {hint && <p className="mt-0.5 text-xs text-muted">{hint}</p>}
      </div>
      {right}
    </div>
  );
}

export function PageHeader({ title, sub, right }: { title: ReactNode; sub?: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div className="min-w-0">
        <h1 className="text-[22px] leading-tight">{title}</h1>
        {sub && <p className="mt-1 text-[13px] text-muted">{sub}</p>}
      </div>
      {right}
    </div>
  );
}

export function EmptyState({ icon, title, body, action }: { icon?: ReactNode; title: string; body?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-line/90 px-6 py-12 text-center animate-fade-in">
      <div className="mb-3 grid h-11 w-11 place-items-center rounded-full border border-line bg-raised text-muted">{icon ?? <Ghost className="h-5 w-5" />}</div>
      <p className="text-sm font-medium">{title}</p>
      {body && <p className="mt-1 max-w-sm text-[13px] leading-relaxed text-muted">{body}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return (
    <div className={cn("relative overflow-hidden rounded-md bg-raised", className)}>
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-ink/5 to-transparent" />
    </div>
  );
}

export function ErrorNote({ error, retry }: { error: unknown; retry?: () => void }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div className="flex items-start gap-3 rounded-xl border border-fail/30 bg-fail/5 px-4 py-3 text-[13px]">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-fail" />
      <div className="min-w-0 flex-1">
        <p className="font-medium text-fail">Something went wrong</p>
        <p className="mt-0.5 break-words text-muted">{msg}</p>
      </div>
      {retry && (
        <button className="btn btn-sm" onClick={retry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Banner({ tone, title, children }: { tone: "warn" | "error" | "info"; title?: string; children?: ReactNode }) {
  const t = { warn: "border-run/30 bg-run/5 text-run", error: "border-fail/30 bg-fail/5 text-fail", info: "border-info/30 bg-info/5 text-info" }[tone];
  return (
    <div className={cn("flex items-start gap-2.5 rounded-lg border px-3 py-2 text-[13px]", t)}>
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="min-w-0 text-ink/90">
        {title && <p className="font-medium text-current">{title}</p>}
        {children}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ switch + segmented */
export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label?: string }) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cn("relative h-5 w-9 shrink-0 rounded-full border transition-colors", checked ? "border-accent/60 bg-accent/80" : "border-line bg-raised")}
    >
      <span className={cn("absolute top-0.5 h-3.5 w-3.5 rounded-full bg-white shadow transition-all", checked ? "left-[18px]" : "left-0.5")} />
    </button>
  );
}

export function Segmented<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void }) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-bg/50 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={cn("rounded-md px-2.5 py-1 text-xs font-medium transition-colors", value === o.value ? "bg-raised text-ink shadow-sm" : "text-muted hover:text-ink")}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Checkbox({ checked, onChange, label, indeterminate }: { checked: boolean; onChange: (v: boolean) => void; label?: string; indeterminate?: boolean }) {
  return (
    <button
      role="checkbox"
      aria-checked={indeterminate ? "mixed" : checked}
      aria-label={label}
      onClick={(e) => {
        e.stopPropagation();
        onChange(!checked);
      }}
      className={cn(
        "grid h-4 w-4 shrink-0 place-items-center rounded border transition-colors",
        checked || indeterminate ? "border-accent bg-accent text-accent-ink" : "border-faint/60 bg-transparent hover:border-ink/60",
      )}
    >
      {checked ? <Check className="h-3 w-3" strokeWidth={3} /> : indeterminate ? <span className="h-0.5 w-2 rounded bg-current" /> : null}
    </button>
  );
}

/* ------------------------------------------------------------------ tabs */
export const Tabs = RTabs.Root;
export function TabList({ children, className }: { children: ReactNode; className?: string }) {
  return <RTabs.List className={cn("flex gap-1 overflow-x-auto border-b hair pb-px", className)}>{children}</RTabs.List>;
}
export function Tab({ value, children }: { value: string; children: ReactNode }) {
  return (
    <RTabs.Trigger
      value={value}
      className="relative -mb-px flex shrink-0 items-center gap-2 rounded-t-lg border-b-2 border-transparent px-3 py-2 text-[13px] font-medium text-muted transition-colors hover:text-ink data-[state=active]:border-accent data-[state=active]:text-ink"
    >
      {children}
    </RTabs.Trigger>
  );
}
export const TabPanel = ({ value, children, className }: { value: string; children: ReactNode; className?: string }) => (
  <RTabs.Content value={value} className={cn("focus:outline-none", className)}>
    {children}
  </RTabs.Content>
);

/* ------------------------------------------------------------------ dialogs */
export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  side,
  wide,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  side?: boolean;
  wide?: boolean;
}) {
  return (
    <RDialog.Root open={open} onOpenChange={onOpenChange}>
      <RDialog.Portal>
        <RDialog.Overlay className="fixed inset-0 z-50 bg-black/60 backdrop-blur-[2px] data-[state=open]:animate-fade-in" />
        <RDialog.Content
          className={cn(
            "fixed z-50 flex flex-col border border-line bg-surface shadow-pop focus:outline-none",
            side
              ? "inset-y-0 right-0 w-full max-w-[560px] animate-slide-in rounded-l-2xl"
              : cn("left-1/2 top-1/2 max-h-[88vh] w-[calc(100vw-2rem)] -translate-x-1/2 -translate-y-1/2 animate-rise rounded-2xl", wide ? "max-w-3xl" : "max-w-md"),
          )}
        >
          <div className="flex items-start justify-between gap-4 border-b hair px-5 py-4">
            <div className="min-w-0">
              <RDialog.Title className="text-[15px] font-semibold tracking-tight">{title}</RDialog.Title>
              <RDialog.Description className={cn("mt-1 text-[13px] text-muted", !description && "sr-only")}>{description ?? title}</RDialog.Description>
            </div>
            <RDialog.Close className="btn btn-ghost btn-icon -mr-2 -mt-1" aria-label="Close">
              <X className="h-4 w-4" />
            </RDialog.Close>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
          {footer && <div className="flex items-center justify-end gap-2 border-t hair px-5 py-3">{footer}</div>}
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  );
}

export function useConfirm() {
  const [state, setState] = useState<{ title: string; body?: ReactNode; confirm: string; danger?: boolean; resolve: (v: boolean) => void } | null>(null);
  const ask = useCallback(
    (opts: { title: string; body?: ReactNode; confirm?: string; danger?: boolean }) =>
      new Promise<boolean>((resolve) => setState({ title: opts.title, body: opts.body, confirm: opts.confirm ?? "Confirm", danger: opts.danger, resolve })),
    [],
  );
  const close = (v: boolean) => {
    state?.resolve(v);
    setState(null);
  };
  const node = (
    <Dialog
      open={!!state}
      onOpenChange={(o) => !o && close(false)}
      title={state?.title ?? ""}
      footer={
        <>
          <button className="btn" onClick={() => close(false)}>
            Cancel
          </button>
          <button className={cn("btn", state?.danger ? "btn-danger" : "btn-primary")} onClick={() => close(true)}>
            {state?.confirm}
          </button>
        </>
      }
    >
      <div className="text-[13px] leading-relaxed text-muted">{state?.body}</div>
    </Dialog>
  );
  return { ask, node };
}

/* ------------------------------------------------------------------ toasts */
interface Toast {
  id: number;
  tone: "ok" | "error" | "info";
  text: string;
}
const ToastCtx = createContext<(tone: Toast["tone"], text: string) => void>(() => {});
export const useToast = () => useContext(ToastCtx);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const next = useRef(1);
  const push = useCallback((tone: Toast["tone"], text: string) => {
    const id = next.current++;
    setItems((cur) => [...cur.slice(-3), { id, tone, text }]);
    window.setTimeout(() => setItems((cur) => cur.filter((t) => t.id !== id)), tone === "error" ? 7000 : 3800);
  }, []);
  const value = useMemo(() => push, [push]);
  return (
    <ToastCtx.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-[200] flex w-[min(360px,calc(100vw-2rem))] flex-col gap-2 max-md:bottom-20">
        {items.map((t) => (
          <div key={t.id} role="status" className="pointer-events-auto flex animate-rise items-start gap-2.5 rounded-xl border border-line bg-raised px-3.5 py-2.5 text-[13px] shadow-pop">
            {t.tone === "ok" ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-ok" /> : t.tone === "error" ? <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-fail" /> : <CircleDashed className="mt-0.5 h-4 w-4 shrink-0 text-info" />}
            <span className="min-w-0 break-words">{t.text}</span>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

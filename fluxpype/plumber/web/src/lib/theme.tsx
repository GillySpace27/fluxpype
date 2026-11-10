import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type ThemeChoice = "dark" | "light" | "system";
const KEY = "flux-plumber-theme";

function read(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "dark" || v === "light" || v === "system") return v;
  } catch {
    /* storage can be blocked */
  }
  return "dark";
}

function resolve(choice: ThemeChoice): "dark" | "light" {
  if (choice === "system") return matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  return choice;
}

const Ctx = createContext<{ choice: ThemeChoice; theme: "dark" | "light"; setChoice: (c: ThemeChoice) => void }>({
  choice: "dark",
  theme: "dark",
  setChoice: () => {},
});

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [choice, setChoiceState] = useState<ThemeChoice>(read);
  const [theme, setTheme] = useState<"dark" | "light">(() => resolve(read()));

  useEffect(() => {
    const apply = () => {
      const t = resolve(choice);
      setTheme(t);
      document.documentElement.dataset.theme = t;
    };
    apply();
    if (choice !== "system") return;
    const mq = matchMedia("(prefers-color-scheme: light)");
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [choice]);

  const setChoice = useCallback((c: ThemeChoice) => {
    setChoiceState(c);
    try {
      localStorage.setItem(KEY, c);
    } catch {
      /* ignore */
    }
  }, []);

  const value = useMemo(() => ({ choice, theme, setChoice }), [choice, theme, setChoice]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useTheme = () => useContext(Ctx);

/** Current value of a design token as an `rgb()` string, for canvas charts. */
export function token(name: string, alpha = 1): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
  return alpha === 1 ? `rgb(${v})` : `rgb(${v} / ${alpha})`;
}

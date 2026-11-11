/** @type {import('tailwindcss').Config} */
const c = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        bg: c("bg"),
        surface: c("surface"),
        raised: c("raised"),
        line: c("line"),
        ink: c("ink"),
        muted: c("muted"),
        faint: c("faint"),
        accent: c("accent"),
        "accent-ink": c("accent-ink"),
        run: c("run"),
        ok: c("ok"),
        fail: c("fail"),
        info: c("info"),
        skip: c("skip"),
      },
      fontFamily: {
        sans: ['"Inter Variable"', "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ['"JetBrains Mono Variable"', "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        card: "0 1px 0 0 rgb(255 255 255 / 0.03) inset, 0 8px 24px -12px rgb(0 0 0 / 0.5)",
        pop: "0 16px 48px -12px rgb(0 0 0 / 0.55), 0 0 0 1px rgb(var(--line))",
      },
      keyframes: {
        pulseRing: {
          "0%": { boxShadow: "0 0 0 0 rgb(var(--run) / 0.45)" },
          "70%": { boxShadow: "0 0 0 6px rgb(var(--run) / 0)" },
          "100%": { boxShadow: "0 0 0 0 rgb(var(--run) / 0)" },
        },
        agentFlash: {
          "0%": { boxShadow: "0 0 0 0 rgb(var(--ok) / 0.8)", backgroundColor: "rgb(var(--ok) / 0.4)" },
          "100%": { boxShadow: "0 0 0 12px rgb(var(--ok) / 0)", backgroundColor: "rgb(var(--ok) / 0)" },
        },
        shimmer: { "100%": { transform: "translateX(100%)" } },
        slideIn: { from: { transform: "translateX(24px)", opacity: "0" }, to: { transform: "none", opacity: "1" } },
        fadeIn: { from: { opacity: "0" }, to: { opacity: "1" } },
        rise: { from: { transform: "translateY(6px)", opacity: "0" }, to: { transform: "none", opacity: "1" } },
      },
      animation: {
        "pulse-ring": "pulseRing 1.8s ease-out infinite",
        "agent-flash": "agentFlash 0.9s ease-out",
        shimmer: "shimmer 1.6s infinite",
        "slide-in": "slideIn 0.22s cubic-bezier(0.2, 0.8, 0.2, 1)",
        "fade-in": "fadeIn 0.18s ease-out",
        rise: "rise 0.28s cubic-bezier(0.2, 0.8, 0.2, 1)",
      },
    },
  },
  plugins: [],
};

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** This browser tab's id (sessionStorage), sent on writes so a job started here can say so (`origin.tab`). */
export const TAB_ID: string = (() => {
  try {
    let v = sessionStorage.getItem("plumber.tab") ?? "";
    if (!/^[a-z0-9]{12}$/.test(v)) {
      v = Array.from(crypto.getRandomValues(new Uint8Array(6)), (b) => b.toString(16).padStart(2, "0")).join("");
      sessionStorage.setItem("plumber.tab", v);
    }
    return v;
  } catch {
    return "";
  }
})();

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method === "POST" && TAB_ID) headers["X-Flux-Tab"] = TAB_ID;
  const res = await fetch(url, {
    method,
    credentials: "same-origin",
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  get: <T,>(url: string) => request<T>("GET", url),
  post: <T,>(url: string, body?: unknown) => request<T>("POST", url, body ?? {}),
  patch: <T,>(url: string, body: unknown) => request<T>("PATCH", url, body),
  put: <T,>(url: string, body: unknown) => request<T>("PUT", url, body),
  del: <T,>(url: string) => request<T>("DELETE", url),
};

export const artifactUrl = (path: string) => `/api/artifacts?path=${encodeURIComponent(path)}`;

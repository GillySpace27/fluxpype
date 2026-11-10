import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { ToastProvider, TooltipProvider } from "./components/ui";
import { LiveProvider } from "./lib/live";
import { ThemeProvider } from "./lib/theme";
import "./styles.css";

const client = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: true, staleTime: 1000 } } });

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={client}>
        <TooltipProvider>
          <ToastProvider>
            <LiveProvider>
              <BrowserRouter>
                <App />
              </BrowserRouter>
            </LiveProvider>
          </ToastProvider>
        </TooltipProvider>
      </QueryClientProvider>
    </ThemeProvider>
  </React.StrictMode>,
);

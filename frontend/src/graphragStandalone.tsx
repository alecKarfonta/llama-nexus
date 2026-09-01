import React, { Suspense } from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Box, CssBaseline, ThemeProvider } from "@mui/material";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";
import { PageSkeleton } from "@/components/common/PageSkeleton";
import { ToastProvider } from "@/contexts/ToastContext";
import { ChatPage } from "@/pages/ChatPage";
import GraphRAGWorkbenchPage from "@/pages/GraphRAGWorkbenchPage";
import { getLocalPreference } from "@/utils/preferences";
import { theme } from "@/utils/theme";

document.documentElement.dataset.density = getLocalPreference("compactControlSurfaces")
  ? "compact"
  : "comfortable";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      refetchInterval: 60_000,
      retry: 3,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <ToastProvider>
          <ErrorBoundary resetKey="graphrag-standalone">
            <Box
              sx={{
                height: "100dvh",
                minHeight: 0,
                overflow: "auto",
                bgcolor: "background.default",
              }}
            >
              <BrowserRouter>
                <Suspense fallback={<PageSkeleton />}>
                  <Routes>
                    <Route path="/graphrag" element={<GraphRAGWorkbenchPage />} />
                    <Route path="/chat" element={<ChatPage />} />
                    <Route path="*" element={<Navigate to="/graphrag" replace />} />
                  </Routes>
                </Suspense>
              </BrowserRouter>
            </Box>
          </ErrorBoundary>
        </ToastProvider>
      </ThemeProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);

const ROUTE_PATHS: Record<string, string> = {
  chat: "/chat",
  configuration: "/graphrag",
  dashboard: "/graphrag",
  deploy: "/graphrag",
  "embedding-deploy": "/graphrag",
  "graphrag-workbench": "/graphrag",
  graphrag: "/graphrag",
  "knowledge-graph": "/graphrag",
  quantization: "/graphrag",
  "stt-deploy": "/graphrag",
  tts: "/graphrag",
  "tts-deploy": "/graphrag",
  finetuning: "/graphrag",
};

export const getRoutePath = (id: string): string => ROUTE_PATHS[id] || "/graphrag";

export const getRouteMeta = (pathname: string) => ({
  id: pathname.startsWith("/chat") ? "chat" : "graphrag-workbench",
  label: pathname.startsWith("/chat") ? "Chat" : "Document Chat",
  title: pathname.startsWith("/chat") ? "Chat" : "Document Chat",
  description: "GraphRAG document workspace",
  path: pathname.startsWith("/chat") ? "/chat" : "/graphrag",
  color: "#06b6d4",
  section: "Knowledge & RAG",
});

export const getNavigationTrail = (pathname: string) => [
  {
    id: pathname.startsWith("/chat") ? "chat" : "graphrag-workbench",
    label: pathname.startsWith("/chat") ? "Chat" : "Document Chat",
    path: pathname.startsWith("/chat") ? "/chat" : "/graphrag",
  },
];

export const searchableNavigationItems = [];

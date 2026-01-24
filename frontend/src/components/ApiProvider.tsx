"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { getBaseUrl, getWsUrl, waitForBackendReady } from "@/lib/api";

interface ApiContextType {
  apiUrl: string;
  wsUrl: string;
  isReady: boolean;
}

const ApiContext = createContext<ApiContextType>({
  apiUrl: "http://localhost:8001/api/v1",
  wsUrl: "ws://localhost:8001",
  isReady: false,
});

export function useApi() {
  return useContext(ApiContext);
}

export function ApiProvider({ children }: { children: React.ReactNode }) {
  const [apiUrl, setApiUrl] = useState("http://localhost:8001/api/v1");
  const [wsUrl, setWsUrl] = useState("ws://localhost:8001");
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    async function init() {
      try {
        // Wait for backend to be ready (in Tauri mode)
        await waitForBackendReady();

        // Get the dynamic URLs
        const baseUrl = await getBaseUrl();
        const wsBaseUrl = await getWsUrl();

        setApiUrl(`${baseUrl}/api/v1`);
        setWsUrl(wsBaseUrl);
        setIsReady(true);

        console.log("API Provider initialized:", { apiUrl: `${baseUrl}/api/v1`, wsUrl: wsBaseUrl });
      } catch (e) {
        console.error("Failed to initialize API provider:", e);
        // Fallback to defaults
        setIsReady(true);
      }
    }

    init();
  }, []);

  // Don't render children until API URLs are resolved
  if (!isReady) {
    return (
      <div className="h-screen bg-black flex items-center justify-center">
        <div className="text-white text-sm">Initializing backend connection...</div>
      </div>
    );
  }

  return (
    <ApiContext.Provider value={{ apiUrl, wsUrl, isReady }}>
      {children}
    </ApiContext.Provider>
  );
}

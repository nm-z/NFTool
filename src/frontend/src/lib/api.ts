/**
 * API Configuration for Tauri Integration
 *
 * This module provides dynamic backend URL resolution for both:
 * - Tauri desktop app (queries Rust for the backend port)
 * - Development mode (uses localhost:8001)
 */

/**
 * Cache for dynamically discovered API port
 */
let API_PORT: number | null = null;

/**
 * Cache for constructed base URL
 */
let BASE_URL_CACHE: string | null = null;

/**
 * Check if we're running in Tauri (must be called at runtime, not at module load)
 */
function isTauriEnvironment(): boolean {
  if (typeof window === "undefined") return false;
  return window.__TAURI__ !== undefined;
}

/**
 * Get the backend base URL, fetching the port from Tauri if needed
 */
export async function getBaseUrl(): Promise<string> {
  // Return cached URL if available
  if (BASE_URL_CACHE) {
    return BASE_URL_CACHE;
  }

  // In Tauri mode, ask Rust for the port
  if (isTauriEnvironment()) {
    try {
      const { invoke } = window.__TAURI__!.tauri;

      // Retry logic: Backend might not be ready immediately
      let retries = 10;
      while (retries > 0) {
        const port = await invoke<number>("get_api_port");
        if (port > 0) {
          BASE_URL_CACHE = `http://127.0.0.1:${port}/api/v1`;
          console.log("Tauri mode: Using dynamic port", port);
          return BASE_URL_CACHE;
        }
        // Wait 100ms before retry
        await new Promise(resolve => setTimeout(resolve, 100));
        retries--;
      }

      console.error("Failed to get backend port from Tauri after retries");
    } catch (e) {
      console.error("Error invoking get_api_port:", e);
    }
  }

  // Fallback to development URL
  const envUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
  BASE_URL_CACHE = envUrl.replace(/\/+$/, "");
  if (!BASE_URL_CACHE.endsWith("/api/v1")) {
    BASE_URL_CACHE += "/api/v1";
  }
  console.log("Dev mode: Using", BASE_URL_CACHE);
  return BASE_URL_CACHE;
}

/**
 * Get the WebSocket base URL
 */
export async function getWsUrl(): Promise<string> {
  const baseUrl = await getBaseUrl();
  // WebSockets are at /ws root, not /api/v1/ws
  return baseUrl.replace("/api/v1", "").replace("http://", "ws://");
}

/**
 * Wait for the backend to be ready
 * Listens for the "backend-ready" event from Tauri
 */
export function waitForBackendReady(): Promise<void> {
  if (!isTauriEnvironment()) {
    // In dev mode, assume backend is ready
    console.log("Dev mode: Backend assumed ready");
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    try {
      const { event } = window.__TAURI__!;

      // Listen for backend-ready event
      event.listen("backend-ready", (backendEvent: { payload: unknown }) => {
        console.log("Backend ready on port:", backendEvent.payload);
        resolve();
      });

      // Also resolve if we can already get the port
      getBaseUrl().then((url) => {
        if (url.includes("127.0.0.1")) {
          resolve();
        }
      });
    } catch (e) {
      console.error("Error setting up backend-ready listener:", e);
      // Fallback: just resolve after a delay
      setTimeout(resolve, 1000);
    }
  });
}

/**
 * Reset the cached URLs (useful for testing or reconnection)
 */
export function resetUrlCache(): void {
  API_PORT = null;
  BASE_URL_CACHE = null;
}

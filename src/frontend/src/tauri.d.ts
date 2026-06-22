/**
 * Type declarations for Tauri globals
 * These are injected by the Tauri runtime when running as a desktop app
 */

interface TauriInternals {
  tauri: {
    invoke: <T>(command: string, args?: Record<string, unknown>) => Promise<T>;
  };
  event: {
    listen: (eventName: string, handler: (event: { payload: unknown }) => void) => Promise<() => void>;
  };
}

declare global {
  interface Window {
    __TAURI__?: TauriInternals;
  }
}

export {};

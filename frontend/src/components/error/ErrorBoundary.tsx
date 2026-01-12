"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertCircle, RefreshCcw, Home } from "lucide-react";

interface Props {
  children?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen w-screen bg-[hsl(var(--background))] flex items-center justify-center p-6 text-[hsl(var(--foreground))]">
          <div className="max-w-md w-full bg-[hsl(var(--panel))] border border-red-500/30 rounded-2xl p-8 shadow-2xl shadow-red-500/5 space-y-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center text-red-400">
                <AlertCircle size={28} />
              </div>
              <div>
                <h1 className="text-lg font-bold uppercase tracking-tight">System Exception</h1>
                <p className="text-[11px] text-[#52525b] font-mono">UI_CRASH_PROTECTION_ACTIVE</p>
              </div>
            </div>

            <div className="bg-[#000000] border border-[hsl(var(--border))] rounded-lg p-4 font-mono text-[10px] text-red-400/80 overflow-auto max-h-32 custom-scrollbar">
              {this.state.error?.toString()}
            </div>

            <div className="flex flex-col gap-3 pt-2">
              <button
                onClick={() => window.location.reload()}
                className="w-full py-2.5 bg-red-500 text-white text-[11px] font-bold rounded-lg hover:bg-red-600 transition-all flex items-center justify-center gap-2"
              >
                <RefreshCcw size={14} />
                RELOAD APPLICATION
              </button>
              <button
                onClick={() => {
                    this.setState({ hasError: false, error: null });
                    window.location.href = '/';
                }}
                className="w-full py-2.5 bg-[hsl(var(--panel-lighter))] text-[#52525b] text-[11px] font-bold rounded-lg hover:text-[hsl(var(--foreground-active))] transition-all flex items-center justify-center gap-2"
              >
                <Home size={14} />
                RETURN TO DASHBOARD
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
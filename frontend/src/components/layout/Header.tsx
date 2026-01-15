import React, { useRef } from "react";
import {
  Activity,
  Database,
  FolderOpen,
  Play,
  RefreshCw,
  StopCircle,
} from "lucide-react";
import { IconButton } from "../common/UIComponents";
import { WorkspaceType, useTrainingStore } from "@/store/useTrainingStore";

interface HeaderProps {
  isRunning: boolean;
  isStarting: boolean;
  isAborting: boolean;
  currentTrial: number;
  totalTrials: number;
  handleStartTraining: () => void;
  handleAbortTraining: () => void;
  handleResetTraining: () => void;
  setActiveWorkspace: (ws: WorkspaceType) => void;
}

export function Header({
  isRunning,
  isStarting,
  isAborting,
  currentTrial,
  totalTrials,
  handleStartTraining,
  handleAbortTraining,
  handleResetTraining,
  setActiveWorkspace,
}: HeaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { addLog, setLoadedModelPath, runs, logs, metricsHistory } = useTrainingStore();
  const [isUploadOpen, setIsUploadOpen] = React.useState(false);
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
  const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";

  // Derive canonical engine/run state from the runs list when available.
  const activeRun =
    Array.isArray(runs) && runs.length > 0
      ? (runs as Array<Record<string, unknown>>).find((r) =>
          ["running", "queued"].includes((r["status"] as string) || ""),
        )
      : undefined;

  const displayIsRunning =
    Boolean(activeRun && (activeRun as Record<string, unknown>)["status"] === "running") || isRunning;
  const displayIsStarting =
    Boolean(activeRun && (activeRun as Record<string, unknown>)["status"] === "queued") || isStarting;
  const extractNumber = (obj: Record<string, unknown> | undefined, ...keys: string[]) => {
    if (!obj) return undefined;
    for (const k of keys) {
      const v = obj[k];
      if (v != null && !Number.isNaN(Number(v))) return Number(v);
    }
    // try nested config
    const cfg = obj["config"] as Record<string, unknown> | undefined;
    if (cfg) {
      for (const k of keys) {
        const v = cfg[k];
        if (v != null && !Number.isNaN(Number(v))) return Number(v);
      }
    }
    return undefined;
  };

  const displayCurrentTrial = currentTrial;
  const displayTotalTrials =
    totalTrials ||
    extractNumber(
      activeRun as Record<string, unknown>,
      "total_trials",
      "optuna_trials",
      "optunaTrials",
      "trials",
    ) ||
    0;
  const normalizedCurrentTrial =
    displayTotalTrials > 0 ? Math.min(displayCurrentTrial + 1, displayTotalTrials) : displayCurrentTrial;

  const hasRunArtifacts = (logs?.length ?? 0) > 0 || (metricsHistory?.length ?? 0) > 0;
  const canReset =
    !displayIsRunning &&
    !displayIsStarting &&
    !isAborting &&
    hasRunArtifacts;

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      addLog({
        time: new Date().toLocaleTimeString(),
        msg: `Uploading weights: ${file.name}...`,
        type: "info",
      });
      const res = await fetch(`${API_URL}/load-weights`, {
        method: "POST",
        headers: { "X-API-Key": API_KEY },
        body: formData,
      });

      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      setLoadedModelPath(data.path);
      addLog({
        time: new Date().toLocaleTimeString(),
        msg: `Weights loaded: ${data.info.model} (R²: ${data.info.r2})`,
        type: "success",
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addLog({
        time: new Date().toLocaleTimeString(),
        msg: `Upload failed: ${msg}`,
        type: "warn",
      });
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <header className="h-[48px] border-b border-zinc-800 flex items-center justify-between px-4 bg-zinc-950 shrink-0 z-30">
      {/* Hidden input kept for direct upload fallback */}
      <input
        type="file"
        ref={fileInputRef}
        className="fixed -top-full -left-full opacity-0 pointer-events-none"
        accept=".pt,.pth"
        onChange={handleFileChange}
      />
      {/* Visible upload modal for automation / environments that block native file chooser */}
      {isUploadOpen && (
        <div className="fixed inset-0 z-[200] flex items-center justify-center">
          <div className="absolute inset-0 bg-black/60" onClick={() => setIsUploadOpen(false)} />
          <div className="relative bg-zinc-900 border border-zinc-800 rounded p-4 z-[201] w-96">
            <h3 className="text-sm font-bold mb-2">Upload Weights</h3>
            <input
              type="file"
              accept=".pt,.pth"
              onChange={(e) => {
                handleFileChange(e as React.ChangeEvent<HTMLInputElement>);
                setIsUploadOpen(false);
              }}
              data-testid="modal-file-input"
            />
            <div className="mt-3 flex justify-end">
              <button
                onClick={() => setIsUploadOpen(false)}
                className="px-3 py-1 bg-zinc-800 rounded text-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
      {/* Left: App Branding */}
      <div className="flex items-center gap-3 w-1/4">
        <div className="w-6 h-6 bg-blue-500 rounded flex items-center justify-center text-white">
          <Activity size={14} strokeWidth={3} />
        </div>
        <span className="text-[12px] font-bold text-white uppercase tracking-tight">
          NFTOOL STUDIO
        </span>
      </div>

      {/* Center: Title / Status */}
      <div className="flex-1 flex justify-center">
        <div className="flex items-center gap-3 px-4 py-1.5 bg-zinc-900 border border-zinc-800 rounded-lg text-[11px] font-medium text-white min-w-[200px] justify-center">
          <Activity
            size={14}
            className={
              displayIsRunning || displayIsStarting
                ? isAborting
                  ? "text-red-400 animate-spin"
                  : "text-blue-500 animate-pulse"
                : "text-zinc-500"
            }
          />
          <span className="font-bold uppercase tracking-wider">
            {isAborting
              ? "Aborting Training..."
              : displayIsStarting
                ? "Starting..."
                : displayIsRunning
                  ? `Running Trial ${normalizedCurrentTrial}/${displayTotalTrials}`
                  : "Engine Ready"}
          </span>
        </div>
      </div>

      {/* Right: Global Actions */}
      <div className="w-1/4 flex justify-end items-center gap-2">
        <IconButton
          icon={Database}
          onClick={() => setActiveWorkspace("Library")}
          tooltip="Dataset Library"
        />
        <IconButton
          icon={FolderOpen}
          onClick={() => {
            // Prefer the native file chooser by triggering the hidden file input.
            // Fall back to the visible modal when the input ref is not available.
            if (fileInputRef.current) {
              try {
                fileInputRef.current.click();
              } catch {
                // If programmatic click is blocked, open the visible modal as fallback.
                setIsUploadOpen(true);
              }
            } else {
              setIsUploadOpen(true);
            }
          }}
          tooltip="Load Weights"
        />
        <div className="h-4 w-px bg-zinc-800 mx-1"></div>
        <div className="flex items-center gap-1.5 bg-zinc-900 rounded-md p-0.5 border border-zinc-800">
        {isRunning ? (
            <button
              type="button"
              data-testid="btn-stop"
              onClick={() => {
                console.debug("Header: Abort clicked", {
                  isRunning,
                  isAborting,
                });
                handleAbortTraining();
              }}
              disabled={isAborting}
              className={`flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold transition-all ${isAborting ? "text-red-400/50 bg-red-500/5 cursor-not-allowed" : "text-red-400 bg-red-500/10 hover:bg-red-500/20"}`}
            >
              {isAborting ? (
                <RefreshCw size={12} className="animate-spin" />
              ) : (
                <StopCircle size={12} />
              )}
              {isAborting ? "Aborting..." : "Stop"}
            </button>
          ) : canReset ? (
            <button
              type="button"
              data-testid="btn-reset"
              onClick={handleResetTraining}
              className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-blue-400 bg-blue-500/10 hover:bg-blue-500/20 transition-all"
            >
              <RefreshCw size={12} />
              Reset
            </button>
          ) : (
            <button
              type="button"
              data-testid="btn-run"
              onClick={() => {
                console.debug("Header: Run clicked", { isRunning, isStarting });
                handleStartTraining();
              }}
              className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-white bg-blue-500 hover:bg-blue-600 transition-all"
            >
              <Play size={12} fill="currentColor" />
              Run
            </button>
          )}
        </div>
      </div>
    </header>
  );
}

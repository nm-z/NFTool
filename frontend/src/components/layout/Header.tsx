import React from "react";
import { Activity, Database, FolderOpen, Play, RefreshCw, StopCircle } from "lucide-react";
import { IconButton } from "../common/UIComponents";
import { WorkspaceType } from "@/store/useTrainingStore";

interface HeaderProps {
  isRunning: boolean;
  isAborting: boolean;
  currentTrial: number;
  totalTrials: number;
  handleStartTraining: () => void;
  handleAbortTraining: () => void;
  setActiveWorkspace: (ws: WorkspaceType) => void;
}

export function Header({ 
  isRunning, isAborting, currentTrial, totalTrials, 
  handleStartTraining, handleAbortTraining, setActiveWorkspace 
}: HeaderProps) {
  return (
    <header className="h-[48px] border-b border-[hsl(var(--border))] flex items-center justify-between px-4 bg-[hsl(var(--panel))] shrink-0 z-30">
      {/* Left: App Branding */}
      <div className="flex items-center gap-3 w-1/4">
        <div className="w-6 h-6 bg-[#3b82f6] rounded flex items-center justify-center text-[hsl(var(--foreground-active))]">
          <Activity size={14} strokeWidth={3} />
        </div>
        <span className="text-[12px] font-bold text-[hsl(var(--foreground-active))] tracking-tight">NFTOOL STUDIO</span>
      </div>

      {/* Center: Title / Status */}
      <div className="flex-1 flex justify-center">
        <div className="flex items-center gap-3 px-4 py-1.5 bg-[hsl(var(--panel-lighter))]/50 border border-[hsl(var(--border-muted))] rounded-lg text-[11px] font-medium text-[hsl(var(--foreground-active))] min-w-[200px] justify-center">
          <Activity size={14} className={isRunning ? (isAborting ? "text-red-400 animate-spin" : "text-[#3b82f6] animate-pulse") : "text-[#52525b]"} />
          <span className="uppercase tracking-widest font-bold">
            {isAborting ? "Aborting Training..." : (isRunning ? `Running Trial ${currentTrial}/${totalTrials}` : "Engine Ready")}
          </span>
        </div>
      </div>

      {/* Right: Global Actions */}
      <div className="w-1/4 flex justify-end items-center gap-2">
        <IconButton icon={Database} onClick={() => setActiveWorkspace("Library")} tooltip="Dataset Library" />
        <IconButton icon={FolderOpen} onClick={() => {}} tooltip="Load weights" />
        <div className="h-4 w-px bg-[hsl(var(--panel-lighter))] mx-1"></div>
        <div className="flex items-center gap-1.5 bg-[hsl(var(--panel-lighter))] rounded-md p-0.5 border border-[hsl(var(--border-muted))]">
          {isRunning ? (
            <button 
              onClick={handleAbortTraining} 
              disabled={isAborting}
              className={`flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold transition-all ${isAborting ? 'text-red-400/50 bg-red-500/5 cursor-not-allowed' : 'text-red-400 bg-red-500/10 hover:bg-red-500/20'}`}
            >
              {isAborting ? <RefreshCw size={12} className="animate-spin" /> : <StopCircle size={12} />}
              {isAborting ? 'Aborting...' : 'Stop'}
            </button>
          ) : (
            <button onClick={handleStartTraining} className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-[hsl(var(--foreground-active))] bg-[#3b82f6] hover:bg-[#2563eb] transition-all">
              <Play size={12} fill="currentColor" />
              Run
            </button>
          )}
        </div>
      </div>
    </header>
  );
}

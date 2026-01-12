import React from "react";
import { ResourceBadge } from "../common/UIComponents";

interface FooterProps {
  isAdvancedMode: boolean;
  setIsAdvancedMode: (mode: boolean) => void;
  isRunning: boolean;
  isAborting: boolean;
  progress: number;
  hardwareStats: any;
  wsStatusLabel: string;
  wsStatusColor: string;
}

export function Footer({ 
  isAdvancedMode, setIsAdvancedMode, isRunning, isAborting, progress, hardwareStats, wsStatusLabel, wsStatusColor 
}: FooterProps) {
  return (
    <footer className="h-8 border-t border-[hsl(var(--border))] bg-[hsl(var(--panel))] flex items-center justify-between px-4 text-[10px] font-mono shrink-0 z-50">
      <div className="flex items-center gap-4">
        <div className="flex bg-[hsl(var(--panel-lighter))] rounded overflow-hidden border border-[hsl(var(--border-muted))]">
          <button 
            onClick={() => setIsAdvancedMode(false)}
            className={`px-3 py-0.5 font-bold transition-colors ${!isAdvancedMode ? 'bg-[#3b82f6] text-[hsl(var(--foreground-active))]' : 'text-[#52525b] hover:text-[hsl(var(--foreground))]'}`}
          >
            BASIC
          </button>
          <button 
            onClick={() => setIsAdvancedMode(true)}
            className={`px-3 py-0.5 font-bold transition-colors ${isAdvancedMode ? 'bg-[#3b82f6] text-[hsl(var(--foreground-active))]' : 'text-[#52525b] hover:text-[hsl(var(--foreground))]'}`}
          >
            ADVANCED
          </button>
        </div>
        <div className="h-3 w-px bg-[hsl(var(--panel-lighter))]"></div>
        <div className="flex items-center gap-2">
          <span className={isRunning ? (isAborting ? "text-red-400" : "text-[#3b82f6] animate-pulse") : "text-[#52525b]"}>
            {isAborting ? "ABORTING_ENGINE..." : (isRunning ? `TRAINING_PASS_${progress}%` : "READY_IDLE")}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-4">
          <ResourceBadge label="CPU" value={`${(hardwareStats?.cpu_percent ?? 0).toFixed(1)}%`} />
          <ResourceBadge label="RAM" value={`${(hardwareStats?.ram_percent ?? 0).toFixed(0)}%`} />
          <div className="h-3 w-px bg-[hsl(var(--panel-lighter))]"></div>
          <ResourceBadge label="GPU" value={`${hardwareStats?.gpu_use_percent || 0}%`} color="text-[#3b82f6]" />
          <ResourceBadge label="VRAM" value={`${hardwareStats?.vram_used_gb || 0} / ${hardwareStats?.vram_total_gb || 0} GB`} color="text-[#3b82f6]" />
        </div>
        <div className="h-3 w-px bg-[hsl(var(--panel-lighter))]"></div>
        <div className="flex items-center gap-2">
          <div className={`w-1.5 h-1.5 rounded-full ${wsStatusColor}`}></div>
          <span className="text-[#52525b] uppercase font-bold tracking-tighter">{wsStatusLabel}</span>
        </div>
      </div>
    </footer>
  );
}

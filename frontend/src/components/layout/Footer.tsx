import React from "react";
import { ResourceBadge } from "../common/UIComponents";
import { useTrainingStore } from "@/store/useTrainingStore";

interface FooterProps {
  progress: number;
  wsStatusLabel: string;
  wsStatusColor: string;
}

export function Footer({ progress, wsStatusLabel, wsStatusColor }: FooterProps) {
  const isAdvancedMode = useTrainingStore(s => s.isAdvancedMode);
  const setIsAdvancedMode = useTrainingStore(s => s.setIsAdvancedMode);
  const isRunning = useTrainingStore(s => s.isRunning);
  const isAborting = useTrainingStore(s => s.isAborting);
  const hardwareStats = useTrainingStore(s => s.hardwareStats);

  return (
    <footer className="h-8 border-t border-zinc-800 bg-zinc-950 flex items-center justify-between px-4 text-[10px] font-mono shrink-0 z-50">
      <div className="flex items-center gap-4">
        <div className="flex bg-zinc-900 rounded overflow-hidden border border-zinc-800">
          <button 
            onClick={() => setIsAdvancedMode(false)}
            className={`px-3 py-0.5 font-bold transition-colors ${!isAdvancedMode ? 'bg-blue-500 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}
          >
            BASIC
          </button>
          <button 
            onClick={() => setIsAdvancedMode(true)}
            className={`px-3 py-0.5 font-bold transition-colors ${isAdvancedMode ? 'bg-blue-500 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}
          >
            ADVANCED
          </button>
        </div>
        <div className="h-3 w-px bg-zinc-800"></div>
        <div className="flex items-center gap-2">
          <span className={isRunning ? (isAborting ? "text-red-400" : "text-blue-500 animate-pulse") : "text-zinc-500"}>
            {isAborting ? "ABORTING_ENGINE..." : (isRunning ? `TRAINING_PASS_${progress}%` : "READY_IDLE")}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-4">
          <ResourceBadge label="CPU" value={`${(hardwareStats?.cpu_percent ?? 0).toFixed(1)}%`} />
          <ResourceBadge label="RAM" value={`${(hardwareStats?.ram_percent ?? 0).toFixed(0)}%`} />
          <div className="h-3 w-px bg-zinc-800"></div>
          <ResourceBadge label="GPU" value={`${hardwareStats?.gpu_use_percent || 0}%`} color="text-blue-500" />
          <ResourceBadge label="VRAM" value={`${hardwareStats?.vram_used_gb || 0} / ${hardwareStats?.vram_total_gb || 0} GB`} color="text-blue-500" />
        </div>
        <div className="h-3 w-px bg-zinc-800"></div>
        <div className="flex items-center gap-2">
          <div className={`w-1.5 h-1.5 rounded-full ${wsStatusColor}`}></div>
          <span className="text-zinc-500 font-bold">{wsStatusLabel}</span>
        </div>
      </div>
    </footer>
  );
}

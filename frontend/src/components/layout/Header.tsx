import React, { useRef } from "react";
import { Activity, Database, FolderOpen, Play, RefreshCw, StopCircle } from "lucide-react";
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
  setActiveWorkspace: (ws: WorkspaceType) => void;
}

export function Header({ 
  isRunning, isStarting, isAborting, currentTrial, totalTrials, 
  handleStartTraining, handleAbortTraining, setActiveWorkspace 
}: HeaderProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { addLog, setLoadedModelPath } = useTrainingStore();
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
  const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      addLog({ time: new Date().toLocaleTimeString(), msg: `Uploading weights: ${file.name}...`, type: "info" });
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
        type: "success" 
      });
    } catch (err: any) {
      addLog({ time: new Date().toLocaleTimeString(), msg: `Upload failed: ${err.message}`, type: "warn" });
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <header className="h-[48px] border-b border-zinc-800 flex items-center justify-between px-4 bg-zinc-950 shrink-0 z-30">
      <input 
        type="file" 
        ref={fileInputRef} 
        className="fixed -top-full -left-full opacity-0 pointer-events-none" 
        accept=".pt,.pth" 
        onChange={handleFileChange}
      />
      {/* Left: App Branding */}
      <div className="flex items-center gap-3 w-1/4">
        <div className="w-6 h-6 bg-blue-500 rounded flex items-center justify-center text-white">
          <Activity size={14} strokeWidth={3} />
        </div>
        <span className="text-[12px] font-bold text-white uppercase tracking-tight">NFTOOL STUDIO</span>
      </div>

      {/* Center: Title / Status */}
      <div className="flex-1 flex justify-center">
        <div className="flex items-center gap-3 px-4 py-1.5 bg-zinc-900 border border-zinc-800 rounded-lg text-[11px] font-medium text-white min-w-[200px] justify-center">
          <Activity size={14} className={isRunning || isStarting ? (isAborting ? "text-red-400 animate-spin" : "text-blue-500 animate-pulse") : "text-zinc-500"} />
          <span className="font-bold uppercase tracking-wider">
            {isAborting ? "Aborting Training..." : (isStarting ? "Starting..." : (isRunning ? `Running Trial ${currentTrial}/${totalTrials}` : "Engine Ready"))}
          </span>
        </div>
      </div>

      {/* Right: Global Actions */}
      <div className="w-1/4 flex justify-end items-center gap-2">
        <IconButton icon={Database} onClick={() => setActiveWorkspace("Library")} tooltip="Dataset Library" />
        <IconButton 
          icon={FolderOpen} 
          onClick={() => {
            fileInputRef.current?.click();
          }} 
          tooltip="Load Weights" 
        />
        <div className="h-4 w-px bg-zinc-800 mx-1"></div>
        <div className="flex items-center gap-1.5 bg-zinc-900 rounded-md p-0.5 border border-zinc-800">
          {isRunning ? (
            <button 
              type="button"
              onClick={handleAbortTraining} 
              disabled={isAborting}
              className={`flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold transition-all ${isAborting ? 'text-red-400/50 bg-red-500/5 cursor-not-allowed' : 'text-red-400 bg-red-500/10 hover:bg-red-500/20'}`}
            >
              {isAborting ? <RefreshCw size={12} className="animate-spin" /> : <StopCircle size={12} />}
              {isAborting ? 'Aborting...' : 'Stop'}
            </button>
          ) : (
            <button 
              type="button"
              onClick={handleStartTraining} 
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

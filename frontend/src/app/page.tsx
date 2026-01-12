"use client";

import React, { useState, useEffect, useRef } from "react";
import { Panel, Group as PanelGroup, Separator as PanelResizeHandle } from "react-resizable-panels";
import { Settings, AlertCircle } from "lucide-react";
import * as Dialog from "@radix-ui/react-dialog";
import { useTrainingStore } from "@/store/useTrainingStore";

// Components
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Sidebar } from "@/components/layout/Sidebar";
import { TrainWorkspace } from "@/components/workspaces/TrainWorkspace";
import { LibraryWorkspace } from "@/components/workspaces/LibraryWorkspace";
import { Inspector } from "@/components/inspector/Inspector";
import { ErrorBoundary } from "@/components/error/ErrorBoundary";

const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
const WS_URL = API_URL.replace(/^http/, "ws");

export default function Dashboard() {
  const { 
    activeWorkspace, setActiveWorkspace,
    isRunning, setIsRunning,
    isStarting, setIsStarting,
    isAborting, setIsAborting,
    progress, setProgress,
    currentTrial, totalTrials, setTrialInfo,
    isAdvancedMode, setIsAdvancedMode,
    hardwareStats, setHardwareStats,
    addLog, setLogs,
    setResult,
    setMetricsHistory,
    addMetric,
    setGpuList,
    datasets, setDatasets,
    selectedPredictor, setSelectedPredictor,
    selectedTarget, setSelectedTarget,
    runs, setRuns,
    split, setSplit,
    deviceChoice, gpuChoice
  } = useTrainingStore();

  const [isMounted, setIsMounted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const [wsStatus, setWsStatus] = useState<number>(3); 

  const getConnectionStatus = () => {
    switch (wsStatus) {
      case 0: return { label: "CONNECTING", color: "bg-[#f59e0b]" };
      case 1: return { label: "CONNECTED", color: "bg-[#22c55e]" };
      case 2: return { label: "CLOSING", color: "bg-[#f59e0b]" };
      case 3: return { label: "CLOSED", color: "bg-[#ef4444]" };
      default: return { label: "DISCONNECTED", color: "bg-[#ef4444]" };
    }
  };

  useEffect(() => {
    setIsMounted(true);
  }, []);

  useEffect(() => {
    if (!isMounted) return;
    const headers = { "X-API-Key": API_KEY };
    const fetchData = async () => {
      try {
        const dsRes = await fetch(`${API_URL}/datasets`, { headers });
        if (dsRes.ok) {
          const data = await dsRes.json();
          if (Array.isArray(data)) {
            setDatasets(data);
            if (data.length > 0 && !selectedPredictor) {
              const pFile = data.find((d: any) => d.name.toLowerCase().includes("predictor")) || data[0];
              const tFile = data.find((d: any) => d.name.toLowerCase().includes("target")) || (data.length > 1 ? data[1] : data[0]);
              if (pFile) setSelectedPredictor(pFile.path);
              if (tFile) setSelectedTarget(tFile.path);
            }
          }
        }
      } catch (e) { console.error("Dataset fetch error:", e); }

      try {
        const runsRes = await fetch(`${API_URL}/runs`, { headers });
        if (runsRes.ok) setRuns(await runsRes.json());
      } catch (e) { console.error("Runs fetch error:", e); }

      try {
        const gpuRes = await fetch(`${API_URL}/gpus`, { headers });
        if (gpuRes.ok) setGpuList(await gpuRes.json());
      } catch (e) { console.error("GPU fetch error:", e); }
    };
    fetchData();
  }, [isMounted, API_URL, API_KEY, setDatasets, setRuns, setGpuList, selectedPredictor, setSelectedPredictor, setSelectedTarget]);

  useEffect(() => {
    if (!isMounted) return;
    let active = true;
    let ws: WebSocket | null = null;

    const connect = () => {
      if (!active) return;
      try {
        ws = new WebSocket(`${WS_URL}/ws`, [`api-key-${API_KEY}`]);
        socketRef.current = ws;
        setWsStatus(ws.readyState);
        
        ws.onopen = () => { 
          if (active) {
            console.log("WebSocket connected to:", WS_URL);
            setWsStatus(1); 
          }
        };
        
        ws.onmessage = (event) => {
          if (!active) return;
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === "init") {
              setIsRunning(msg.data.is_running);
              if (msg.data.is_running) setIsStarting(false);
              setIsAborting(msg.data.is_aborting || false);
              setProgress(msg.data.progress);
              setTrialInfo(msg.data.current_trial, msg.data.total_trials);
              setResult(msg.data.result);
              if (msg.data.metrics_history) setMetricsHistory(msg.data.metrics_history);
              if (msg.data.hardware_stats) setHardwareStats(msg.data.hardware_stats);
              if (msg.data.logs) setLogs(msg.data.logs);
            } else if (msg.type === "status") {
              setIsRunning(msg.data.is_running);
              if (msg.data.is_running) setIsStarting(false);
              setIsAborting(msg.data.is_aborting || false);
              setProgress(msg.data.progress);
              setTrialInfo(msg.data.current_trial, msg.data.total_trials);
              if (msg.data.result) setResult(msg.data.result);
            } else if (msg.type === "log") {
              addLog(msg.data);
            } else if (msg.type === "metrics") {
              addMetric(msg.data);
            } else if (msg.type === "hardware") {
              setHardwareStats(msg.data);
            }
          } catch (e) { console.error("WS message parse error:", e); }
        };
        
        ws.onclose = () => { 
          if (active) { 
            console.log("WebSocket closed, retrying...");
            setWsStatus(3); 
            setTimeout(connect, 5000); 
          } 
        };
        
        ws.onerror = (e) => { 
          if (active) {
            console.error("WebSocket error:", e);
            setWsStatus(3); 
          }
        };
      } catch (e) { console.error("WS connection instantiation error:", e); }
    };
    connect();
    return () => { active = false; if (ws) ws.close(); };
  }, [isMounted, WS_URL, API_KEY, setIsRunning, setProgress, setTrialInfo, setResult, setMetricsHistory, setHardwareStats, setLogs, addLog, addMetric, setIsAborting]);

  const parseRange = (val: string): [number, number] => {
    const parts = val.split("→").map(part => part.trim());
    const min = parseFloat(parts[0]) || 0;
    const max = parseFloat(parts[parts.length - 1]) || min;
    return [min, max];
  };

  const handleStartTraining = async () => {
    const s = useTrainingStore.getState();
    
    const [lrMin, lrMax] = parseRange(s.lrRange);
    const [layersMin, layersMax] = parseRange(s.layersRange);
    const [lSizeMin, lSizeMax] = parseRange(s.layerSizeRange);
    const [dropMin, dropMax] = parseRange(s.dropoutRange);
    const [hDimMin, hDimMax] = parseRange(s.hDimRange);
    const [convMin, convMax] = parseRange(s.convBlocks);
    const [cnnFilterMin, cnnFilterMax] = parseRange(s.cnnFilterCapRange);

    const config = {
      model_choice: s.modelType,
      seed: parseInt(s.seed) || 42,
      patience: parseInt(s.patience) || 100,
      train_ratio: s.split / 100,
      val_ratio: (100 - s.split) / 200,
      test_ratio: (100 - s.split) / 200,
      optuna_trials: parseInt(s.trials) || 10,
      optimizers: ["AdamW"],
      n_layers_min: Math.round(layersMin),
      n_layers_max: Math.round(layersMax),
      l_size_min: Math.round(lSizeMin),
      l_size_max: Math.round(lSizeMax),
      lr_min: lrMin,
      lr_max: lrMax,
      drop_min: dropMin,
      drop_max: dropMax,
      h_dim_min: hDimMin,
      h_dim_max: hDimMax,
      conv_blocks_min: Math.round(convMin),
      conv_blocks_max: Math.round(convMax),
      kernel_size: parseInt(s.kernelSize) || 3,
      gpu_throttle_sleep: parseFloat(s.gpuThrottle) || 0.1,
      cnn_filter_cap_min: Math.round(cnnFilterMin),
      cnn_filter_cap_max: Math.round(cnnFilterMax),
      max_epochs: parseInt(s.maxEpochs) || 200,
      device: s.deviceChoice,
      gpu_id: s.gpuChoice,
      predictor_path: s.selectedPredictor,
      target_path: s.selectedTarget
    };

    setIsStarting(true);
    try {
      const res = await fetch(`${API_URL}/train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": API_KEY
        },
        body: JSON.stringify(config)
      });
      
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || `Server responded with ${res.status}`);
      }
      
      const data = await res.json();
      addLog({ 
        time: new Date().toLocaleTimeString(), 
        msg: `Training pipeline initiated: ${data.run_id}`, 
        type: "info" 
      });
    } catch (e: any) {
      console.error("Training error:", e);
      setError(`Execution Failed: ${e.message}`);
      addLog({ 
        time: new Date().toLocaleTimeString(), 
        msg: `Execution Error: ${e.message}`, 
        type: "warn" 
      });
      setIsStarting(false);
    }
  };

  const handleAbortTraining = async () => {
    try {
      const res = await fetch(`${API_URL}/abort`, {
        method: "POST",
        headers: { "X-API-Key": API_KEY }
      });
      if (!res.ok) throw new Error(await res.text());
      addLog({ 
        time: new Date().toLocaleTimeString(), 
        msg: "Termination signal dispatched to core engine", 
        type: "warn" 
      });
    } catch (e: any) {
      setError(`Abort Failed: ${e.message}`);
    }
  };

  if (!isMounted) return <div className="h-screen bg-black" />;

  const connStatus = getConnectionStatus();

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-white selection:bg-blue-500/20 font-sans overflow-hidden">
      <Header 
        isRunning={isRunning} isStarting={isStarting} isAborting={isAborting} 
        currentTrial={currentTrial} totalTrials={totalTrials} 
        handleStartTraining={handleStartTraining} 
        handleAbortTraining={handleAbortTraining} 
        setActiveWorkspace={setActiveWorkspace} 
      />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar 
          activeWorkspace={activeWorkspace} 
          setActiveWorkspace={setActiveWorkspace} 
          onOpenSettings={() => setIsSettingsOpen(true)}
        />

        <div className="flex-1 flex flex-col min-w-0">
          <PanelGroup orientation="horizontal">
            <Panel defaultSize={75} minSize={40}>
              <div className="flex flex-col h-full overflow-hidden bg-zinc-950">
                {activeWorkspace === "Train" && (
                  <ErrorBoundary>
                    <TrainWorkspace />
                  </ErrorBoundary>
                )}
                {activeWorkspace === "Library" && (
                  <ErrorBoundary>
                    <LibraryWorkspace />
                  </ErrorBoundary>
                )}
              </div>
            </Panel>

            {activeWorkspace === "Train" && (
              <>
                <PanelResizeHandle className="w-px bg-zinc-800 hover:bg-blue-500/50 transition-colors" />
                <Panel defaultSize={25} minSize={20}>
                  <ErrorBoundary>
                    <Inspector 
                      setError={setError}
                    />
                  </ErrorBoundary>
                </Panel>
              </>
            )}
          </PanelGroup>
        </div>
      </div>

      <Footer 
        progress={progress} 
        wsStatusLabel={connStatus.label} wsStatusColor={connStatus.color}
      />

      {error && (
        <div className="fixed bottom-12 right-6 z-[100] max-w-md animate-in fade-in slide-in-from-right-4">
          <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 flex gap-3 shadow-2xl backdrop-blur-md">
            <AlertCircle className="text-red-500 shrink-0" size={18} />
            <div className="flex-1">
              <h4 className="text-[11px] font-bold text-red-500 uppercase tracking-widest mb-1">System Error</h4>
              <p className="text-[11px] text-red-200/80 font-mono leading-relaxed">{error}</p>
              <button 
                onClick={() => setError(null)}
                className="mt-3 text-[10px] font-bold text-red-500 hover:text-red-400 uppercase"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}

      <Dialog.Root open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[100]" />
          <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 bg-zinc-900 border border-zinc-800 rounded-xl p-6 shadow-2xl z-[101]">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 rounded bg-blue-500/10 flex items-center justify-center text-blue-500">
                <Settings size={18} />
              </div>
              <Dialog.Title className="text-sm font-bold uppercase tracking-widest">Global Settings</Dialog.Title>
            </div>
            <Dialog.Description className="sr-only">
              System configuration and preference persistence settings.
            </Dialog.Description>
            <div className="space-y-4 py-4 border-y border-zinc-800/50">
              <p className="text-[11px] text-zinc-400 font-mono leading-relaxed">
                System configuration module is currently under development. Preference persistence and API endpoint configuration will be available in V3.1.
              </p>
            </div>
            <div className="mt-6 flex justify-end">
              <button 
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setIsSettingsOpen(false);
                }}
                className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white text-[11px] font-bold rounded-lg transition-all"
              >
                CLOSE
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}

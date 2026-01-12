"use client";

import React, { useState, useEffect, useRef } from "react";
import { Panel, Group as PanelGroup, Separator as PanelResizeHandle } from "react-resizable-panels";
import { FastForward, AlertCircle } from "lucide-react";
import { useTrainingStore } from "@/store/useTrainingStore";

// Components
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Sidebar } from "@/components/layout/Sidebar";
import { TrainWorkspace } from "@/components/workspaces/TrainWorkspace";
import { LibraryWorkspace } from "@/components/workspaces/LibraryWorkspace";
import { Inspector } from "@/components/inspector/Inspector";

const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
const WS_URL = API_URL.replace(/^http/, "ws");

export default function Dashboard() {
  const { 
    activeWorkspace,
    setActiveWorkspace,
    isAdvancedMode,
    setIsAdvancedMode,
    isRunning, 
    isAborting,
    progress, 
    currentTrial,
    totalTrials,
    logs, 
    result,
    metricsHistory,
    hardwareStats,
    deviceChoice,
    setDeviceChoice,
    gpuChoice,
    setGpuChoice,
    gpuList,
    setGpuList,
    loadedModelPath,
    setLoadedModelPath,
    activePreset,
    setActivePreset,
    hasUnsavedChanges,
    setHasUnsavedChanges,
    setIsRunning, 
    setIsAborting,
    setProgress, 
    setTrialInfo,
    addLog, 
    setResult, 
    clearLogs,
    addMetric,
    setMetricsHistory,
    setHardwareStats,
    setLogs
  } = useTrainingStore();

  const [modelInputPath, setModelInputPath] = useState("");
  const [isMounted, setIsMounted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const [wsStatus, setWsStatus] = useState<number>(3); // CLOSED

  const getConnectionStatus = () => {
    switch (wsStatus) {
      case 0: // CONNECTING
        return { label: "CONNECTING", color: "bg-[#f59e0b]" };
      case 1: // OPEN
        return { label: "CONNECTED", color: "bg-[#22c55e]" };
      case 2: // CLOSING
        return { label: "CLOSING", color: "bg-[#f59e0b]" };
      case 3: // CLOSED
        return { label: "CLOSED", color: "bg-[#ef4444]" };
      default:
        return { label: "DISCONNECTED", color: "bg-[#ef4444]" };
    }
  };

  // Set isMounted to true on mount
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Inspector States
  const [seed, setSeed] = useState("42");
  const [patience, setPatience] = useState("100");
  const [trials, setTrials] = useState("10");
  const [lrRange, setLrRange] = useState("1e-4 → 1e-3");
  const [split, setSplit] = useState(70);
  const [modelType, setModelType] = useState("NN");
  const [convBlocks, setConvBlocks] = useState("1 → 3");
  const [kernelSize, setKernelSize] = useState("3");

  // Search/Bound States
  const [layersRange, setLayersRange] = useState("1 → 8");
  const [layerSizeRange, setLayerSizeRange] = useState("128 → 1024");
  const [dropoutRange, setDropoutRange] = useState("0.0 → 0.0");
  const [hDimRange, setHDimRange] = useState("32 → 256");

  // Performance States
  const [maxEpochs, setMaxEpochs] = useState("200");
  const [gpuThrottle, setGpuThrottle] = useState("0.1");
  const [cnnFilterCapRange, setCnnFilterCapRange] = useState("512 → 1024");

  // Dataset States
  const [datasets, setDatasets] = useState<{name: string, path: string}[]>([]);
  const [selectedPredictor, setSelectedPredictor] = useState("");
  const [selectedTarget, setSelectedTarget] = useState("");
  const [runs, setRuns] = useState<any[]>([]);
  const [selectedRun, setSelectedRun] = useState<any>(null);

  // Fetch Datasets and Runs
  useEffect(() => {
    const headers = { "X-API-Key": API_KEY };
    
    const fetchData = async () => {
      try {
        const dsRes = await fetch(`${API_URL}/datasets`, { headers });
        if (!dsRes.ok) throw new Error(`HTTP error! status: ${dsRes.status}`);
        const data = await dsRes.json();
        if (Array.isArray(data)) {
          setDatasets(data);
          if (data.length > 0) {
            const predictorFile = data.find((d: any) => d.name.toLowerCase().includes("predictor")) || data[0];
            const targetFile = data.find((d: any) => d.name.toLowerCase().includes("target")) || (data.length > 1 ? data[1] : data[0]);
            if (predictorFile) setSelectedPredictor(predictorFile.path);
            if (targetFile) setSelectedTarget(targetFile.path);
          }
        }
      } catch (e: any) {
        console.error("Failed to fetch datasets:", e);
        setError(`Failed to load datasets: ${e.message}`);
      }

      try {
        const runsRes = await fetch(`${API_URL}/runs`, { headers });
        if (!runsRes.ok) throw new Error(`HTTP error! status: ${runsRes.status}`);
        const data = await runsRes.json();
        setRuns(data);
      } catch (e: any) {
        console.error("Failed to fetch runs:", e);
      }

      try {
        const gpuRes = await fetch(`${API_URL}/gpus`, { headers });
        if (!gpuRes.ok) throw new Error(`HTTP error! status: ${gpuRes.status}`);
        const data = await gpuRes.json();
        setGpuList(data);
      } catch (e: any) {
        console.error("Failed to fetch GPUs:", e);
      }
    };

    fetchData();
  }, []);

// WebSocket Handshake
  useEffect(() => {
    let active = true;
    let ws: WebSocket | null = null;

    const connect = () => {
      if (!active) return;
      
      // Use subprotocol for API key authentication
      ws = new WebSocket(`${WS_URL}/ws`, [`api-key-${API_KEY}`]);
      socketRef.current = ws;
      setWsStatus(ws.readyState);

      ws.onopen = () => {
        if (active) setWsStatus(1); // OPEN
      };

      ws.onmessage = (event) => {
        if (!active) return;
        const msg = JSON.parse(event.data);
        if (msg.type === "init") {
          setIsRunning(msg.data.is_running);
          setIsAborting(msg.data.is_aborting || false);
          setProgress(msg.data.progress);
          setTrialInfo(msg.data.current_trial, msg.data.total_trials);
          setResult(msg.data.result);
          if (msg.data.metrics_history) setMetricsHistory(msg.data.metrics_history);
          if (msg.data.hardware_stats) setHardwareStats(msg.data.hardware_stats);
          if (msg.data.logs) setLogs(msg.data.logs);
        } else if (msg.type === "status") {
          setIsRunning(msg.data.is_running);
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
      };

      ws.onclose = () => {
        if (active) {
          setWsStatus(3); // CLOSED
          setTimeout(connect, 2000);
        }
      };

      ws.onerror = () => {
        if (active) setWsStatus(3); // CLOSED
      };
    };

    connect();

    return () => {
      active = false;
      if (ws) {
        ws.close();
      }
    };
  }, [setIsRunning, setProgress, setTrialInfo, setResult, setMetricsHistory, setHardwareStats, setLogs, addLog, addMetric]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const handleStartTraining = async () => {
    if (isRunning) return;
    setIsRunning(true); // Optimistic update to prevent double-click
    setHasUnsavedChanges(false);
    setMetricsHistory([]);
    
    // Parse range strings helper
    const parseRange = (rangeStr: string, defaultMin: number, defaultMax: number) => {
      const parts = rangeStr.split("→").map(p => p.trim());
      return {
        min: parseFloat(parts[0]) || defaultMin,
        max: parseFloat(parts[1]) || defaultMax
      };
    };

    const lrBounds = parseRange(lrRange, 1e-4, 1e-3);
    const layersBounds = parseRange(layersRange, 1, 8);
    const layerSizeBounds = parseRange(layerSizeRange, 128, 1024);
    const dropoutBounds = parseRange(dropoutRange, 0.0, 0.0);
    const hDimBounds = parseRange(hDimRange, 32, 256);
    const convBlocksBounds = parseRange(convBlocks, 1, 5);
    const filterCapBounds = parseRange(cnnFilterCapRange, 512, 1024);

    const config = {
      model_choice: modelType, 
      seed: parseInt(seed), 
      patience: parseInt(patience),
      train_ratio: split / 100, 
      val_ratio: (100 - split) / 2 / 100, 
      test_ratio: (100 - split) / 2 / 100,
      optuna_trials: parseInt(trials), 
      optimizers: ["AdamW"],
      n_layers_min: layersBounds.min, 
      n_layers_max: layersBounds.max, 
      l_size_min: layerSizeBounds.min, 
      l_size_max: layerSizeBounds.max,
      lr_min: lrBounds.min, 
      lr_max: lrBounds.max, 
      drop_min: dropoutBounds.min, 
      drop_max: dropoutBounds.max,
      h_dim_min: hDimBounds.min, 
      h_dim_max: hDimBounds.max,
      conv_blocks_min: convBlocksBounds.min,
      conv_blocks_max: convBlocksBounds.max,
      kernel_size: parseInt(kernelSize) || 3,
      // Performance
      max_epochs: parseInt(maxEpochs) || 200,
      gpu_throttle_sleep: parseFloat(gpuThrottle) || 0.1,
      cnn_filter_cap_min: filterCapBounds.min,
      cnn_filter_cap_max: filterCapBounds.max,
      device: deviceChoice,
      gpu_id: gpuChoice,
      predictor_path: selectedPredictor,
      target_path: selectedTarget
    };

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
        const errData = await res.json();
        const errorMessage = Array.isArray(errData.detail) 
          ? errData.detail.map((e: any) => e.msg).join(", ") 
          : errData.detail || `Server error: ${res.status}`;
        throw new Error(errorMessage);
      }
    } catch (e: any) {
      console.error("Training start failed:", e);
      setError(`Failed to start training: ${e.message}`);
    }
  };

  const handleAbortTraining = async () => {
    try {
      const res = await fetch(`${API_URL}/abort`, { 
        method: "POST",
        headers: { "X-API-Key": API_KEY }
      });
      if (!res.ok) throw new Error(`Abort failed: ${res.status}`);
    } catch (e: any) {
      console.error("Abort failed:", e);
      setError(`Failed to abort training: ${e.message}`);
    }
  };

  if (!isMounted) {
    return <div className="h-screen bg-[hsl(var(--background))]" />;
  }

  return (
    <div className="flex flex-col h-screen bg-[hsl(var(--background))] text-[hsl(var(--foreground))] selection:bg-[#3b82f6]/20 font-sans overflow-hidden outline-none" suppressHydrationWarning tabIndex={-1}>
      
      {error && (
        <div className="bg-red-500/10 border-b border-red-500/20 px-4 py-2 flex items-center justify-between z-[100] animate-in slide-in-from-top duration-300">
          <div className="flex items-center gap-2 text-red-400 text-[10px] font-bold uppercase tracking-widest">
            <AlertCircle size={14} />
            <span>SYSTEM_ALERT: {error}</span>
          </div>
          <button onClick={() => setError(null)} className="text-red-400/50 hover:text-red-400 transition-colors">
            <FastForward size={14} className="rotate-90" />
          </button>
        </div>
      )}

      <Header 
        isRunning={isRunning} 
        isAborting={isAborting} 
        currentTrial={currentTrial} 
        totalTrials={totalTrials} 
        handleStartTraining={handleStartTraining} 
        handleAbortTraining={handleAbortTraining} 
        setActiveWorkspace={setActiveWorkspace} 
      />

      {/* MAIN CONTENT AREA */}
      <div className="flex-1 flex overflow-hidden">
        <Sidebar activeWorkspace={activeWorkspace} setActiveWorkspace={setActiveWorkspace} />

        {/* CENTER + RIGHT PANELS */}
        <div className="flex-1 flex flex-col min-w-0">
          <PanelGroup orientation="horizontal">
            {/* CENTER PANEL */}
            <Panel defaultSize={75} minSize={40}>
              <div className="flex flex-col h-full overflow-hidden bg-[hsl(var(--background))]">
                {activeWorkspace === "Train" && (
                  <TrainWorkspace 
                    metricsHistory={metricsHistory} 
                    isRunning={isRunning} 
                    progress={progress} 
                    currentTrial={currentTrial} 
                    totalTrials={totalTrials} 
                    result={result} 
                    logs={logs}
                    clearLogs={clearLogs}
                    logEndRef={logEndRef}
                    hardwareStats={hardwareStats}
                    split={split}
                    setSplit={setSplit}
                    loadedModelPath={loadedModelPath}
                    selectedPredictor={selectedPredictor}
                  />
                )}
                {activeWorkspace === "Library" && (
                  <LibraryWorkspace 
                    runs={runs} 
                    onSelectRun={setSelectedRun} 
                    selectedRun={selectedRun} 
                    selectedPredictor={selectedPredictor}
                  />
                )}
              </div>
            </Panel>

            {activeWorkspace === "Train" && (
              <>
                <PanelResizeHandle className="w-px bg-[hsl(var(--panel-lighter))] hover:bg-[#3b82f6]/50 transition-colors" />
                {/* RIGHT INSPECTOR */}
                <Panel defaultSize={25} minSize={20}>
                  <Inspector 
                    activeWorkspace={activeWorkspace}
                    modelType={modelType}
                    setModelType={setModelType}
                    seed={seed} setSeed={setSeed}
                    patience={patience} setPatience={setPatience}
                    trials={trials} setTrials={setTrials}
                    lrRange={lrRange} setLrRange={setLrRange}
                    convBlocks={convBlocks} setConvBlocks={setConvBlocks}
                    kernelSize={kernelSize} setKernelSize={setKernelSize}
                    layersRange={layersRange} setLayersRange={setLayersRange}
                    layerSizeRange={layerSizeRange} setLayerSizeRange={setLayerSizeRange}
                    dropoutRange={dropoutRange} setDropoutRange={setDropoutRange}
                    hDimRange={hDimRange} setHDimRange={setHDimRange}
                    maxEpochs={maxEpochs} setMaxEpochs={setMaxEpochs}
                    gpuThrottle={gpuThrottle} setGpuThrottle={setGpuThrottle}
                    cnnFilterCapRange={cnnFilterCapRange} setCnnFilterCapRange={setCnnFilterCapRange}
                    datasets={datasets}
                    selectedPredictor={selectedPredictor}
                    setSelectedPredictor={setSelectedPredictor}
                    selectedTarget={selectedTarget}
                    setSelectedTarget={setSelectedTarget}
                    isAdvancedMode={isAdvancedMode}
                    deviceChoice={deviceChoice}
                    setDeviceChoice={setDeviceChoice}
                    gpuChoice={gpuChoice}
                    setGpuChoice={setGpuChoice}
                    gpuList={gpuList}
                    setGpuList={setGpuList}
                    hardwareStats={hardwareStats}
                    setError={setError}
                  />
                </Panel>
              </>
            )}
          </PanelGroup>
        </div>
      </div>

      <Footer 
        isAdvancedMode={isAdvancedMode} 
        setIsAdvancedMode={setIsAdvancedMode} 
        isRunning={isRunning} 
        isAborting={isAborting} 
        progress={progress} 
        hardwareStats={hardwareStats} 
        wsStatusLabel={getConnectionStatus().label}
        wsStatusColor={getConnectionStatus().color}
      />
    </div>
  );
}
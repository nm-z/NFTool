"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  Panel, 
  Group as PanelGroup, 
  Separator as PanelResizeHandle 
} from "react-resizable-panels";
import * as Accordion from "@radix-ui/react-accordion";
import * as Tabs from "@radix-ui/react-tabs";
import * as Separator from "@radix-ui/react-separator";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { Command } from "cmdk";
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { 
  Activity, 
  Settings, 
  Play, 
  FileCode, 
  Database, 
  LineChart as LineChartIcon, 
  Cpu, 
  Layers,
  ChevronDown,
  Terminal,
  StopCircle,
  BarChart3,
  Search,
  Box,
  LayoutDashboard,
  History as HistoryIcon,
  FileText,
  Upload,
  Command as CommandIcon,
  Sun,
  Moon,
  ChevronRight,
  Info,
  MoreVertical,
  Plus,
  Filter,
  Download,
  Trash2,
  RefreshCw,
  Check,
  AlertCircle,
  FolderOpen,
  FastForward,
  Cpu as CpuIcon,
  HardDrive,
  MousePointer2,
  Table
} from "lucide-react";
import { useTrainingStore, WorkspaceType } from "@/store/useTrainingStore";

export default function Dashboard() {
  const { 
    activeWorkspace,
    setActiveWorkspace,
    isAdvancedMode,
    setIsAdvancedMode,
    isRunning, 
    progress, 
    currentTrial,
    totalTrials,
    logs, 
    result,
    metricsHistory,
    hardwareStats,
    loadedModelPath,
    setLoadedModelPath,
    activePreset,
    setActivePreset,
    hasUnsavedChanges,
    setHasUnsavedChanges,
    setIsRunning, 
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

  const [openCommand, setOpenCommand] = useState(false);
  const [modelInputPath, setModelInputPath] = useState("");
  const logEndRef = useRef<HTMLDivElement>(null);
  const socketRef = useRef<WebSocket | null>(null);

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
  const [cnnFilterCap, setCnnFilterCap] = useState("512");

  // Dataset States
  const [datasets, setDatasets] = useState<{name: string, path: string}[]>([]);
  const [selectedPredictor, setSelectedPredictor] = useState("");
  const [selectedTarget, setSelectedTarget] = useState("");
  const [runs, setRuns] = useState<any[]>([]);
  const [selectedRun, setSelectedRun] = useState<any>(null);

  // Fetch Datasets and Runs
  useEffect(() => {
    fetch("http://localhost:8001/datasets")
      .then(res => res.json())
      .then(data => {
        setDatasets(data);
        if (data.length > 0) {
          setSelectedPredictor(data[0].path);
          if (data.length > 1) setSelectedTarget(data[1].path);
          else setSelectedTarget(data[0].path);
        }
      });

    fetch("http://localhost:8001/runs")
      .then(res => res.json())
      .then(data => setRuns(data));
  }, []);

  // WebSocket Handshake
  useEffect(() => {
    let active = true;
    let ws: WebSocket | null = null;

    const connect = () => {
      if (!active) return;
      
      ws = new WebSocket("ws://localhost:8001/ws");
      socketRef.current = ws;

      ws.onmessage = (event) => {
        if (!active) return;
        const msg = JSON.parse(event.data);
        if (msg.type === "init") {
          setIsRunning(msg.data.is_running);
          setProgress(msg.data.progress);
          setTrialInfo(msg.data.current_trial, msg.data.total_trials);
          setResult(msg.data.result);
          if (msg.data.metrics_history) setMetricsHistory(msg.data.metrics_history);
          if (msg.data.hardware_stats) setHardwareStats(msg.data.hardware_stats);
          if (msg.data.logs) setLogs(msg.data.logs);
        } else if (msg.type === "status") {
          setIsRunning(msg.data.is_running);
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
          setTimeout(connect, 2000);
        }
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

  // Keyboard Shortcuts
  useEffect(() => {
    const handleKeys = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpenCommand(true);
      }
      if (e.key === "Escape") setOpenCommand(false);
    };
    window.addEventListener("keydown", handleKeys);
    return () => window.removeEventListener("keydown", handleKeys);
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const handleStartTraining = async () => {
    if (isRunning) return;
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
      cnn_filter_cap: parseInt(cnnFilterCap) || 512,
      predictor_path: selectedPredictor,
      target_path: selectedTarget
    };
    await fetch("http://localhost:8001/train", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config)
    });
  };

  const handleAbortTraining = async () => {
    await fetch("http://localhost:8001/abort", { method: "POST" });
  };

  return (
    <div className="flex flex-col h-screen bg-[#09090b] text-[#a1a1aa] selection:bg-[#3b82f6]/20 font-sans overflow-hidden">
      
      {/* GLOBAL TOP BAR */}
      <header className="h-[48px] border-b border-[#18181b] flex items-center justify-between px-4 bg-[#0c0c0e] shrink-0 z-30">
        {/* Left: App Branding */}
        <div className="flex items-center gap-3 w-1/4">
          <div className="w-6 h-6 bg-[#3b82f6] rounded flex items-center justify-center text-[#fafafa]">
            <Activity size={14} strokeWidth={3} />
          </div>
          <span className="text-[12px] font-bold text-[#fafafa] tracking-tight">NFTOOL STUDIO</span>
        </div>

        {/* Center: Title / Status */}
        <div className="flex-1 flex justify-center">
          <div className="flex items-center gap-3 px-4 py-1.5 bg-[#18181b]/50 border border-[#27272a] rounded-lg text-[11px] font-medium text-[#fafafa] min-w-[200px] justify-center">
            <Activity size={14} className={isRunning ? "text-[#3b82f6] animate-pulse" : "text-[#52525b]"} />
            <span className="uppercase tracking-widest font-bold">
              {isRunning ? `Running Trial ${currentTrial}/${totalTrials}` : "Engine Ready"}
            </span>
          </div>
        </div>

        {/* Right: Global Actions */}
        <div className="w-1/4 flex justify-end items-center gap-2">
          <IconButton icon={Database} onClick={() => setActiveWorkspace("Library")} tooltip="Dataset Library" />
          <IconButton icon={FolderOpen} onClick={() => setOpenCommand(true)} tooltip="Load weights" />
          <div className="h-4 w-px bg-[#18181b] mx-1"></div>
          <div className="flex items-center gap-1.5 bg-[#18181b] rounded-md p-0.5 border border-[#27272a]">
            {isRunning ? (
              <button onClick={handleAbortTraining} className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-red-400 bg-red-500/10 hover:bg-red-500/20 transition-all">
                <StopCircle size={12} />
                Stop
              </button>
            ) : (
              <button onClick={handleStartTraining} className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-[#fafafa] bg-[#3b82f6] hover:bg-[#2563eb] transition-all">
                <Play size={12} fill="currentColor" />
                Run
              </button>
            )}
          </div>
          <div className="h-4 w-px bg-[#18181b] mx-1"></div>
          <button onClick={() => setOpenCommand(true)} className="flex items-center gap-2 px-2 py-1 hover:bg-[#18181b] rounded text-[10px] font-mono text-[#52525b] transition-colors">
            CTRL+K
          </button>
        </div>
      </header>

      {/* MAIN CONTENT AREA */}
      <div className="flex-1 flex overflow-hidden">
        {/* LEFT NAV WORKSPACES */}
        <aside className="w-[56px] border-r border-[#18181b] bg-[#0c0c0e] flex flex-col items-center py-4 gap-4 shrink-0 z-40">
          <div className="flex-1 w-full flex flex-col items-center gap-1">
            <NavIcon icon={Layers} active={activeWorkspace === "Train"} onClick={() => setActiveWorkspace("Train")} tooltip="Train" />
            <NavIcon icon={Database} active={activeWorkspace === "Library"} onClick={() => setActiveWorkspace("Library")} tooltip="Library" />
          </div>
          <NavIcon icon={Settings} active={false} onClick={() => {}} tooltip="Global Settings" />
        </aside>

        {/* CENTER + RIGHT PANELS */}
        <div className="flex-1 flex flex-col min-w-0">
          <PanelGroup orientation="horizontal">
            {/* CENTER PANEL */}
            <Panel defaultSize={75} minSize={40}>
              <div className="flex flex-col h-full overflow-hidden bg-[#09090b]">
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
                  />
                )}
                {activeWorkspace === "Library" && <LibraryWorkspace runs={runs} onSelectRun={setSelectedRun} selectedRun={selectedRun} />}
              </div>
            </Panel>

            {activeWorkspace === "Train" && (
              <>
                <PanelResizeHandle className="w-px bg-[#18181b] hover:bg-[#3b82f6]/50 transition-colors" />
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
                    cnnFilterCap={cnnFilterCap} setCnnFilterCap={setCnnFilterCap}
                    datasets={datasets}
                    selectedPredictor={selectedPredictor}
                    setSelectedPredictor={setSelectedPredictor}
                    selectedTarget={selectedTarget}
                    setSelectedTarget={setSelectedTarget}
                    isAdvancedMode={isAdvancedMode}
                  />
                </Panel>
              </>
            )}
          </PanelGroup>
        </div>
      </div>

      {/* GLOBAL BOTTOM BAR */}
      <footer className="h-8 border-t border-[#18181b] bg-[#0c0c0e] flex items-center justify-between px-4 text-[10px] font-mono shrink-0 z-50">
        <div className="flex items-center gap-4">
          <div className="flex bg-[#18181b] rounded overflow-hidden border border-[#27272a]">
            <button 
              onClick={() => setIsAdvancedMode(false)}
              className={`px-3 py-0.5 font-bold transition-colors ${!isAdvancedMode ? 'bg-[#3b82f6] text-[#fafafa]' : 'text-[#52525b] hover:text-[#a1a1aa]'}`}
            >
              BASIC
            </button>
            <button 
              onClick={() => setIsAdvancedMode(true)}
              className={`px-3 py-0.5 font-bold transition-colors ${isAdvancedMode ? 'bg-[#3b82f6] text-[#fafafa]' : 'text-[#52525b] hover:text-[#a1a1aa]'}`}
            >
              ADVANCED
            </button>
          </div>
          <div className="h-3 w-px bg-[#18181b]"></div>
          <div className="flex items-center gap-2">
            <span className={isRunning ? "text-[#3b82f6] animate-pulse" : "text-[#52525b]"}>
              {isRunning ? `TRAINING_PASS_${progress}%` : "READY_IDLE"}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-4">
            <ResourceBadge label="CPU" value={`${hardwareStats?.cpu_percent.toFixed(1) || 0}%`} />
            <ResourceBadge label="RAM" value={`${hardwareStats?.ram_percent.toFixed(0) || 0}%`} />
            <div className="h-3 w-px bg-[#18181b]"></div>
            <ResourceBadge label="GPU" value={`${hardwareStats?.gpu_use_percent || 0}%`} color="text-[#3b82f6]" />
            <ResourceBadge label="VRAM" value={`${hardwareStats?.vram_used_gb || 0} / ${hardwareStats?.vram_total_gb || 0} GB`} color="text-[#3b82f6]" />
          </div>
          <div className="h-3 w-px bg-[#18181b]"></div>
          <div className="flex items-center gap-2">
            <div className={`w-1.5 h-1.5 rounded-full ${socketRef.current?.readyState === WebSocket.OPEN ? 'bg-[#22c55e]' : 'bg-[#ef4444]'}`}></div>
            <span className="text-[#52525b] uppercase font-bold tracking-tighter">Connected</span>
          </div>
        </div>
      </footer>

      {/* COMMAND PALETTE */}
      {openCommand && (
        <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh] bg-black/60 backdrop-blur-[2px]">
          <div className="w-[640px] max-h-[400px]">
            <Command label="Command Palette" onKeyDown={(e) => e.key === "Escape" && setOpenCommand(false)}>
              <Command.Input placeholder="Search system actions (Ctrl+K)..." autoFocus className="w-full bg-[#18181b] border border-[#27272a] text-[#fafafa] px-4 py-3 rounded-t-lg outline-none" />
              <Command.List className="bg-[#0c0c0e] border-x border-b border-[#27272a] rounded-b-lg overflow-y-auto max-h-[300px] custom-scrollbar p-2">
                <Command.Empty className="p-4 text-[12px] text-[#52525b]">No results found.</Command.Empty>
                <Command.Group heading="Execution" className="text-[10px] text-[#52525b] uppercase font-bold px-2 py-1">
                  <CommandItem icon={Play} label="Execute Active Core" onSelect={() => { setOpenCommand(false); handleStartTraining(); }} />
                  <CommandItem icon={StopCircle} label="Terminate Running Pass" onSelect={() => { setOpenCommand(false); handleAbortTraining(); }} />
                </Command.Group>
                <Command.Group heading="Workspace" className="text-[10px] text-[#52525b] uppercase font-bold px-2 py-1 mt-2">
                  <CommandItem icon={Layers} label="Switch to Training" onSelect={() => { setOpenCommand(false); setActiveWorkspace("Train"); }} />
                  <CommandItem icon={Database} label="Open Library" onSelect={() => { setOpenCommand(false); setActiveWorkspace("Library"); }} />
                </Command.Group>
              </Command.List>
            </Command>
          </div>
          <div className="absolute inset-0 -z-10" onClick={() => setOpenCommand(false)}></div>
        </div>
      )}
    </div>
  );
}

/* WORKSPACE COMPONENTS */

function TrainWorkspace({ 
  metricsHistory, isRunning, progress, currentTrial, totalTrials, result, logs, clearLogs, logEndRef, hardwareStats,
  split, setSplit, loadedModelPath
}: any) {
  return (
    <div className="flex flex-col h-full bg-[#09090b]">
      <Tabs.Root defaultValue="optimization" className="flex flex-col h-full">
        <div className="h-12 border-b border-[#18181b] flex items-center justify-between px-6 bg-[#0c0c0e]/50 shrink-0">
          <Tabs.List className="flex gap-8">
            <TabTrigger value="optimization" label="Optimization" />
            <TabTrigger value="inference" label="Inference Playground" />
            <TabTrigger value="preview" label="Dataset Preview" />
          </Tabs.List>
          
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
              <span className="text-[10px] font-bold text-[#52525b] uppercase">Split</span>
              <div className="flex items-center gap-2">
                <input 
                  type="range" 
                  value={split} 
                  onChange={(e) => setSplit(parseInt(e.target.value))} 
                  min="50" max="95" step="5"
                  className="w-24 h-1 bg-[#18181b] rounded-full appearance-none accent-[#3b82f6]" 
                />
                <span className="text-[10px] font-mono text-[#3b82f6] w-8">{split}%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-hidden relative">
          <Tabs.Content value="optimization" className="h-full flex flex-col data-[state=inactive]:hidden">
            <div className="flex-1 overflow-y-auto custom-scrollbar p-6 space-y-6">
              {/* Dataset Summary Header */}
              <div className="grid grid-cols-3 gap-4">
                <SummaryCard icon={Table} label="Active Dataset" value="Hold-2 Predictors" subValue="3,204 Features" />
                <SummaryCard icon={Activity} label="Best R² Score" value={result?.best_r2?.toFixed(4) || "0.0000"} subValue="Optimization Peak" />
                <SummaryCard icon={Clock} label="Engine Status" value={isRunning ? "Active" : "Idle"} subValue={isRunning ? `Trial ${currentTrial}/${totalTrials}` : "Awaiting Run"} />
              </div>

              {/* Live Performance View */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-[11px] uppercase font-bold tracking-widest text-[#52525b]">Live Metrics History</h3>
                  <div className="flex items-center gap-4 text-[10px] font-mono">
                    <span className="text-[#a1a1aa] flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-[#ef4444]"></div> Loss</span>
                    <span className="text-[#a1a1aa] flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-[#3b82f6]"></div> R²</span>
                  </div>
                </div>
                <div className="border border-[#18181b] rounded-lg p-4 bg-[#0c0c0e]/50">
                  <ResponsiveContainer width="100%" height={240}>
                  <RechartsLineChart data={[...metricsHistory]}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#18181b" vertical={false} />
                    <XAxis dataKey="trial" hide />
                    <YAxis yAxisId="left" stroke="#52525b" tick={{ fill: "#71717a", fontSize: 10 }} />
                    <YAxis yAxisId="right" orientation="right" stroke="#52525b" tick={{ fill: "#71717a", fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "6px", fontSize: "10px" }} />
                    <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line yAxisId="right" type="monotone" dataKey="r2" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </RechartsLineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Recent Trials Table */}
              <div className="space-y-4">
                <h3 className="text-[11px] uppercase font-bold tracking-widest text-[#52525b]">Optimization Results</h3>
                <div className="border border-[#18181b] rounded-lg overflow-hidden bg-[#0c0c0e]/30">
                  <table className="w-full text-left text-[11px]">
                    <thead className="bg-[#0c0c0e] text-[#52525b] font-bold">
                      <tr>
                        <th className="px-4 py-2 border-b border-[#18181b]">TRIAL</th>
                        <th className="px-4 py-2 border-b border-[#18181b]">R²</th>
                        <th className="px-4 py-2 border-b border-[#18181b]">VAL_LOSS</th>
                        <th className="px-4 py-2 border-b border-[#18181b]">MAE</th>
                      </tr>
                    </thead>
                    <tbody className="text-[#a1a1aa] font-mono">
                      {metricsHistory.slice(-10).reverse().map((m: any, i: number) => (
                        <tr key={i} className="border-b border-[#18181b]/50 hover:bg-[#18181b]/30">
                          <td className="px-4 py-2 text-[#52525b]">#{m.trial}</td>
                          <td className="px-4 py-2 text-[#fafafa]">{m.r2.toFixed(4)}</td>
                          <td className="px-4 py-2">{m.val_loss.toFixed(6)}</td>
                          <td className="px-4 py-2">{m.mae.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* DOCKED BOTTOM LOGS */}
            <div className="h-48 border-t border-[#18181b] bg-[#0c0c0e] flex flex-col shrink-0">
              <div className="h-8 border-b border-[#18181b] flex items-center justify-between px-4 bg-[#09090b] shrink-0">
                <div className="flex items-center gap-2">
                  <Terminal size={12} className="text-[#3b82f6]" />
                  <span className="text-[10px] font-bold uppercase tracking-wider text-[#fafafa]">Process Stream</span>
                </div>
                <button onClick={clearLogs} className="text-[10px] text-[#52525b] hover:text-[#fafafa] flex items-center gap-1.5"><Trash2 size={10} /> Clear</button>
              </div>
              <div className="flex-1 overflow-y-auto p-3 font-mono text-[11px] leading-snug custom-scrollbar bg-[#09090b]">
                {logs.map((log: any, i: number) => (
                  <div key={i} className="flex gap-3 mb-0.5 group">
                    <span className="text-[#27272a] shrink-0 tabular-nums">[{log.time}]</span>
                    <span className={log.type === 'success' ? 'text-[#22c55e]' : log.type === 'warn' ? 'text-[#f59e0b]' : log.type === 'info' ? 'text-[#3b82f6]' : log.type === 'optuna' ? 'text-[#a855f7]' : 'text-[#71717a]'}>
                      {log.msg}
                    </span>
                  </div>
                ))}
                <div ref={logEndRef} />
              </div>
            </div>
          </Tabs.Content>

          <Tabs.Content value="inference" className="h-full overflow-y-auto custom-scrollbar p-10 data-[state=inactive]:hidden">
            <div className="max-w-3xl mx-auto">
              <InferencePlayground loadedPath={loadedModelPath} />
            </div>
          </Tabs.Content>

          <Tabs.Content value="preview" className="h-full overflow-y-auto custom-scrollbar p-6 data-[state=inactive]:hidden">
            <DatasetPreview />
          </Tabs.Content>
        </div>
      </Tabs.Root>
    </div>
  );
}

function LibraryWorkspace({ runs, onSelectRun, selectedRun }: any) {
  if (selectedRun) {
    return <RunDetailView run={selectedRun} onBack={() => onSelectRun(null)} />;
  }

  return (
    <div className="flex flex-col h-full bg-[#09090b]">
      <Tabs.Root defaultValue="history" className="flex flex-col h-full">
        <div className="h-12 border-b border-[#18181b] flex items-center px-6 bg-[#0c0c0e]/50 shrink-0">
          <Tabs.List className="flex gap-8">
            <TabTrigger value="history" label="Run History" />
            <TabTrigger value="datasets" label="Dataset Assets" />
            <TabTrigger value="models" label="Model Checkpoints" />
          </Tabs.List>
        </div>
        
        <div className="flex-1 overflow-hidden">
          <Tabs.Content value="history" className="h-full flex flex-col data-[state=inactive]:hidden">
            <div className="h-12 border-b border-[#18181b] flex items-center px-6 bg-[#0c0c0e]/30 gap-4">
              <div className="flex items-center gap-2 flex-1 max-w-sm">
                <Search size={14} className="text-[#52525b]" />
                <input type="text" placeholder="Filter run history..." className="bg-transparent border-none outline-none text-[11px] w-full placeholder-[#3f3f46]" />
              </div>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
              <div className="border border-[#18181b] rounded-lg overflow-hidden bg-[#0c0c0e]/20">
                <table className="w-full text-left text-[12px]">
                  <thead className="bg-[#0c0c0e] text-[#52525b] font-bold">
                    <tr>
                      <th className="px-6 py-3 border-b border-[#18181b]">RUN_ID</th>
                      <th className="px-6 py-3 border-b border-[#18181b]">MODEL</th>
                      <th className="px-6 py-3 border-b border-[#18181b]">STATUS</th>
                      <th className="px-6 py-3 border-b border-[#18181b]">R²</th>
                      <th className="px-6 py-3 border-b border-[#18181b]">DATE</th>
                    </tr>
                  </thead>
                  <tbody className="text-[#a1a1aa] font-mono">
                    {runs.map((run: any) => (
                      <tr 
                        key={run.id} 
                        onClick={() => onSelectRun(run)}
                        className="border-b border-[#18181b]/50 hover:bg-[#18181b]/30 cursor-pointer group"
                      >
                        <td className="px-6 py-4 text-[#3b82f6] group-hover:text-[#60a5fa] font-bold">#{run.run_id}</td>
                        <td className="px-6 py-4">{run.model_choice}</td>
                        <td className="px-6 py-4">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                            run.status === 'completed' ? 'bg-green-500/10 text-green-400' :
                            run.status === 'failed' ? 'bg-red-500/10 text-red-400' :
                            run.status === 'running' ? 'bg-blue-500/10 text-blue-400 animate-pulse' :
                            'bg-yellow-500/10 text-yellow-400'
                          }`}>
                            {run.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-[#fafafa]">{run.best_r2?.toFixed(4) || "—"}</td>
                        <td className="px-6 py-4 opacity-50">{new Date(run.timestamp).toLocaleDateString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </Tabs.Content>

          <Tabs.Content value="datasets" className="h-full overflow-y-auto custom-scrollbar p-6 data-[state=inactive]:hidden">
            <DatasetPreview />
          </Tabs.Content>

          <Tabs.Content value="models" className="h-full overflow-y-auto custom-scrollbar p-6 data-[state=inactive]:hidden">
             <div className="text-[11px] text-[#52525b] italic">No local checkpoints found in /home/nate/Desktop/NFTool/checkpoints</div>
          </Tabs.Content>
        </div>
      </Tabs.Root>
    </div>
  );
}

function RunDetailView({ run, onBack }: { run: any, onBack: () => void }) {
  // Assuming plots are saved with timestamps or run IDs in the filename
  // This is a bit tricky without knowing exact filename mapping from DB.
  // We'll use a placeholder logic or assume the report_path tells us.
  
  return (
    <div className="flex flex-col h-full bg-[#09090b]">
      <div className="h-12 border-b border-[#18181b] flex items-center justify-between px-6 bg-[#0c0c0e]/50 shrink-0">
        <div className="flex items-center gap-4">
          <button onClick={onBack} className="p-1.5 hover:bg-[#18181b] rounded text-[#52525b] hover:text-[#fafafa] transition-all">
            <ChevronRight size={16} className="rotate-180" />
          </button>
          <h2 className="text-[12px] font-bold text-[#fafafa] uppercase tracking-widest">Run Details: #{run.run_id}</h2>
        </div>
        <div className="flex items-center gap-2">
          {run.report_path && (
            <a 
              href={`http://localhost:8001/reports/${run.report_path}`} 
              target="_blank" 
              className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-[#fafafa] bg-[#3b82f6] hover:bg-[#2563eb] transition-all"
            >
              <FileText size={12} />
              Open HTML Report
            </a>
          )}
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto custom-scrollbar p-8 space-y-8">
        <div className="grid grid-cols-4 gap-6">
          <SummaryCard icon={Activity} label="Status" value={run.status.toUpperCase()} subValue={run.model_choice} />
          <SummaryCard icon={Activity} label="Best R²" value={run.best_r2?.toFixed(4) || "0.0000"} subValue="Optimization Peak" />
          <SummaryCard icon={Cpu} label="Trials" value={run.optuna_trials} subValue="Configured Budget" />
          <SummaryCard icon={Database} label="Date" value={new Date(run.timestamp).toLocaleDateString()} subValue={new Date(run.timestamp).toLocaleTimeString()} />
        </div>

        <div className="space-y-4">
          <h3 className="text-[11px] uppercase font-bold tracking-widest text-[#52525b]">Generated Analytics</h3>
          <div className="grid grid-cols-2 gap-6">
            <PlotCard title="Optimization History" src={`http://localhost:8001/results/optuna_optimization_history.png`} />
            <PlotCard title="Parameter Importances" src={`http://localhost:8001/results/optuna_param_importances.png`} />
            <PlotCard title="Pred vs Actual" src={`http://localhost:8001/results/r2_pred_vs_actual.png`} />
            <PlotCard title="Error Histogram" src={`http://localhost:8001/results/r2_error_histogram.png`} />
          </div>
        </div>
      </div>
    </div>
  );
}

function PlotCard({ title, src }: { title: string, src: string }) {
  return (
    <div className="border border-[#18181b] rounded-xl overflow-hidden bg-[#0c0c0e]/50 flex flex-col">
      <div className="px-4 py-2 border-b border-[#18181b] bg-[#0c0c0e]">
        <span className="text-[10px] font-bold uppercase tracking-widest text-[#52525b]">{title}</span>
      </div>
      <div className="flex-1 p-2 flex items-center justify-center min-h-[300px]">
        <img 
          src={src} 
          alt={title} 
          className="max-h-full max-w-full object-contain"
          onError={(e: any) => {
            e.target.src = "https://placehold.co/600x400/0c0c0e/52525b?text=Plot+Not+Found";
          }}
        />
      </div>
    </div>
  );
}

function Inspector({ 
  activeWorkspace, modelType, setModelType, seed, setSeed, patience, setPatience,
  trials, setTrials, lrRange, setLrRange, convBlocks, setConvBlocks, kernelSize, setKernelSize,
  layersRange, setLayersRange, layerSizeRange, setLayerSizeRange, dropoutRange, setDropoutRange,
  hDimRange, setHDimRange, maxEpochs, setMaxEpochs, gpuThrottle, setGpuThrottle, cnnFilterCap, setCnnFilterCap,
  datasets, selectedPredictor, setSelectedPredictor, selectedTarget, setSelectedTarget,
  isAdvancedMode
}: any) {
  return (
    <aside className="h-full flex flex-col bg-[#0c0c0e] border-l border-[#18181b]">
      <Tabs.Root defaultValue="model" className="flex flex-col flex-1 overflow-hidden">
        <div className="px-4 pt-4 shrink-0 bg-[#0c0c0e]">
          <Tabs.List className="flex border-b border-[#18181b] gap-6">
            <TabTrigger value="model" label="Model" />
            <TabTrigger value="perf" label="Performance" />
          </Tabs.List>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
          <Tabs.Content value="model" className="space-y-1">
            <Accordion.Root type="multiple" defaultValue={["dataset", "arch", "optuna"]} className="w-full">
              <InspectorSection value="dataset" title="Dataset Assets">
                <div className="space-y-4 pt-2">
                  <div className="space-y-1.5">
                    <label className="text-[9px] font-bold text-[#52525b] uppercase">Predictors (X)</label>
                    <select 
                      value={selectedPredictor} 
                      onChange={(e) => setSelectedPredictor(e.target.value)}
                      className="w-full bg-[#09090b] border border-[#18181b] rounded px-2.5 py-1.5 text-[11px] text-[#fafafa] focus:outline-none focus:border-[#3b82f6] transition-colors"
                    >
                      {datasets.map((d: any) => <option key={d.path} value={d.path}>{d.name}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-[9px] font-bold text-[#52525b] uppercase">Targets (y)</label>
                    <select 
                      value={selectedTarget} 
                      onChange={(e) => setSelectedTarget(e.target.value)}
                      className="w-full bg-[#09090b] border border-[#18181b] rounded px-2.5 py-1.5 text-[11px] text-[#fafafa] focus:outline-none focus:border-[#3b82f6] transition-colors"
                    >
                      {datasets.map((d: any) => <option key={d.path} value={d.path}>{d.name}</option>)}
                    </select>
                  </div>
                </div>
              </InspectorSection>

              <InspectorSection value="arch" title="Architecture">
                <div className="space-y-4 pt-2">
                  <div className="space-y-1.5">
                    <label className="text-[9px] font-bold text-[#52525b] uppercase">Base Class</label>
                    <div className="grid grid-cols-2 bg-[#09090b] border border-[#18181b] rounded p-0.5">
                      <button onClick={() => setModelType("NN")} className={`py-1 rounded text-[10px] font-bold transition-all ${modelType === "NN" ? "bg-[#18181b] text-[#fafafa]" : "text-[#52525b]"}`}>NN</button>
                      <button onClick={() => setModelType("CNN")} className={`py-1 rounded text-[10px] font-bold transition-all ${modelType === "CNN" ? "bg-[#18181b] text-[#fafafa]" : "text-[#52525b]"}`}>CNN</button>
                    </div>
                  </div>
                  
                  <ControlInput label="Random Seed" value={seed} onChange={(v: string) => setSeed(v)} />
                  {isAdvancedMode && <ControlInput label="Early Stop Patience" value={patience} onChange={(v: string) => setPatience(v)} />}
                  
                  {modelType === "CNN" && (
                    <ControlInput label="Kernel Size" value={kernelSize} onChange={(v: string) => setKernelSize(v)} />
                  )}
                </div>
              </InspectorSection>

              <InspectorSection value="optuna" title="Optuna Settings">
                <div className="space-y-4 pt-2">
                  <ControlInput label="Trial Budget" value={trials} onChange={(v: string) => setTrials(v)} />
                  {isAdvancedMode && <ControlInput label="LR Bounds" value={lrRange} onChange={(v: string) => setLrRange(v)} />}
                  
                  {modelType === "NN" ? (
                    <>
                      <ControlInput label="Layers (Range)" value={layersRange} onChange={(v: string) => setLayersRange(v)} />
                      <ControlInput label="Layer Size (Range)" value={layerSizeRange} onChange={(v: string) => setLayerSizeRange(v)} />
                    </>
                  ) : (
                    <>
                      <ControlInput label="Conv Blocks (Range)" value={convBlocks} onChange={(v: string) => setConvBlocks(v)} />
                      <ControlInput label="Hidden Dim (Range)" value={hDimRange} onChange={(v: string) => setHDimRange(v)} />
                    </>
                  )}
                  {isAdvancedMode && <ControlInput label="Dropout (Range)" value={dropoutRange} onChange={(v: string) => setDropoutRange(v)} />}
                </div>
              </InspectorSection>
            </Accordion.Root>
          </Tabs.Content>

          <Tabs.Content value="perf" className="space-y-1">
            <Accordion.Root type="multiple" defaultValue={["system"]} className="w-full">
              <InspectorSection value="system" title="Runtime Performance">
                <div className="space-y-4 pt-2">
                  <ControlInput label="Max Epochs" value={maxEpochs} onChange={(v: string) => setMaxEpochs(v)} />
                  {isAdvancedMode && (
                    <>
                      <ControlInput label="GPU Throttle (s)" value={gpuThrottle} onChange={(v: string) => setGpuThrottle(v)} />
                      <ControlInput label="CNN Filter Cap" value={cnnFilterCap} onChange={(v: string) => setCnnFilterCap(v)} />
                    </>
                  )}
                </div>
              </InspectorSection>
            </Accordion.Root>
          </Tabs.Content>
        </div>
      </Tabs.Root>
    </aside>
  );
}

/* HELPER SUB-COMPONENTS */

function SummaryCard({ icon: Icon, label, value, subValue }: any) {
  return (
    <div className="bg-[#0c0c0e] border border-[#18181b] rounded-lg p-4 flex items-center gap-4">
      <div className="w-10 h-10 rounded-lg bg-[#18181b] flex items-center justify-center text-[#3b82f6]">
        <Icon size={20} />
      </div>
      <div>
        <div className="text-[10px] font-bold text-[#52525b] uppercase tracking-wider">{label}</div>
        <div className="text-[16px] font-bold text-[#fafafa] leading-tight mt-0.5">{value}</div>
        <div className="text-[10px] text-[#52525b] font-mono mt-0.5">{subValue}</div>
      </div>
    </div>
  );
}

function HardwarePanel({ label, util, extra }: any) {
  return (
    <div className="bg-[#0c0c0e] border border-[#18181b] rounded-lg p-3">
      <div className="text-[9px] font-bold text-[#52525b] uppercase mb-2">{label}</div>
      <div className="flex items-end justify-between mb-1">
        <div className="text-[14px] font-bold text-[#fafafa] font-mono">{util}%</div>
        <div className="text-[9px] text-[#52525b] mb-0.5">{extra}</div>
      </div>
      <div className="h-1 w-full bg-[#18181b] rounded-full overflow-hidden">
        <div className="h-full bg-[#3b82f6] transition-all duration-1000" style={{ width: `${util}%` }}></div>
      </div>
    </div>
  );
}

function ResourceBadge({ label, value, color }: { label: string, value: string, color?: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[#52525b] font-bold">{label}</span>
      <span className={`font-mono ${color || 'text-[#fafafa]'}`}>{value}</span>
    </div>
  );
}

function NavIcon({ icon: Icon, active, onClick, tooltip }: { icon: any, active: boolean, onClick: () => void, tooltip: string }) {
  return (
    <button 
      onClick={onClick}
      className={`p-2.5 rounded-md transition-all relative group ${active ? "text-[#fafafa] bg-[#18181b]" : "text-[#52525b] hover:text-[#a1a1aa] hover:bg-[#18181b]/50"}`}
    >
      <Icon size={20} strokeWidth={active ? 2.5 : 2} />
      {active && <div className="absolute left-[-4px] top-2 bottom-2 w-[3px] bg-[#3b82f6] rounded-r-full"></div>}
      <div className="absolute left-[64px] bg-[#18181b] text-[#fafafa] text-[10px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none border border-[#27272a] shadow-xl z-50">
        {tooltip}
      </div>
    </button>
  );
}

function IconButton({ icon: Icon, tooltip, onClick }: any) {
  return (
    <button onClick={onClick} className="p-1.5 rounded hover:bg-[#18181b] text-[#a1a1aa] hover:text-[#fafafa] transition-all relative group">
      <Icon size={16} />
      <div className="absolute top-[32px] left-1/2 -translate-x-1/2 bg-[#18181b] text-[#fafafa] text-[9px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none border border-[#27272a] shadow-xl z-50">
        {tooltip}
      </div>
    </button>
  );
}

function TabTrigger({ value, label }: { value: string, label: string }) {
  return (
    <Tabs.Trigger value={value} className="text-[10px] font-bold uppercase tracking-[0.1em] text-[#52525b] data-[state=active]:text-[#fafafa] data-[state=active]:after:content-[''] data-[state=active]:after:block data-[state=active]:after:h-[2px] data-[state=active]:after:bg-[#3b82f6] data-[state=active]:after:mt-[2px] transition-colors outline-none pb-1">
      {label}
    </Tabs.Trigger>
  );
}

function InspectorSection({ value, title, children }: { value: string, title: string, children: React.ReactNode }) {
  return (
    <Accordion.Item value={value} className="border-b border-[#18181b]/50">
      <Accordion.Header className="flex">
        <Accordion.Trigger className="flex flex-1 items-center justify-between py-2 text-[10px] font-bold text-[#a1a1aa] hover:text-[#fafafa] uppercase tracking-wider transition-colors group outline-none">
          {title}
          <ChevronRight size={12} className="text-[#52525b] group-data-[state=open]:rotate-90 transition-transform" />
        </Accordion.Trigger>
      </Accordion.Header>
      <Accordion.Content className="pb-4 overflow-hidden data-[state=closed]:animate-slideUp data-[state=open]:animate-slideDown">
        {children}
      </Accordion.Content>
    </Accordion.Item>
  );
}

function ControlInput({ label, value, onChange }: { label: string, value: string, onChange?: (v: string) => void }) {
  return (
    <div className="space-y-1.5">
      <label className="text-[9px] font-bold text-[#52525b] uppercase">{label}</label>
      <input type="text" value={value} onChange={(e) => onChange?.(e.target.value)} className="w-full bg-[#09090b] border border-[#18181b] rounded px-2.5 py-1.5 text-[11px] text-[#fafafa] focus:outline-none focus:border-[#3b82f6] transition-colors placeholder-[#3f3f46]" />
    </div>
  );
}

function CommandItem({ icon: Icon, label, onSelect }: { icon: any, label: string, onSelect?: () => void }) {
  return (
    <Command.Item onSelect={() => onSelect?.()} className="flex items-center gap-3 px-4 py-2 text-[13px] cursor-default select-none aria-selected:bg-[#18181b] aria-selected:text-[#fafafa] rounded-md mx-2 transition-colors outline-none">
      <Icon size={16} className="text-[#52525b]" />
      {label}
    </Command.Item>
  );
}

function DatasetPreview() {
  const [previewData, setPreviewData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [filePath, setFilePath] = useState("/home/nate/Desktop/NFTool/dataset/Predictors_2025-04-15_10-43_Hold-2.csv");

  const loadPreview = async () => {
    if (!filePath) return;
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8001/dataset/preview?path=${encodeURIComponent(filePath)}&rows=20`);
      const data = await res.json();
      setPreviewData(data);
    } catch (e) {
      console.error("Failed to load preview:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (filePath) loadPreview();
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 bg-[#0c0c0e] p-3 rounded-lg border border-[#18181b]">
        <FolderOpen size={16} className="text-[#3b82f6]" />
        <input
          type="text"
          value={filePath}
          onChange={(e) => setFilePath(e.target.value)}
          placeholder="Dataset path..."
          className="flex-1 bg-transparent border-none outline-none text-[11px] text-[#fafafa] placeholder-[#3f3f46]"
        />
        <button
          onClick={loadPreview}
          disabled={loading}
          className="px-4 py-1.5 bg-[#3b82f6]/10 text-[#3b82f6] border border-[#3b82f6]/30 hover:bg-[#3b82f6] hover:text-[#fafafa] text-[11px] font-bold rounded transition-all disabled:opacity-50"
        >
          {loading ? "Loading..." : "Preview"}
        </button>
      </div>
      {previewData && (
        <div className="border border-[#18181b] rounded-lg overflow-hidden bg-[#0c0c0e]/30">
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full text-left border-collapse">
              <thead className="bg-[#0c0c0e] text-[10px] text-[#52525b] uppercase font-bold">
                <tr>
                  {previewData.headers.map((h: string, i: number) => (
                    <th key={i} className="px-3 py-2 border-b border-[#18181b] font-mono whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="text-[11px] text-[#a1a1aa] font-mono">
                {previewData.rows.map((row: any[], i: number) => (
                  <tr key={i} className="hover:bg-[#18181b]/30 group transition-colors">
                    {row.map((cell: any, j: number) => (
                      <td key={j} className="px-3 py-1.5 border-b border-[#18181b]/50 group-hover:text-[#fafafa] whitespace-nowrap">{typeof cell === 'number' ? cell.toFixed(4) : cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function InferencePlayground({ loadedPath }: { loadedPath: string | null }) {
  const [features, setFeatures] = useState("");
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runInference = async () => {
    if (!loadedPath || !features) {
      setError("Please load model weights first");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const featureArray = features.split(",").map(f => parseFloat(f.trim()));
      const res = await fetch("http://localhost:8001/inference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: loadedPath, features: featureArray })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Inference failed");
      }
      const data = await res.json();
      setPrediction(data.prediction);
    } catch (e: any) {
      setError(e.message || "Inference error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-[#0c0c0e]/50 border border-[#18181b] rounded-xl p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-[#3b82f6]/10 flex items-center justify-center text-[#3b82f6]">
              <FastForward size={18} />
            </div>
            <div>
              <h3 className="text-[12px] font-bold text-[#fafafa]">Prediction Engine</h3>
              <p className="text-[10px] text-[#52525b]">Execute single-row inference on active weights</p>
            </div>
          </div>
          {loadedPath ? (
            <div className="flex items-center gap-2 px-3 py-1 bg-green-500/10 border border-green-500/30 rounded text-[10px] font-bold text-green-400 uppercase">
              <Check size={12} /> Live
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1 bg-red-500/10 border border-red-500/30 rounded text-[10px] font-bold text-red-400 uppercase">
              <AlertCircle size={12} /> Weights Missing
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-[10px] font-bold text-[#52525b] uppercase tracking-widest">Input Features (CSV Vector)</label>
            <textarea
              value={features}
              onChange={(e) => setFeatures(e.target.value)}
              placeholder="Enter comma-separated features (e.g., 0.12, 0.45, -0.11...)"
              rows={4}
              className="w-full bg-[#09090b] border border-[#18181b] rounded-lg px-4 py-3 text-[12px] text-[#fafafa] font-mono focus:outline-none focus:border-[#3b82f6] placeholder-[#3f3f46] transition-all"
            />
          </div>
          <button
            onClick={runInference}
            disabled={loading || !loadedPath}
            className="w-full py-2.5 bg-[#3b82f6] hover:bg-[#2563eb] text-[#fafafa] text-[11px] font-bold rounded-lg transition-all disabled:opacity-50 disabled:grayscale shadow-lg shadow-[#3b82f6]/10"
          >
            {loading ? "COMPUTING..." : "EXECUTE INFERENCE"}
          </button>
        </div>

        {error && (
          <div className="text-[11px] text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 flex items-center gap-3">
            <AlertCircle size={14} />
            {error}
          </div>
        )}

        {prediction !== null && (
          <div className="bg-[#18181b] border border-[#27272a] rounded-lg p-5 transition-all animate-in fade-in slide-in-from-bottom-2">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-bold text-[#52525b] uppercase tracking-widest">Inference Output</span>
              <span className="text-[10px] text-[#3b82f6] font-mono">FLOAT64</span>
            </div>
            <div className="text-[32px] font-mono text-[#3b82f6] tracking-tighter tabular-nums leading-none">
              {prediction.toFixed(8)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Additional missing icon components for TrainWorkspace
function Clock(props: any) { return <HistoryIcon {...props} /> }

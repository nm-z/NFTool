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
import { 
  Activity, 
  Settings, 
  Play, 
  FileCode, 
  Database, 
  LineChart, 
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
  AlertCircle
} from "lucide-react";
import { useTrainingStore } from "@/store/useTrainingStore";

export default function Dashboard() {
  const { 
    isRunning, 
    progress, 
    currentTrial,
    totalTrials,
    logs, 
    result, 
    setIsRunning, 
    setProgress, 
    setTrialInfo,
    addLog, 
    setResult, 
    clearLogs 
  } = useTrainingStore();

  const [activeWorkspace, setActiveWorkspace] = useState("Architectures");
  const [modelType, setModelType] = useState("NN");
  const [isAdvanced, setIsAdvanced] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [openCommand, setOpenCommand] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);
  const socketRef = useRef<WebSocket | null>(null);

  // Inspector States
  const [seed, setSeed] = useState("15557");
  const [patience, setPatience] = useState("100");
  const [trials, setTrials] = useState("10");
  const [lrRange, setLrRange] = useState("1e-4 → 1e-3");
  const [split, setSplit] = useState(70);
  
  // CNN Specific Inspector States
  const [convBlocks, setConvBlocks] = useState("1 → 3");
  const [kernelSize, setKernelSize] = useState("3");

  // WebSocket Handshake
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket("ws://localhost:8001/ws");
      socketRef.current = ws;
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "init") {
          setIsRunning(msg.data.is_running);
          setProgress(msg.data.progress);
          setTrialInfo(msg.data.current_trial, msg.data.total_trials);
          setResult(msg.data.result);
        } else if (msg.type === "status") {
          setIsRunning(msg.data.is_running);
          setProgress(msg.data.progress);
          setTrialInfo(msg.data.current_trial, msg.data.total_trials);
          if (msg.data.result) setResult(msg.data.result);
        } else if (msg.type === "log") {
          addLog(msg.data);
        }
      };
      ws.onclose = () => setTimeout(connect, 2000);
    };
    connect();
    return () => socketRef.current?.close();
  }, []);

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
    const config = {
      model_choice: modelType, seed: parseInt(seed), patience: parseInt(patience),
      train_ratio: split / 100, val_ratio: (100 - split) / 2 / 100, test_ratio: (100 - split) / 2 / 100,
      optuna_trials: parseInt(trials), optimizers: ["AdamW"],
      n_layers_min: 1, n_layers_max: 8, l_size_min: 128, l_size_max: 1024,
      lr_min: 1e-4, lr_max: 1e-3, drop_min: 0.0, drop_max: 0.0,
      h_dim_min: 1, h_dim_max: 100,
      conv_blocks_min: parseInt(convBlocks.split("→")[0]) || 1,
      conv_blocks_max: parseInt(convBlocks.split("→")[1]) || 5,
      kernel_size: parseInt(kernelSize) || 3,
      predictor_path: "/home/nate/Desktop/NFTool/dataset/Predictors_2025-04-15_10-43_Hold-2.csv",
      target_path: "/home/nate/Desktop/NFTool/dataset/9_10_24_Hold_02_targets.csv"
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
    <div className="flex h-screen bg-[#09090b] text-[#a1a1aa] selection:bg-[#3b82f6]/20 font-sans">
      
      {/* LEFT NAVIGATION: Narrow Icon Rail */}
      <aside className="w-[56px] border-r border-[#18181b] bg-[#0c0c0e] flex flex-col items-center py-4 gap-4 z-40">
        <div className="w-8 h-8 bg-[#3b82f6]/10 rounded flex items-center justify-center text-[#3b82f6] mb-2">
          <Activity size={18} strokeWidth={2.5} />
        </div>
        <div className="flex-1 w-full flex flex-col items-center gap-1">
          <NavIcon icon={LayoutDashboard} active={activeWorkspace === "Dashboard"} onClick={() => setActiveWorkspace("Dashboard")} tooltip="Dashboard" />
          <NavIcon icon={Layers} active={activeWorkspace === "Architectures"} onClick={() => setActiveWorkspace("Architectures")} tooltip="Architectures" />
          <NavIcon icon={Database} active={activeWorkspace === "Datasets"} onClick={() => setActiveWorkspace("Datasets")} tooltip="Datasets" />
          <NavIcon icon={HistoryIcon} active={activeWorkspace === "History"} onClick={() => setActiveWorkspace("History")} tooltip="Run History" />
        </div>
        <NavIcon icon={Settings} active={false} onClick={() => {}} tooltip="Global Settings" />
      </aside>

      {/* MAIN SHELL */}
      <div className="flex-1 flex flex-col min-w-0">
        
        {/* TOP BAR */}
        <header className="h-[48px] border-b border-[#18181b] flex items-center justify-between px-4 bg-[#09090b] z-30">
          <div className="flex items-center gap-4">
            <span className="text-[#fafafa] font-medium text-[13px]">{activeWorkspace}</span>
            <div className="h-3 w-px bg-[#18181b]"></div>
            <div className="flex items-center gap-2">
              <StatusChip icon={Cpu} label="ROCm GPU" />
              <StatusChip icon={Box} label="3,204 Feat" />
              {isRunning && <StatusChip icon={RefreshCw} label={`${progress}%`} active />}
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <div className="flex items-center bg-[#18181b] rounded-md p-0.5 border border-[#27272a] mr-2">
              <button 
                onClick={handleStartTraining} 
                disabled={isRunning}
                className={`flex items-center gap-2 px-3 py-1 rounded text-[11px] font-medium transition-all ${isRunning ? 'text-[#52525b] cursor-default' : 'text-[#fafafa] bg-[#3b82f6] hover:bg-[#2563eb]'}`}
              >
                {isRunning ? <RefreshCw size={12} className="animate-spin" /> : <Play size={12} fill="currentColor" />}
                {isRunning ? "Optimizing..." : "Start Run"}
              </button>
              <Separator.Root orientation="vertical" className="mx-1 h-4 bg-[#27272a]" />
              <IconButton icon={StopCircle} disabled={!isRunning} onClick={handleAbortTraining} tooltip="Abort Run" color="hover:text-red-500" />
            </div>
            <IconButton icon={Upload} tooltip="Import Dataset" />
            <IconButton icon={Download} tooltip="Export Report" />
            <div className="h-4 w-px bg-[#18181b] mx-1"></div>
            <span className="text-[10px] font-mono text-[#52525b] px-2">Ctrl+K</span>
          </div>
        </header>

        {/* 3-PANE CONTENT GROUP */}
        <PanelGroup direction="horizontal" className="flex-1">
          
          {/* CENTER WORKSPACE: List-Detail Pattern */}
          <Panel defaultSize={75} minSize={40}>
            <div className="flex flex-col h-full bg-[#09090b]">
              {/* Workspace Header: Search & Filter */}
              <div className="h-10 border-b border-[#18181b] flex items-center justify-between px-4 bg-[#0c0c0e]/50">
                <div className="flex items-center gap-3 flex-1 max-w-md">
                  <Search size={14} className="text-[#52525b]" />
                  <input type="text" placeholder="Filter models or datasets..." className="bg-transparent border-none outline-none text-[12px] w-full placeholder-[#3f3f46]" />
                </div>
                <div className="flex items-center gap-2">
                  <button className="flex items-center gap-1.5 px-2 py-1 hover:bg-[#18181b] rounded text-[11px] transition-colors"><Filter size={12} /> Filter</button>
                  <button className="flex items-center gap-1.5 px-2 py-1 bg-[#18181b] hover:bg-[#27272a] text-[#fafafa] rounded text-[11px] transition-colors"><Plus size={12} /> New Run</button>
                </div>
              </div>

              {/* Primary Content Scroll Area */}
              <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
                <div className="max-w-5xl mx-auto space-y-8">
                  
                  {/* Active Job Row (If running) */}
                  {isRunning && (
                    <div className="border border-[#3b82f6]/30 bg-[#3b82f6]/5 rounded-lg p-4 flex items-center gap-6">
                      <div className="flex-1 space-y-2">
                        <div className="flex justify-between text-[12px]">
                          <span className="text-[#fafafa] font-medium italic">NFtool Optimization Pass #1</span>
                          <span className="font-mono">{progress}%</span>
                        </div>
                        <div className="h-1 w-full bg-[#18181b] rounded-full overflow-hidden">
                          <div className="h-full bg-[#3b82f6] transition-all duration-500" style={{ width: `${progress}%` }}></div>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
                        <span className="text-[#52525b]">Trials</span> <span className="text-right text-[#fafafa] font-mono">{currentTrial} / {totalTrials}</span>
                        <span className="text-[#52525b]">Best R²</span> <span className="text-right text-[#3b82f6] font-mono">{result?.best_r2?.toFixed(4) || "0.0000"}</span>
                      </div>
                    </div>
                  )}

                  {/* Data Table Mockup */}
                  <div className="space-y-4">
                    <h3 className="text-[12px] uppercase font-bold tracking-widest text-[#52525b]">Recent Model Trials</h3>
                    <div className="border border-[#18181b] rounded-lg overflow-hidden">
                      <table className="w-full text-left border-collapse">
                        <thead className="bg-[#0c0c0e] text-[11px] text-[#52525b] uppercase font-bold">
                          <tr>
                            <th className="px-4 py-2 border-b border-[#18181b]">ID</th>
                            <th className="px-4 py-2 border-b border-[#18181b]">Architecture</th>
                            <th className="px-4 py-2 border-b border-[#18181b]">R² Score</th>
                            <th className="px-4 py-2 border-b border-[#18181b]">Loss</th>
                            <th className="px-4 py-2 border-b border-[#18181b]">Duration</th>
                            <th className="px-4 py-2 border-b border-[#18181b]"></th>
                          </tr>
                        </thead>
                        <tbody className="text-[12px] text-[#a1a1aa]">
                          {logs.filter(l => l.type === 'optuna').slice(-5).reverse().map((trial, idx) => (
                            <tr key={idx} className="hover:bg-[#18181b]/30 group transition-colors">
                              <td className="px-4 py-2 border-b border-[#18181b] font-mono text-[11px]">#{logs.length - idx}</td>
                              <td className="px-4 py-2 border-b border-[#18181b]">{modelType}</td>
                              <td className="px-4 py-2 border-b border-[#18181b] text-[#fafafa] font-mono">{trial.msg.split("R²: ")[1] || "0.0000"}</td>
                              <td className="px-4 py-2 border-b border-[#18181b] font-mono">--</td>
                              <td className="px-4 py-2 border-b border-[#18181b]">1.2s</td>
                              <td className="px-4 py-2 border-b border-[#18181b] text-right"><IconButton icon={MoreVertical} tooltip="Trial Details" /></td>
                            </tr>
                          ))}
                          {logs.filter(l => l.type === 'optuna').length === 0 && (
                            <tr>
                              <td colSpan={6} className="px-4 py-12 text-center text-[#3f3f46] italic">No active optimization data available.</td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>

              {/* DOCKED BOTTOM PANEL: Metrics & Logs */}
              <div className="h-48 border-t border-[#18181b] bg-[#0c0c0e]">
                <Tabs.Root defaultValue="logs" className="flex flex-col h-full">
                  <div className="h-8 border-b border-[#18181b] flex items-center justify-between px-4 bg-[#09090b]">
                    <Tabs.List className="flex gap-4">
                      <TabTrigger value="logs" label="Process Stream" />
                      <TabTrigger value="plots" label="Live Plot" />
                      <TabTrigger value="system" label="Hardware" />
                    </Tabs.List>
                    <div className="flex items-center gap-3">
                      <button onClick={clearLogs} className="text-[10px] text-[#52525b] hover:text-[#fafafa] flex items-center gap-1.5"><Trash2 size={10} /> Clear</button>
                      <IconButton icon={MoreVertical} tooltip="Stream Settings" size={12} />
                    </div>
                  </div>
                  <Tabs.Content value="logs" className="flex-1 overflow-y-auto p-3 font-mono text-[11px] leading-snug custom-scrollbar bg-[#09090b]">
                    {logs.map((log, i) => (
                      <div key={i} className="flex gap-3 mb-0.5 group">
                        <span className="text-[#27272a] shrink-0 tabular-nums">[{log.time}]</span>
                        <span className={log.type === 'success' ? 'text-[#22c55e]' : log.type === 'warn' ? 'text-[#f59e0b]' : log.type === 'info' ? 'text-[#3b82f6]' : log.type === 'optuna' ? 'text-[#a855f7]' : 'text-[#71717a]'}>
                          {log.msg}
                        </span>
                      </div>
                    ))}
                    <div ref={logEndRef} />
                  </Tabs.Content>
                </Tabs.Root>
              </div>
            </div>
          </Panel>

          <PanelResizeHandle className="w-px bg-[#18181b] hover:bg-[#3b82f6]/50 transition-colors" />

          {/* RIGHT INSPECTOR: Configuration */}
          <Panel defaultSize={25} minSize={20}>
            <aside className="h-full flex flex-col bg-[#0c0c0e] border-l border-[#18181b]">
              <div className="h-10 border-b border-[#18181b] flex items-center justify-between px-4 bg-[#09090b]">
                <div className="flex items-center gap-2">
                  <span className="text-[#fafafa] font-medium text-[11px] uppercase tracking-wider">Inspector</span>
                  {hasUnsavedChanges && <div className="w-1.5 h-1.5 rounded-full bg-[#3b82f6] shadow-[0_0_8px_rgba(59,130,246,0.5)]"></div>}
                </div>
                <div className="flex items-center gap-2">
                  <button disabled={!hasUnsavedChanges} className="text-[10px] text-[#52525b] hover:text-[#fafafa] disabled:opacity-30">Revert</button>
                  <button onClick={() => setHasUnsavedChanges(false)} disabled={!hasUnsavedChanges} className="text-[10px] text-[#3b82f6] font-bold disabled:opacity-30">Apply</button>
                </div>
              </div>

              {/* Preset Selector */}
              <div className="p-3 pane-border-b bg-[#0c0c0e]">
                <label className="text-[9px] font-bold text-[#52525b] uppercase mb-1.5 block">Config Preset</label>
                <div className="flex items-center justify-between px-2 py-1.5 bg-[#09090b] border border-[#18181b] rounded text-[11px] text-[#fafafa] cursor-default hover:border-[#27272a]">
                  <span>Gold Standard MLP</span>
                  <ChevronDown size={14} className="text-[#52525b]" />
                </div>
              </div>

              <div className="flex-1 overflow-y-auto custom-scrollbar">
                <Accordion.Root type="multiple" defaultValue={["model", "optim"]} className="w-full">
                  <InspectorSection value="model" title="Model Definition">
                    <div className="space-y-4 pt-2">
                      <ControlGroup label="Architecture Class">
                        <div className="flex bg-[#09090b] border border-[#18181b] rounded p-0.5">
                          {["NN", "CNN"].map((t) => (
                            <button key={t} onClick={() => { setModelType(t); setHasUnsavedChanges(true); }} className={`flex-1 py-1 rounded text-[10px] font-bold transition-colors ${modelType === t ? "bg-[#18181b] text-[#fafafa]" : "text-[#52525b]"}`}>{t === "NN" ? "Multi-Layer" : "Conv-Net"}</button>
                          ))}
                        </div>
                      </ControlGroup>
                      <ControlInput label="Random Seed" value={seed} onChange={(v) => { setSeed(v); setHasUnsavedChanges(true); }} />
                      <ControlInput label="Early Stop Patience" value={patience} onChange={(v) => { setPatience(v); setHasUnsavedChanges(true); }} />
                      
                      {modelType === "CNN" && (
                        <div className="space-y-4 pt-2 border-t border-[#18181b] mt-4">
                          <label className="text-[9px] font-bold text-[#3b82f6] uppercase">CNN Specific</label>
                          <ControlInput label="Conv Blocks (Range)" value={convBlocks} onChange={(v) => { setConvBlocks(v); setHasUnsavedChanges(true); }} />
                          <ControlInput label="Kernel Size" value={kernelSize} onChange={(v) => { setKernelSize(v); setHasUnsavedChanges(true); }} />
                        </div>
                      )}
                    </div>
                  </InspectorSection>

                  <InspectorSection value="optim" title="Optimization">
                    <div className="space-y-4 pt-2">
                      <ControlInput label="Optuna Trial Budget" value={trials} onChange={(v) => { setTrials(v); setHasUnsavedChanges(true); }} />
                      <ControlInput label="Learning Rate Bound" value={lrRange} onChange={(v) => { setLrRange(v); setHasUnsavedChanges(true); }} />
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <label className="text-[10px] font-bold text-[#52525b] uppercase">Train Split</label>
                          <span className="text-[11px] font-mono text-[#3b82f6]">{split}%</span>
                        </div>
                        <input className="w-full h-1 bg-[#18181b] rounded-full appearance-none cursor-pointer accent-[#3b82f6]" type="range" value={split} onChange={(e) => { setSplit(parseInt(e.target.value)); setHasUnsavedChanges(true); }} min="50" max="90" />
                      </div>
                    </div>
                  </InspectorSection>

                  {isAdvanced && (
                    <InspectorSection value="advanced" title="Advanced Tensors">
                      <div className="space-y-4 pt-2">
                        <div className="flex items-center justify-between">
                          <label className="text-[10px] font-bold text-[#52525b] uppercase">Weight Decay</label>
                          <div className="w-8 h-4 bg-[#18181b] border border-[#27272a] rounded-full flex items-center px-0.5"><div className="w-2.5 h-2.5 bg-[#52525b] rounded-full"></div></div>
                        </div>
                        <ControlInput label="Dropout Min" value="0.0" />
                        <ControlInput label="Alpha Scale" value="0.0" />
                      </div>
                    </InspectorSection>
                  )}
                </Accordion.Root>
              </div>

              {/* Advanced Toggle Footer */}
              <div className="p-4 border-t border-[#18181b] bg-[#09090b]">
                <button 
                  onClick={() => setIsAdvanced(!isAdvanced)}
                  className="flex items-center gap-2 text-[10px] font-bold text-[#52525b] hover:text-[#a1a1aa] transition-colors"
                >
                  <Settings size={12} />
                  {isAdvanced ? "HIDE ADVANCED" : "SHOW ADVANCED SETTINGS"}
                </button>
              </div>
            </aside>
          </Panel>
        </PanelGroup>
      </div>

      {/* COMMAND PALETTE DIALOG */}
      {openCommand && (
        <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] bg-black/60 backdrop-blur-[2px]">
          <div className="w-[640px] max-h-[400px]">
            <Command label="Command Palette" onKeyDown={(e) => e.key === "Escape" && setOpenCommand(false)}>
              <Command.Input placeholder="Search system actions..." autoFocus />
              <Command.List className="custom-scrollbar">
                <Command.Empty className="p-4 text-[12px] text-[#52525b]">No results found.</Command.Empty>
                <Command.Group heading="Training Engine">
                  <CommandItem icon={Play} label="Execute Active Architecture" onSelect={handleStartTraining} />
                  <CommandItem icon={StopCircle} label="Terminate Running Core" onSelect={handleAbortTraining} />
                </Command.Group>
                <Command.Group heading="Workspace">
                  <CommandItem icon={LayoutDashboard} label="Go to Dashboard" onSelect={() => setActiveWorkspace("Dashboard")} />
                  <CommandItem icon={HistoryIcon} label="View Run History" onSelect={() => setActiveWorkspace("History")} />
                  <CommandItem icon={Trash2} label="Clear Engine Logs" onSelect={clearLogs} />
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

/* SUB-COMPONENTS */

function NavIcon({ icon: Icon, active, onClick, tooltip }: any) {
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

function StatusChip({ icon: Icon, label, active }: any) {
  return (
    <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded border text-[10px] font-medium tracking-tight ${active ? 'bg-[#3b82f6]/10 border-[#3b82f6]/30 text-[#3b82f6]' : 'border-[#18181b] bg-[#0c0c0e] text-[#52525b]'}`}>
      <Icon size={10} className={active ? 'animate-pulse' : ''} />
      <span>{label}</span>
    </div>
  );
}

function IconButton({ icon: Icon, tooltip, disabled, color, size = 14, onClick }: any) {
  return (
    <button onClick={onClick} disabled={disabled} className={`p-1.5 rounded hover:bg-[#18181b] transition-colors disabled:opacity-20 relative group ${color || 'text-[#a1a1aa]'}`}>
      <Icon size={size} />
      <div className="absolute top-[32px] left-1/2 -translate-x-1/2 bg-[#18181b] text-[#fafafa] text-[9px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none border border-[#27272a] shadow-xl z-50">
        {tooltip}
      </div>
    </button>
  );
}

function TabTrigger({ value, label }: any) {
  return (
    <Tabs.Trigger value={value} className="text-[10px] font-bold uppercase tracking-[0.1em] text-[#52525b] data-[state=active]:text-[#fafafa] data-[state=active]:after:content-[''] data-[state=active]:after:block data-[state=active]:after:h-[2px] data-[state=active]:after:bg-[#3b82f6] data-[state=active]:after:mt-[2px] transition-colors">
      {label}
    </Tabs.Trigger>
  );
}

function InspectorSection({ value, title, children }: any) {
  return (
    <Accordion.Item value={value} className="border-b border-[#18181b]">
      <Accordion.Header className="flex">
        <Accordion.Trigger className="flex flex-1 items-center justify-between px-4 py-2.5 text-[10px] font-bold text-[#fafafa] uppercase tracking-wider hover:bg-[#18181b]/30 transition-colors group">
          {title}
          <ChevronRight size={12} className="text-[#52525b] group-data-[state=open]:rotate-90 transition-transform" />
        </Accordion.Trigger>
      </Accordion.Header>
      <Accordion.Content className="px-4 pb-4 overflow-hidden data-[state=closed]:animate-slideUp data-[state=open]:animate-slideDown">
        {children}
      </Accordion.Content>
    </Accordion.Item>
  );
}

function ControlGroup({ label, children }: any) {
  return (
    <div className="space-y-1.5">
      <label className="text-[9px] font-bold text-[#52525b] uppercase">{label}</label>
      {children}
    </div>
  );
}

function ControlInput({ label, value, onChange }: any) {
  return (
    <div className="space-y-1.5">
      <label className="text-[9px] font-bold text-[#52525b] uppercase">{label}</label>
      <input type="text" value={value} onChange={(e) => onChange?.(e.target.value)} className="w-full bg-[#09090b] border border-[#18181b] rounded px-2.5 py-1.5 text-[11px] text-[#fafafa] focus:outline-none focus:border-[#3b82f6] transition-colors placeholder-[#3f3f46]" />
    </div>
  );
}

function CommandItem({ icon: Icon, label, onSelect }: any) {
  return (
    <Command.Item onSelect={() => onSelect?.()} className="flex items-center gap-3 px-4 py-2 text-[13px] cursor-default select-none aria-selected:bg-[#18181b] aria-selected:text-[#fafafa] rounded-md mx-2 transition-colors">
      <Icon size={16} className="text-[#52525b]" />
      {label}
    </Command.Item>
  );
}

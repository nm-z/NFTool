import React from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Activity, Table, History as HistoryIcon, Terminal, Trash2 } from "lucide-react";
import { Panel, Group as PanelGroup, Separator as PanelResizeHandle } from "react-resizable-panels";

import { SummaryCard } from "../common/Cards";
import { TabTrigger } from "../common/UIComponents";
import { DatasetPreview } from "./tools/DatasetPreview";
import { InferencePlayground } from "./tools/InferencePlayground";

interface TrainWorkspaceProps {
  metricsHistory: any[];
  isRunning: boolean;
  progress: number;
  currentTrial: number;
  totalTrials: number;
  result: any;
  logs: any[];
  clearLogs: () => void;
  logEndRef: React.RefObject<HTMLDivElement | null>;
  hardwareStats: any;
  split: number;
  setSplit: (v: number) => void;
  loadedModelPath: string | null;
  selectedPredictor: string;
}

export function TrainWorkspace({
  metricsHistory, isRunning, progress, currentTrial, totalTrials, result, logs, clearLogs, logEndRef, hardwareStats,
  split, setSplit, loadedModelPath, selectedPredictor
}: TrainWorkspaceProps) {
  return (
    <div className="flex flex-col h-full bg-[hsl(var(--background))] outline-none" tabIndex={-1}>
      <Tabs.Root defaultValue="optimization" className="flex flex-col h-full">
        <div className="h-12 border-b border-[hsl(var(--border))] flex items-center justify-between px-6 bg-[hsl(var(--panel))]/50 shrink-0">
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
                  className="w-24 h-1 bg-[hsl(var(--panel-lighter))] rounded-full appearance-none accent-[#3b82f6]" 
                />
                <span className="text-[10px] font-mono text-[#3b82f6] w-8">{split}%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-hidden relative">
          <Tabs.Content value="optimization" className="h-full flex flex-col data-[state=inactive]:hidden">
            <PanelGroup orientation="vertical">
              {/* TOP CONTENT AREA */}
              <Panel defaultSize={70} minSize={30}>
                <div className="h-full overflow-y-auto custom-scrollbar p-6 space-y-6">
                  {/* Dataset Summary Header */}
                  <div className="grid grid-cols-3 gap-4">
                    <SummaryCard icon={Table} label="Active Dataset" value="Hold-2 Predictors" subValue="3,204 Features" />
                    <SummaryCard icon={Activity} label="Best R² Score" value={result?.best_r2?.toFixed(4) || "0.0000"} subValue="Optimization Peak" />
                    <SummaryCard icon={HistoryIcon} label="Engine Status" value={isRunning ? "Active" : "Idle"} subValue={isRunning ? `Trial ${currentTrial}/${totalTrials}` : "Awaiting Run"} />
                  </div>

                  {/* Live Performance View */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-[11px] uppercase font-bold tracking-widest text-[#52525b]">Live Metrics History</h3>
                      <div className="flex items-center gap-4 text-[10px] font-mono">
                        <span className="text-[hsl(var(--foreground))] flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-[#ef4444]"></div> Loss</span>
                        <span className="text-[hsl(var(--foreground))] flex items-center gap-1.5"><div className="w-2 h-2 rounded-full bg-[#3b82f6]"></div> R²</span>
                      </div>
                    </div>
                    <div className="border border-[hsl(var(--border))] rounded-lg p-4 bg-[#000000]">
                      <ResponsiveContainer width="100%" height={240}>
                      <RechartsLineChart data={metricsHistory} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#18181b" vertical={false} />
                        <XAxis dataKey="trial" hide />
                        <YAxis yAxisId="left" stroke="#3f3f46" tick={{ fill: "#52525b", fontSize: 9, fontWeight: 600 }} tickLine={false} axisLine={false} />
                        <YAxis yAxisId="right" orientation="right" stroke="#3f3f46" tick={{ fill: "#52525b", fontSize: 9, fontWeight: 600 }} tickLine={false} axisLine={false} />
                        <Tooltip contentStyle={{ backgroundColor: "#09090b", border: "1px solid #27272a", borderRadius: "6px", fontSize: "10px", color: "#fafafa" }} itemStyle={{ padding: "2px 0" }} cursor={{ stroke: '#27272a', strokeWidth: 1 }} />
                        <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                        <Line yAxisId="right" type="monotone" dataKey="r2" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} activeDot={{ r: 4, strokeWidth: 0 }} />
                      </RechartsLineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Recent Trials Table */}
                  <div className="space-y-4">
                    <h3 className="text-[11px] uppercase font-bold tracking-widest text-[#52525b]">Optimization Results</h3>
                    <div className="border border-[hsl(var(--border))] rounded-lg overflow-hidden bg-[hsl(var(--panel))]/30">
                      <table className="w-full text-left text-[11px]">
                        <thead className="bg-[hsl(var(--panel))] text-[#52525b] font-bold">
                          <tr>
                            <th className="px-4 py-2 border-b border-[hsl(var(--border))]">TRIAL</th>
                            <th className="px-4 py-2 border-b border-[hsl(var(--border))]">R²</th>
                            <th className="px-4 py-2 border-b border-[hsl(var(--border))]">VAL_LOSS</th>
                            <th className="px-4 py-2 border-b border-[hsl(var(--border))]">MAE</th>
                          </tr>
                        </thead>
                        <tbody className="text-[hsl(var(--foreground))] font-mono">
                          {metricsHistory.slice(-10).reverse().map((m: any, i: number) => (
                            <tr key={i} className="border-b border-[hsl(var(--border))]/50 hover:bg-[hsl(var(--panel-lighter))]/30">
                              <td className="px-4 py-2 text-[#52525b]">#{m.trial}</td>
                              <td className="px-4 py-2 text-[hsl(var(--foreground-active))]">{m.r2?.toFixed(4) || "0.0000"}</td>
                              <td className="px-4 py-2">{m.val_loss?.toFixed(6) || "0.000000"}</td>
                              <td className="px-4 py-2">{m.mae?.toFixed(4) || "0.0000"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </Panel>

              {/* RESIZE HANDLE */}
              <PanelResizeHandle className="h-px bg-[hsl(var(--border))] hover:bg-[#3b82f6]/50 transition-colors cursor-row-resize" />

              {/* DOCKED BOTTOM LOGS (PROCESS STREAM) */}
              <Panel defaultSize={30} minSize={10}>
                <div className="h-full bg-[hsl(var(--panel))] flex flex-col overflow-hidden">
                  <div className="h-8 border-b border-[hsl(var(--border))] flex items-center justify-between px-4 bg-[hsl(var(--background))] shrink-0">
                    <div className="flex items-center gap-2">
                      <Terminal size={12} className="text-[#3b82f6]" />
                      <span className="text-[10px] font-bold uppercase tracking-wider text-[hsl(var(--foreground-active))]">Process Stream</span>
                    </div>
                    <button onClick={clearLogs} className="text-[10px] text-[#52525b] hover:text-[hsl(var(--foreground-active))] flex items-center gap-1.5"><Trash2 size={10} /> Clear</button>
                  </div>
                  <div className="flex-1 overflow-y-auto p-3 font-mono text-[11px] leading-snug custom-scrollbar bg-[hsl(var(--background))]">
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
              </Panel>
            </PanelGroup>
          </Tabs.Content>

          <Tabs.Content value="inference" className="h-full overflow-y-auto custom-scrollbar p-10 data-[state=inactive]:hidden">
            <div className="max-w-3xl mx-auto">
              <InferencePlayground loadedPath={loadedModelPath} />
            </div>
          </Tabs.Content>

          <Tabs.Content value="preview" className="h-full overflow-y-auto custom-scrollbar p-6 data-[state=inactive]:hidden">
            <DatasetPreview initialPath={selectedPredictor} />
          </Tabs.Content>
        </div>
      </Tabs.Root>
    </div>
  );
}
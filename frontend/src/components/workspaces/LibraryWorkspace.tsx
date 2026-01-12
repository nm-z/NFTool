import React, { useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { Search, ChevronRight, FileText, Activity, Cpu, Database } from "lucide-react";

import { TabTrigger } from "../common/UIComponents";
import { DatasetPreview } from "./tools/DatasetPreview";
import { SummaryCard, PlotCard } from "../common/Cards";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

export function LibraryWorkspace({ runs, onSelectRun, selectedRun, selectedPredictor }: any) {
  if (selectedRun) {
    return <RunDetailView run={selectedRun} onBack={() => onSelectRun(null)} />;
  }

  return (
    <div className="flex flex-col h-full bg-[hsl(var(--background))]">
      <Tabs.Root defaultValue="history" className="flex flex-col h-full">
        <div className="h-12 border-b border-[hsl(var(--border))] flex items-center px-6 bg-[hsl(var(--panel))]/50 shrink-0">
          <Tabs.List className="flex gap-8">
            <TabTrigger value="history" label="Run History" />
            <TabTrigger value="datasets" label="Dataset Assets" />
            <TabTrigger value="models" label="Model Checkpoints" />
          </Tabs.List>
        </div>
        
        <div className="flex-1 overflow-hidden">
          <Tabs.Content value="history" className="h-full flex flex-col data-[state=inactive]:hidden">
            <div className="h-12 border-b border-[hsl(var(--border))] flex items-center px-6 bg-[hsl(var(--panel))]/30 gap-4">
              <div className="flex items-center gap-2 flex-1 max-w-sm">
                <Search size={14} className="text-[#52525b]" />
                <input type="text" placeholder="Filter run history..." className="bg-transparent border-none outline-none text-[11px] w-full placeholder-[#3f3f46]" />
              </div>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
              <div className="border border-[hsl(var(--border))] rounded-lg overflow-hidden bg-[hsl(var(--panel))]/20">
                <table className="w-full text-left text-[12px]">
                  <thead className="bg-[hsl(var(--panel))] text-[#52525b] font-bold">
                    <tr>
                      <th className="px-6 py-3 border-b border-[hsl(var(--border))]">RUN_ID</th>
                      <th className="px-6 py-3 border-b border-[hsl(var(--border))]">MODEL</th>
                      <th className="px-6 py-3 border-b border-[hsl(var(--border))]">STATUS</th>
                      <th className="px-6 py-3 border-b border-[hsl(var(--border))]">R²</th>
                      <th className="px-6 py-3 border-b border-[hsl(var(--border))]">DATE</th>
                    </tr>
                  </thead>
                  <tbody className="text-[hsl(var(--foreground))] font-mono">
                    {runs.map((run: any) => (
                      <tr 
                        key={run.id} 
                        onClick={() => onSelectRun(run)}
                        className="border-b border-[hsl(var(--border))]/50 hover:bg-[hsl(var(--panel-lighter))]/30 cursor-pointer group"
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
                        <td className="px-6 py-4 text-[hsl(var(--foreground-active))]">{run.best_r2?.toFixed(4) || "—"}</td>
                        <td className="px-6 py-4 opacity-50">{new Date(run.timestamp).toLocaleDateString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </Tabs.Content>

          <Tabs.Content value="datasets" className="h-full overflow-y-auto custom-scrollbar p-6 data-[state=inactive]:hidden">
            <DatasetPreview initialPath={selectedPredictor} />
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
  return (
    <div className="flex flex-col h-full bg-[hsl(var(--background))]">
      <div className="h-12 border-b border-[hsl(var(--border))] flex items-center justify-between px-6 bg-[hsl(var(--panel))]/50 shrink-0">
        <div className="flex items-center gap-4">
          <button onClick={onBack} className="p-1.5 hover:bg-[hsl(var(--panel-lighter))] rounded text-[#52525b] hover:text-[hsl(var(--foreground-active))] transition-all">
            <ChevronRight size={16} className="rotate-180" />
          </button>
          <h2 className="text-[12px] font-bold text-[hsl(var(--foreground-active))] uppercase tracking-widest">Run Details: #{run.run_id}</h2>
        </div>
        <div className="flex items-center gap-2">
          {run.report_path && (
            <a 
              href={`${API_URL}/reports/${run.report_path}`} 
              target="_blank" 
              className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-[hsl(var(--foreground-active))] bg-[#3b82f6] hover:bg-[#2563eb] transition-all"
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
            <PlotCard title="Optimization History" src={`${API_URL}/results/optuna_optimization_history.png`} />
            <PlotCard title="Parameter Importances" src={`${API_URL}/results/optuna_param_importances.png`} />
            <PlotCard title="Pred vs Actual" src={`${API_URL}/results/r2_pred_vs_actual.png`} />
            <PlotCard title="Error Histogram" src={`${API_URL}/results/r2_error_histogram.png`} />
          </div>
        </div>
      </div>
    </div>
  );
}

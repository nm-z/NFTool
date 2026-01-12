import React, { useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { Search, ChevronRight, FileText, Activity, Cpu, Database, Download, Rocket } from "lucide-react";

import { TabTrigger } from "../common/UIComponents";
import { DatasetPreview } from "./tools/DatasetPreview";
import { SummaryCard, PlotCard } from "../common/Cards";
import { useTrainingStore } from "@/store/useTrainingStore";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";

export function LibraryWorkspace() {
  const { runs, selectedPredictor } = useTrainingStore();
  const [selectedRun, setSelectedRun] = useState<any>(null);

  if (selectedRun) {
    return <RunDetailView run={selectedRun} onBack={() => setSelectedRun(null)} />;
  }

  return (
    <div className="flex flex-col h-full bg-zinc-950">
      <Tabs.Root defaultValue="history" className="flex flex-col h-full">
        <div className="h-12 border-b border-zinc-800 flex items-center px-6 bg-zinc-900/50 shrink-0">
          <Tabs.List className="flex gap-8">
            <TabTrigger value="history" label="Run History" />
            <TabTrigger value="datasets" label="Dataset Assets" />
            <TabTrigger value="models" label="Model Checkpoints" />
          </Tabs.List>
        </div>
        
        <div className="flex-1 overflow-hidden">
          <Tabs.Content value="history" className="h-full flex flex-col data-[state=inactive]:hidden">
            <div className="h-12 border-b border-zinc-800 flex items-center px-6 bg-zinc-900/30 gap-4">
              <div className="flex items-center gap-2 flex-1 max-w-sm">
                <Search size={14} className="text-[#52525b]" />
                <input 
                  type="text" 
                  placeholder="Filter run history..." 
                  className="bg-transparent border-none outline-none text-[11px] w-full placeholder-[#3f3f46]" 
                  data-testid="input-filter-history"
                />
              </div>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
              <div className="border border-zinc-800 rounded-lg overflow-hidden bg-zinc-900/20">
                <table className="w-full text-left text-[12px]" suppressHydrationWarning>
                  <thead className="bg-zinc-900 text-[#52525b] font-bold">
                    <tr>
                      <th className="px-6 py-3 border-b border-zinc-800">RUN_ID</th>
                      <th className="px-6 py-3 border-b border-zinc-800">MODEL</th>
                      <th className="px-6 py-3 border-b border-zinc-800">STATUS</th>
                      <th className="px-6 py-3 border-b border-zinc-800">R²</th>
                      <th className="px-6 py-3 border-b border-zinc-800">DATE</th>
                    </tr>
                  </thead>
                  <tbody className="text-white font-mono">
                    {runs?.map((run: any) => (
                      <tr 
                        key={run.id} 
                        onClick={() => setSelectedRun(run)}
                        data-testid={`run-row-${run.run_id}`}
                        className="border-b border-zinc-800/50 hover:bg-zinc-800/30 cursor-pointer group"
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
                        <td className="px-6 py-4 text-white">{run.best_r2?.toFixed(4) || "—"}</td>
                        <td className="px-6 py-4 opacity-50">
                          {run.timestamp ? new Date(run.timestamp).toLocaleDateString() : "—"}
                        </td>
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
             <div className="text-[11px] text-[#52525b] italic">No local checkpoints found in workspace/runs/reports</div>
          </Tabs.Content>
        </div>
      </Tabs.Root>
    </div>
  );
}

function RunDetailView({ run, onBack }: { run: any, onBack: () => void }) {
  const { setLoadedModelPath, setActiveWorkspace, addLog } = useTrainingStore();

  const handleExport = async () => {
    try {
      const res = await fetch(`${API_URL}/download-weights/${run.run_id}`, {
        headers: { "X-API-Key": API_KEY }
      });
      if (!res.ok) throw new Error("Download failed");
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${run.run_id}_weights.pt`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      addLog({ time: new Date().toLocaleTimeString(), msg: `Exported weights for ${run.run_id}.`, type: "success" });
    } catch (err: any) {
      addLog({ time: new Date().toLocaleTimeString(), msg: `Export failed: ${err.message}`, type: "warn" });
    }
  };

  const handleDeploy = () => {
    if (run.status !== 'completed') {
      addLog({ time: new Date().toLocaleTimeString(), msg: "Cannot deploy uncompleted run.", type: "warn" });
      return;
    }
    setLoadedModelPath(run.run_id); 
    addLog({ time: new Date().toLocaleTimeString(), msg: `Model ${run.run_id} deployed to Playground.`, type: "success" });
    setActiveWorkspace("Train"); 
  };

  const formattedDate = run.timestamp ? new Date(run.timestamp).toLocaleDateString() : "—";
  const formattedTime = run.timestamp ? new Date(run.timestamp).toLocaleTimeString() : "—";

  return (
    <div className="flex flex-col h-full bg-zinc-950">
      <div className="h-12 border-b border-zinc-800 flex items-center justify-between px-6 bg-zinc-900/50 shrink-0">
        <div className="flex items-center gap-4">
          <button 
            onClick={onBack} 
            data-testid="btn-back-to-history"
            className="p-1.5 hover:bg-zinc-800 rounded text-[#52525b] hover:text-white transition-all"
          >
            <ChevronRight size={16} className="rotate-180" />
          </button>
          <h2 className="text-[12px] font-bold text-white uppercase tracking-widest">Run Details: #{run.run_id}</h2>
        </div>
        <div className="flex items-center gap-3">
          <button 
            onClick={handleExport}
            data-testid="btn-export-weights"
            className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-[#52525b] hover:text-white bg-zinc-900 border border-zinc-800 transition-all"
          >
            <Download size={12} />
            Export Weights
          </button>
          <button 
            onClick={handleDeploy}
            data-testid="btn-deploy-playground"
            className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-white bg-[#22c55e] hover:bg-[#16a34a] transition-all"
          >
            <Rocket size={12} />
            Deploy to Playground
          </button>
          {run.report_path && (
            <a 
              href={`${API_URL}/reports/${run.report_path}`} 
              target="_blank" 
              data-testid="link-view-report"
              className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-white bg-[#3b82f6] hover:bg-[#2563eb] transition-all"
            >
              <FileText size={12} />
              View Report
            </a>
          )}
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto custom-scrollbar p-8 space-y-8">
        <div className="grid grid-cols-4 gap-6">
          <SummaryCard icon={Activity} label="Status" value={run.status.toUpperCase()} subValue={run.model_choice} />
          <SummaryCard icon={Activity} label="Best R²" value={run.best_r2?.toFixed(4) || "0.0000"} subValue="Optimization Peak" />
          <SummaryCard icon={Cpu} label="Trials" value={run.optuna_trials} subValue="Configured Budget" />
          <SummaryCard icon={Database} label="Date" value={formattedDate} subValue={formattedTime} />
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

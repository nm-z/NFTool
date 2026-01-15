import React, { useState } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import {
  Search,
  ChevronRight,
  FileText,
  Activity,
  Cpu,
  Database,
  Download,
  Rocket,
  Folder,
  FolderOpen,
  File,
  UploadCloud,
} from "lucide-react";

import { TabTrigger } from "../common/UIComponents";
import { SummaryCard, PlotCard } from "../common/Cards";
import { useTrainingStore } from "@/store/useTrainingStore";

const API_ROOT = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
const API_BASE = API_ROOT.replace(/\/+$/, "").endsWith("/api/v1")
  ? API_ROOT.replace(/\/+$/, "")
  : `${API_ROOT.replace(/\/+$/, "")}/api/v1`;
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";

type RunRecord = {
  run_id?: string | number;
  id?: string | number;
  model_choice?: string;
  status?: string;
  best_r2?: number;
  timestamp?: string;
  report_path?: string;
  optuna_trials?: number;
  [k: string]: unknown;
};

type AssetNode = {
  name: string;
  type: "folder" | "file";
  path: string;
  size?: number;
  children?: AssetNode[];
};

export function LibraryWorkspace() {
  const { runs } = useTrainingStore();
  const [selectedRun, setSelectedRun] = useState<RunRecord | null>(null);
  const [assetRoots, setAssetRoots] = useState<AssetNode[]>([]);
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({});
  const [datasetFolderName, setDatasetFolderName] = useState("");
  const [predictorFiles, setPredictorFiles] = useState<FileList | null>(null);
  const [targetFiles, setTargetFiles] = useState<FileList | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const fetchAssetTree = async () => {
    const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
    const res = await fetch(`${API_BASE}/data/assets/tree`, {
      headers: { "X-API-Key": API_KEY },
    });
    if (!res.ok) {
      throw new Error("Failed to load data tree");
    }
    const data = await res.json();
    const roots = Array.isArray(data?.roots) ? data.roots : [];
    setAssetRoots(roots);
    setExpandedNodes((prev) => {
      const next = { ...prev };
      for (const node of roots) {
        if (node?.path) next[node.path] = true;
      }
      return next;
    });
  };

  React.useEffect(() => {
    fetchAssetTree().catch((err) => {
      console.warn("Failed to fetch data tree:", err);
    });
  }, []);

  const toggleNode = (path: string) => {
    setExpandedNodes((prev) => ({ ...prev, [path]: !prev[path] }));
  };

  const handleDownload = async (node: AssetNode) => {
    const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
    const res = await fetch(
      `${API_BASE}/data/assets/download?path=${encodeURIComponent(node.path)}`,
      { headers: { "X-API-Key": API_KEY } },
    );
    if (!res.ok) {
      setUploadStatus("Download failed.");
      return;
    }
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download =
      node.type === "folder" ? `${node.name || "download"}.zip` : node.name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  };

  const handleUpload = async () => {
    if (!datasetFolderName.trim()) {
      setUploadStatus("Folder name is required.");
      return;
    }
    if (!predictorFiles?.length && !targetFiles?.length) {
      setUploadStatus("Select predictor and/or target files to upload.");
      return;
    }
    setIsUploading(true);
    setUploadStatus(null);
    try {
      const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
      const form = new FormData();
      form.append("folder_name", datasetFolderName.trim());
      Array.from(predictorFiles ?? []).forEach((file) =>
        form.append("predictor_files", file),
      );
      Array.from(targetFiles ?? []).forEach((file) =>
        form.append("target_files", file),
      );

      const res = await fetch(`${API_BASE}/data/datasets/upload`, {
        method: "POST",
        headers: { "X-API-Key": API_KEY },
        body: form,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err?.detail || "Upload failed");
      }
      setUploadStatus("Upload complete.");
      setDatasetFolderName("");
      setPredictorFiles(null);
      setTargetFiles(null);
      await fetchAssetTree();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setUploadStatus(msg);
    } finally {
      setIsUploading(false);
    }
  };

  const renderNode = (node: AssetNode, depth = 0) => {
    const isFolder = node.type === "folder";
    const isExpanded = !!expandedNodes[node.path];
    const hasChildren = (node.children?.length ?? 0) > 0;
    return (
      <div key={node.path}>
        <div
          className="flex items-center justify-between gap-3 py-1.5 px-2 rounded hover:bg-zinc-900/60"
          style={{ paddingLeft: 12 + depth * 12 }}
        >
          <div className="flex items-center gap-2">
            {isFolder ? (
              <button
                onClick={() => toggleNode(node.path)}
                className="text-zinc-500 hover:text-white"
              >
                {isExpanded ? <FolderOpen size={14} /> : <Folder size={14} />}
              </button>
            ) : (
              <File size={14} className="text-zinc-500" />
            )}
            <span className="text-[11px] text-white font-mono">
              {node.name}
            </span>
            {!isFolder && typeof node.size === "number" ? (
              <span className="text-[9px] text-zinc-600">
                {Math.max(1, Math.round(node.size / 1024))} KB
              </span>
            ) : null}
          </div>
          <button
            onClick={() => handleDownload(node)}
            className="text-[9px] uppercase font-bold text-blue-400 hover:text-blue-300"
          >
            Download
          </button>
        </div>
        {isFolder && isExpanded && hasChildren
          ? node.children?.map((child) => renderNode(child, depth + 1))
          : null}
        {isFolder && isExpanded && !hasChildren ? (
          <div
            className="text-[10px] text-zinc-600 px-2 py-1"
            style={{ paddingLeft: 28 + depth * 12 }}
          >
            Empty folder
          </div>
        ) : null}
      </div>
    );
  };

  if (selectedRun) {
    return (
      <RunDetailView run={selectedRun} onBack={() => setSelectedRun(null)} />
    );
  }

  return (
    <div className="flex flex-col h-full bg-zinc-950">
      <Tabs.Root defaultValue="history" className="flex flex-col h-full">
        <div className="h-12 border-b border-zinc-800 flex items-center px-6 bg-zinc-900/50 shrink-0">
          <Tabs.List className="flex gap-8">
            <TabTrigger value="history" label="Run History" />
            <TabTrigger value="data" label="Data" />
          </Tabs.List>
        </div>

        <div className="flex-1 overflow-hidden">
          <Tabs.Content
            value="history"
            className="h-full flex flex-col data-[state=inactive]:hidden"
          >
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
                <table
                  className="w-full text-left text-[12px]"
                  suppressHydrationWarning
                >
                  <thead className="bg-zinc-900 text-[#52525b] font-bold">
                    <tr>
                      <th className="px-6 py-3 border-b border-zinc-800">
                        RUN_ID
                      </th>
                      <th className="px-6 py-3 border-b border-zinc-800">
                        MODEL
                      </th>
                      <th className="px-6 py-3 border-b border-zinc-800">
                        STATUS
                      </th>
                      <th className="px-6 py-3 border-b border-zinc-800">R²</th>
                      <th className="px-6 py-3 border-b border-zinc-800">
                        DATE
                      </th>
                    </tr>
                  </thead>
                  <tbody className="text-white font-mono">
                    {runs?.map((run: RunRecord) => (
                      <tr
                        key={run.id}
                        onClick={() => setSelectedRun(run)}
                        data-testid={`run-row-${run.run_id}`}
                        className="border-b border-zinc-800/50 hover:bg-zinc-800/30 cursor-pointer group"
                      >
                        <td className="px-6 py-4 text-[#3b82f6] group-hover:text-[#60a5fa] font-bold">
                          #{run.run_id}
                        </td>
                        <td className="px-6 py-4">{run.model_choice}</td>
                        <td className="px-6 py-4">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                              run.status === "completed"
                                ? "bg-green-500/10 text-green-400"
                                : run.status === "failed"
                                  ? "bg-red-500/10 text-red-400"
                                  : run.status === "running"
                                    ? "bg-blue-500/10 text-blue-400 animate-pulse"
                                    : "bg-yellow-500/10 text-yellow-400"
                            }`}
                          >
                            {run.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-white">
                          {run.best_r2?.toFixed(4) || "—"}
                        </td>
                        <td className="px-6 py-4 opacity-50">
                          {run.timestamp
                            ? new Date(run.timestamp).toLocaleDateString()
                            : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </Tabs.Content>

          <Tabs.Content
            value="data"
            className="h-full overflow-y-auto custom-scrollbar p-6 data-[state=inactive]:hidden"
          >
            <div className="space-y-6">
              <div className="bg-zinc-900/40 border border-zinc-800 rounded-lg p-5 space-y-4">
                <div className="flex items-center gap-2 text-[11px] uppercase font-bold text-zinc-400 tracking-widest">
                  <UploadCloud size={14} />
                  Upload Datasets
                </div>
                <div className="text-[10px] text-zinc-500">
                  Name the dataset folder and separate predictor/target files. If
                  there is exactly one of each, they stay in the folder root.
                  If there are more than two predictors or targets, they are
                  placed under Predictors/Targets folders respectively.
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="col-span-1">
                    <label className="text-[9px] font-bold text-zinc-500 uppercase">
                      Folder Name
                    </label>
                    <input
                      type="text"
                      value={datasetFolderName}
                      onChange={(e) => setDatasetFolderName(e.target.value)}
                      placeholder="e.g. Hold-2"
                      className="mt-1 w-full bg-black border border-zinc-800 rounded px-3 py-2 text-[11px] text-white focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="text-[9px] font-bold text-zinc-500 uppercase">
                      Predictors
                    </label>
                    <input
                      type="file"
                      multiple
                      onChange={(e) => setPredictorFiles(e.target.files)}
                      className="mt-1 w-full text-[11px] text-zinc-400"
                    />
                  </div>
                  <div>
                    <label className="text-[9px] font-bold text-zinc-500 uppercase">
                      Targets
                    </label>
                    <input
                      type="file"
                      multiple
                      onChange={(e) => setTargetFiles(e.target.files)}
                      className="mt-1 w-full text-[11px] text-zinc-400"
                    />
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <button
                    onClick={handleUpload}
                    disabled={isUploading}
                    className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white text-[10px] font-bold uppercase rounded disabled:opacity-60"
                  >
                    {isUploading ? "Uploading..." : "Upload"}
                  </button>
                  {uploadStatus && (
                    <span className="text-[10px] text-zinc-500">
                      {uploadStatus}
                    </span>
                  )}
                </div>
              </div>

              <div className="bg-zinc-900/40 border border-zinc-800 rounded-lg p-4">
                <div className="text-[11px] uppercase font-bold tracking-widest text-zinc-500 mb-3">
                  Data Explorer
                </div>
                {assetRoots.length === 0 ? (
                  <div className="text-[11px] text-zinc-500 italic">
                    No datasets or models found.
                  </div>
                ) : (
                  <div className="space-y-1">
                    {assetRoots.map((node) => renderNode(node))}
                  </div>
                )}
              </div>
            </div>
          </Tabs.Content>
        </div>
      </Tabs.Root>
    </div>
  );
}

function RunDetailView({
  run,
  onBack,
}: {
  run: RunRecord;
  onBack: () => void;
}) {
  const { setLoadedModelPath, setActiveWorkspace, addLog } = useTrainingStore();

  const handleExport = async () => {
    try {
      const res = await fetch(`${API_BASE}/training/download-weights/${run.run_id}`, {
        headers: { "X-API-Key": API_KEY },
      });
      if (!res.ok) throw new Error("Download failed");
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${run.run_id}_weights.pt`;
      document.body.appendChild(a);
      // Attempt a direct click; if the environment blocks programmatic clicks,
      // dispatch a synthetic MouseEvent as a fallback. The outer try/catch will
      // capture any errors and log them.
      try {
        a.click();
      } catch {
        a.dispatchEvent(
          new MouseEvent("click", { bubbles: true, cancelable: true, view: window }),
        );
      }
      a.remove();
      addLog({
        time: new Date().toLocaleTimeString(),
        msg: `Exported weights for ${run.run_id}.`,
        type: "success",
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addLog({
        time: new Date().toLocaleTimeString(),
        msg: `Export failed: ${msg}`,
        type: "warn",
      });
    }
  };

  const handleDeploy = () => {
    if (run.status !== "completed") {
      addLog({
        time: new Date().toLocaleTimeString(),
        msg: "Cannot deploy uncompleted run.",
        type: "warn",
      });
      return;
    }
    setLoadedModelPath(run.run_id != null ? String(run.run_id) : null);
    addLog({
      time: new Date().toLocaleTimeString(),
      msg: `Model ${run.run_id} deployed to Playground.`,
      type: "success",
    });
    setActiveWorkspace("Train");
  };

  const handleViewReport = async () => {
    if (!run.run_id) return;
    try {
      const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
      const res = await fetch(`${API_BASE}/training/runs/${run.run_id}/log`, {
        headers: { "X-API-Key": API_KEY },
      });
      if (!res.ok) throw new Error("Failed to fetch run log");
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      window.open(url, "_blank", "noopener,noreferrer");
      setTimeout(() => window.URL.revokeObjectURL(url), 5000);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addLog({
        time: new Date().toLocaleTimeString(),
        msg: `View report failed: ${msg}`,
        type: "warn",
      });
    }
  };

  const formattedDate = run.timestamp
    ? new Date(run.timestamp).toLocaleDateString()
    : "—";
  const formattedTime = run.timestamp
    ? new Date(run.timestamp).toLocaleTimeString()
    : "—";

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
          <h2 className="text-[12px] font-bold text-white uppercase tracking-widest">
            Run Details: #{run.run_id}
          </h2>
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
          <button
            onClick={handleViewReport}
            data-testid="link-view-report"
            className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-white bg-[#3b82f6] hover:bg-[#2563eb] transition-all"
          >
            <FileText size={12} />
            View Report
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar p-8 space-y-8">
        <div className="grid grid-cols-4 gap-6">
          <SummaryCard
            icon={Activity}
            label="Status"
            value={run.status ? run.status.toUpperCase() : "UNKNOWN"}
            subValue={run.model_choice ?? ""}
          />
          <SummaryCard
            icon={Activity}
            label="Best R²"
            value={run.best_r2?.toFixed(4) || "0.0000"}
            subValue="Optimization Peak"
          />
          <SummaryCard
            icon={Cpu}
            label="Trials"
            value={run.optuna_trials != null ? String(run.optuna_trials) : "0"}
            subValue="Configured Budget"
          />
          <SummaryCard
            icon={Database}
            label="Date"
            value={formattedDate}
            subValue={formattedTime}
          />
        </div>

        <div className="space-y-4">
          <h3 className="text-[11px] uppercase font-bold tracking-widest text-[#52525b]">
            Generated Analytics
          </h3>
          {run.status !== "completed" ? (
            <div className="text-[10px] text-zinc-500">
              Plots are generated only after a completed run.
            </div>
          ) : (
            <>
              <div className="text-[10px] text-zinc-500">
                Run assets are served from the report directory for this run.
              </div>
              {run.run_id ? null : (
                <div className="text-[10px] text-yellow-500">
                  Missing run id; plot links may be unavailable.
                </div>
              )}
              <div className="grid grid-cols-2 gap-6">
                {(() => {
                  const runId = run.run_id ? String(run.run_id) : "";
                  const base = runId
                    ? `${API_ROOT}/reports/${runId}`
                    : `${API_ROOT}/results`;
                  return (
                    <>
                      <PlotCard
                        title="Optimization History"
                        src={`${base}/optuna_optimization_history.png`}
                      />
                      <PlotCard
                        title="Parameter Importances"
                        src={`${base}/optuna_param_importances.png`}
                      />
                      <PlotCard
                        title="Pred vs Actual"
                        src={`${base}/predicted_vs_actual.png`}
                      />
                      <PlotCard
                        title="Residual Distribution"
                        src={`${base}/residual_distribution.png`}
                      />
                    </>
                  );
                })()}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

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
import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  ReferenceLine,
} from "recharts";

import { TabTrigger } from "../common/UIComponents";
import { SummaryCard } from "../common/Cards";
import { useTrainingStore, type MetricPoint } from "@/store/useTrainingStore";
import { DatasetPreview } from "./tools/DatasetPreview";
import { useApi } from "../ApiProvider";

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
  metrics_history?: MetricPoint[];
  [k: string]: unknown;
};

type AssetNode = {
  name: string;
  type: "folder" | "file";
  path: string;
  size?: number;
  children?: AssetNode[];
};

type EvalComparison = {
  index: number;
  actual: number;
  predicted: number;
  abs_error: number;
  percent_error: number;
};

export type EvalResult = {
  run_id: string;
  r2: number;
  mae: number;
  rmse: number;
  mse: number;
  accuracy_percent: number;
  mape_percent: number;
  r2_score: number;
  count: number;
  predictions?: number[];
  actuals?: number[];
  comparisons: EvalComparison[];
};

export function LibraryWorkspace() {
  const { apiUrl: API_BASE } = useApi();
  const { runs, datasets } = useTrainingStore();
  const [selectedRun, setSelectedRun] = useState<RunRecord | null>(null);
  const [assetRoots, setAssetRoots] = useState<AssetNode[]>([]);
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({});
  const [datasetFolderName, setDatasetFolderName] = useState("");
  const [predictorFiles, setPredictorFiles] = useState<FileList | null>(null);
  const [targetFiles, setTargetFiles] = useState<FileList | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const initialPreviewPath = datasets?.[0]?.path;

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
            className="text-[9px] uppercase font-bold text-[hsl(var(--primary-soft))] hover:text-[hsl(var(--primary))]"
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
                <Search size={14} className="text-[hsl(var(--foreground-dim))]" />
                <input
                  type="text"
                  placeholder="Filter run history..."
                  className="bg-transparent border-none outline-none text-[11px] w-full placeholder-[hsl(var(--foreground-subtle))]"
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
                  <thead className="bg-zinc-900 text-[hsl(var(--foreground-dim))] font-bold">
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
                        <td className="px-6 py-4 text-[hsl(var(--primary))] group-hover:text-[hsl(var(--primary-soft))] font-bold">
                          #{run.run_id}
                        </td>
                        <td className="px-6 py-4">{run.model_choice}</td>
                        <td className="px-6 py-4">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                              run.status === "completed"
                                ? "bg-[hsl(var(--success)/0.1)] text-[hsl(var(--success))]"
                                : run.status === "failed"
                                  ? "bg-[hsl(var(--danger)/0.1)] text-[hsl(var(--danger))]"
                                  : run.status === "running"
                                    ? "bg-[hsl(var(--primary)/0.1)] text-[hsl(var(--primary-soft))] animate-pulse"
                                    : "bg-[hsl(var(--warning)/0.1)] text-[hsl(var(--warning))]"
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
                      className="mt-1 w-full bg-black border border-zinc-800 rounded px-3 py-2 text-[11px] text-white focus:outline-none focus:border-[hsl(var(--primary))]"
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
                    className="px-4 py-2 bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-strong))] text-[hsl(var(--foreground-active))] text-[10px] font-bold uppercase rounded disabled:opacity-60"
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

              <div className="bg-zinc-900/40 border border-zinc-800 rounded-lg p-4">
                <div className="text-[11px] uppercase font-bold tracking-widest text-zinc-500 mb-3">
                  Dataset Preview
                </div>
                <DatasetPreview initialPath={initialPreviewPath} />
              </div>
            </div>
          </Tabs.Content>
        </div>
      </Tabs.Root>
    </div>
  );
}

function ChartCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="border border-[hsl(var(--border))] rounded-xl overflow-hidden bg-[hsl(var(--panel))]/50 flex flex-col">
      <div className="px-4 py-2 border-b border-[hsl(var(--border))] bg-[hsl(var(--panel))]">
        <span className="text-[10px] font-bold uppercase text-[hsl(var(--foreground-dim))]">
          {title}
        </span>
      </div>
      <div className="flex-1 p-3 min-h-[300px] flex items-center justify-center">
        {children}
      </div>
    </div>
  );
}

function EmptyChart({ message }: { message: string }) {
  return (
    <div className="text-[10px] text-zinc-600 text-center">{message}</div>
  );
}

function buildResidualBins(comparisons: EvalComparison[], binCount = 24) {
  if (!comparisons.length) return [];
  const residuals = comparisons.map((c) => c.predicted - c.actual);
  const min = Math.min(...residuals);
  const max = Math.max(...residuals);
  const span = max - min;
  const safeSpan = span === 0 ? 1 : span;
  const step = safeSpan / binCount;
  const bins: Array<{ bin: number; label: string; count: number }> = [];
  for (let i = 0; i < binCount; i += 1) {
    const start = min + i * step;
    const end = start + step;
    bins.push({
      bin: start + step / 2,
      label: `${start.toFixed(3)} – ${end.toFixed(3)}`,
      count: 0,
    });
  }
  for (const res of residuals) {
    const idx = Math.max(
      0,
      Math.min(binCount - 1, Math.floor((res - min) / step)),
    );
    bins[idx].count += 1;
  }
  return bins;
}

function RunDetailView({
  run,
  onBack,
}: {
  run: RunRecord;
  onBack: () => void;
}) {
  const { apiUrl: API_BASE } = useApi();
  const { setLoadedModelPath, setActiveWorkspace, addLog } = useTrainingStore();
  const [evaluation, setEvaluation] = useState<EvalResult | null>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);

  const metricsHistory = React.useMemo(
    () =>
      Array.isArray(run.metrics_history)
        ? (run.metrics_history as MetricPoint[])
        : [],
    [run.metrics_history],
  );

  const dedupedMetrics = React.useMemo(() => {
    const seen = new Map<string, MetricPoint>();
    metricsHistory.forEach((point, index) => {
      const trialVal = typeof point.trial === "number" ? point.trial : 0;
      const epochVal = typeof point.epoch === "number" ? point.epoch : index;
      seen.set(`${trialVal}-${epochVal}`, point);
    });
    return Array.from(seen.values());
  }, [metricsHistory]);

  const chartData = React.useMemo(
    () =>
      dedupedMetrics.map((point, index) => ({
        ...point,
        step: index,
      })),
    [dedupedMetrics],
  );

  const trialData = React.useMemo(() => {
    const trialMap = new Map<number, { trial: number; bestR2: number; minVal: number }>();
    for (const point of metricsHistory) {
      const trialVal = typeof point.trial === "number" ? point.trial : 0;
      const r2Val = typeof point.r2 === "number" ? point.r2 : Number.NEGATIVE_INFINITY;
      const valLoss = typeof point.val_loss === "number" ? point.val_loss : Number.POSITIVE_INFINITY;
      const current = trialMap.get(trialVal) || {
        trial: trialVal,
        bestR2: Number.NEGATIVE_INFINITY,
        minVal: Number.POSITIVE_INFINITY,
      };
      current.bestR2 = Math.max(current.bestR2, r2Val);
      current.minVal = Math.min(current.minVal, valLoss);
      trialMap.set(trialVal, current);
    }
    return Array.from(trialMap.values())
      .sort((a, b) => a.trial - b.trial)
      .map((item) => ({
        trial: item.trial + 1,
        best_r2: Number.isFinite(item.bestR2) ? item.bestR2 : 0,
        val_loss: Number.isFinite(item.minVal) ? item.minVal : 0,
      }));
  }, [metricsHistory]);

  const evalComparisons = React.useMemo(
    () => evaluation?.comparisons ?? [],
    [evaluation?.comparisons],
  );
  const residualBins = React.useMemo(
    () => buildResidualBins(evalComparisons),
    [evalComparisons],
  );

  React.useEffect(() => {
    if (!run.run_id || run.status !== "completed") return;
    let cancelled = false;
    const fetchEvaluation = async () => {
      setEvaluationLoading(true);
      setEvaluationError(null);
      try {
        const res = await fetch(`${API_BASE}/training/inference/evaluate`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY,
          },
          body: JSON.stringify({
            run_id: String(run.run_id),
            max_points: 1000,
          }),
        });
        if (!res.ok) {
          let errText = "Evaluation failed";
          try {
            const errJson = await res.json();
            errText = errJson?.detail || errText;
          } catch {
            errText = await res.text();
          }
          throw new Error(errText || "Evaluation failed");
        }
        const data = (await res.json()) as EvalResult;
        if (!cancelled) {
          setEvaluation(data);
        }
      } catch (err: unknown) {
        if (!cancelled) {
          const msg = err instanceof Error ? err.message : String(err);
          setEvaluationError(msg || "Evaluation error");
        }
      } finally {
        if (!cancelled) setEvaluationLoading(false);
      }
    };
    fetchEvaluation();
    return () => {
      cancelled = true;
    };
  }, [run.run_id, run.status]);

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
            className="p-1.5 hover:bg-zinc-800 rounded text-[hsl(var(--foreground-dim))] hover:text-white transition-all"
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
            className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-[hsl(var(--foreground-dim))] hover:text-white bg-zinc-900 border border-zinc-800 transition-all"
          >
            <Download size={12} />
            Export Weights
          </button>
          <button
            onClick={handleDeploy}
            data-testid="btn-deploy-playground"
            className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-[hsl(var(--foreground-active))] bg-[hsl(var(--success))] hover:bg-[hsl(var(--success-strong))] transition-all"
          >
            <Rocket size={12} />
            Deploy to Playground
          </button>
          <button
            onClick={handleViewReport}
            data-testid="link-view-report"
            className="flex items-center gap-2 px-3 py-1 rounded text-[11px] font-bold text-[hsl(var(--foreground-active))] bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-strong))] transition-all"
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
          <h3 className="text-[11px] uppercase font-bold tracking-widest text-[hsl(var(--foreground-dim))]">
            Generated Analytics
          </h3>
          {run.status !== "completed" ? (
            <div className="text-[10px] text-zinc-500">
              Plots are generated only after a completed run.
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-6">
                <ChartCard title="Optimization History">
                  {chartData.length === 0 ? (
                    <EmptyChart message="No metrics history recorded for this run." />
                  ) : (
                    <ResponsiveContainer width="100%" height={280}>
                      <RechartsLineChart
                        data={chartData}
                        margin={{ top: 5, right: 5, left: -20, bottom: 5 }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="hsl(var(--border-strong))"
                          vertical={false}
                          horizontal={true}
                        />
                        <XAxis dataKey="step" hide />
                        <YAxis
                          yAxisId="left"
                          stroke="hsl(var(--foreground-subtle))"
                          tick={{
                            fill: "hsl(var(--foreground-dim))",
                            fontSize: 9,
                            fontWeight: 600,
                          }}
                          tickLine={false}
                          axisLine={false}
                        />
                        <YAxis
                          yAxisId="right"
                          orientation="right"
                          stroke="hsl(var(--foreground-subtle))"
                          tick={{
                            fill: "hsl(var(--foreground-dim))",
                            fontSize: 9,
                            fontWeight: 600,
                          }}
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--background))",
                            border: "1px solid hsl(var(--border-strong))",
                            borderRadius: "6px",
                            fontSize: "10px",
                            color: "hsl(var(--foreground-active))",
                          }}
                          itemStyle={{ padding: "2px 0" }}
                          cursor={{
                            stroke: "hsl(var(--border-strong))",
                            strokeWidth: 1,
                          }}
                          labelFormatter={(_label, payload) => {
                            const entry = Array.isArray(payload)
                              ? payload[0]?.payload
                              : undefined;
                            if (!entry) return "Step";
                            const trialLabel =
                              entry.trial != null ? `Trial ${entry.trial + 1}` : "Trial ?";
                            const epochLabel =
                              entry.epoch != null ? `Epoch ${entry.epoch}` : "Epoch ?";
                            return `${trialLabel} • ${epochLabel}`;
                          }}
                          formatter={(value: number | undefined, name: string | undefined) => {
                            const precision = name === "val_loss" ? 6 : 4;
                            const label = name === "val_loss" ? "Val Loss" : "R²";
                            return [value?.toFixed(precision) ?? "—", label];
                          }}
                        />
                        <Line
                          yAxisId="left"
                          type="monotone"
                          dataKey="val_loss"
                          stroke="hsl(var(--danger))"
                          strokeWidth={2}
                          dot={false}
                          isAnimationActive={false}
                        />
                        <Line
                          yAxisId="right"
                          type="monotone"
                          dataKey="r2"
                          stroke="hsl(var(--primary))"
                          strokeWidth={2}
                          dot={false}
                          isAnimationActive={false}
                        />
                      </RechartsLineChart>
                    </ResponsiveContainer>
                  )}
                </ChartCard>

                <ChartCard title="Trial Performance (Best R²)">
                  {trialData.length === 0 ? (
                    <EmptyChart message="No trial summaries available." />
                  ) : (
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={trialData} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border-strong))" vertical={false} />
                        <XAxis dataKey="trial" hide />
                        <YAxis
                          stroke="hsl(var(--foreground-subtle))"
                          tick={{ fill: "hsl(var(--foreground-dim))", fontSize: 9, fontWeight: 600 }}
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--background))",
                            border: "1px solid hsl(var(--border-strong))",
                            borderRadius: "6px",
                            fontSize: "10px",
                            color: "hsl(var(--foreground-active))",
                          }}
                          formatter={(value: number | undefined) => [value?.toFixed(4) ?? "—", "Best R²"]}
                        />
                        <Bar dataKey="best_r2" fill="hsl(var(--success))" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </ChartCard>

                <ChartCard title="Pred vs Actual">
                  {evaluationLoading ? (
                    <EmptyChart message="Loading evaluation data..." />
                  ) : evaluationError ? (
                    <EmptyChart message={evaluationError} />
                  ) : evalComparisons.length === 0 ? (
                    <EmptyChart message="No evaluation comparisons available." />
                  ) : (
                    <ResponsiveContainer width="100%" height={280}>
                      <ScatterChart margin={{ top: 10, right: 10, left: -10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border-strong))" />
                        <XAxis
                          dataKey="actual"
                          tick={{ fill: "hsl(var(--foreground-dim))", fontSize: 9, fontWeight: 600 }}
                          tickLine={false}
                          axisLine={false}
                          type="number"
                        />
                        <YAxis
                          dataKey="predicted"
                          tick={{ fill: "hsl(var(--foreground-dim))", fontSize: 9, fontWeight: 600 }}
                          tickLine={false}
                          axisLine={false}
                          type="number"
                        />
                        <Tooltip
                          cursor={{ stroke: "hsl(var(--border-strong))", strokeWidth: 1 }}
                          contentStyle={{
                            backgroundColor: "hsl(var(--background))",
                            border: "1px solid hsl(var(--border-strong))",
                            borderRadius: "6px",
                            fontSize: "10px",
                            color: "hsl(var(--foreground-active))",
                          }}
                          formatter={(value: number | undefined, name: string | undefined) => [
                            value?.toFixed(4) ?? "—",
                            name === "predicted" ? "Predicted" : "Actual",
                          ]}
                        />
                        <ReferenceLine
                          segment={[
                            { x: Math.min(...evalComparisons.map((c) => c.actual)), y: Math.min(...evalComparisons.map((c) => c.actual)) },
                            { x: Math.max(...evalComparisons.map((c) => c.actual)), y: Math.max(...evalComparisons.map((c) => c.actual)) },
                          ]}
                          stroke="hsl(var(--primary))"
                          strokeDasharray="3 3"
                        />
                        <Scatter data={evalComparisons} fill="hsl(var(--warning))" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  )}
                </ChartCard>

                <ChartCard title="Residual Distribution">
                  {evaluationLoading ? (
                    <EmptyChart message="Loading evaluation data..." />
                  ) : evaluationError ? (
                    <EmptyChart message={evaluationError} />
                  ) : residualBins.length === 0 ? (
                    <EmptyChart message="No residual data available." />
                  ) : (
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={residualBins} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border-strong))" vertical={false} />
                        <XAxis dataKey="bin" hide />
                        <YAxis
                          stroke="hsl(var(--foreground-subtle))"
                          tick={{ fill: "hsl(var(--foreground-dim))", fontSize: 9, fontWeight: 600 }}
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--background))",
                            border: "1px solid hsl(var(--border-strong))",
                            borderRadius: "6px",
                            fontSize: "10px",
                            color: "hsl(var(--foreground-active))",
                          }}
                          labelFormatter={(_label, payload) => {
                            const entry = Array.isArray(payload)
                              ? payload[0]?.payload
                              : undefined;
                            return entry?.label || "Residuals";
                          }}
                          formatter={(value: number | undefined) => [value ?? 0, "Count"]}
                        />
                        <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </ChartCard>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

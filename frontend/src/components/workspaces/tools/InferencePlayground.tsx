import React, { useState } from "react";
import { FastForward, Check, AlertCircle, HardDrive } from "lucide-react";
import { useTrainingStore } from "@/store/useTrainingStore";

const API_ROOT = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
const API_URL = `${API_ROOT}/api/v1`;

type EvalComparison = {
  index: number;
  actual: number;
  predicted: number;
  abs_error: number;
  percent_error: number;
};

type EvalResult = {
  run_id: string;
  accuracy_percent: number;
  mape_percent: number;
  r2_score: number;
  count: number;
  comparisons: EvalComparison[];
};

export function InferencePlayground({
  loadedPath,
}: {
  loadedPath: string | null;
}) {
  const { isAdvancedMode, setLoadedModelPath, runs } = useTrainingStore();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [evaluation, setEvaluation] = useState<EvalResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadedModelPath, setUploadedModelPath] = useState("");
  const [uploadInfo, setUploadInfo] = useState<{
    r2?: string;
    mae?: string;
    model?: string;
  } | null>(null);

  const completedRuns = Array.isArray(runs)
    ? (runs as Array<Record<string, unknown>>).filter(
        (run) => (run.status as string | undefined) === "completed",
      )
    : [];

  const resolvedRunId = selectedRunId || loadedPath || "";

  const runEvaluation = React.useCallback(async () => {
    if (!resolvedRunId) {
      setError("Select a completed run to evaluate.");
      return;
    }
    setLoading(true);
    setError("");
    setEvaluation(null);
    try {
      const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
      const res = await fetch(`${API_URL}/training/inference/evaluate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": API_KEY,
        },
        body: JSON.stringify({
          run_id: resolvedRunId,
          model_path: uploadedModelPath || null,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Evaluation failed");
      }
      const data = await res.json();
      setEvaluation(data);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg || "Evaluation error");
    } finally {
      setLoading(false);
    }
  }, [resolvedRunId, uploadedModelPath]);

  React.useEffect(() => {
    if (resolvedRunId) {
      runEvaluation();
    }
  }, [resolvedRunId, runEvaluation]);

  const handleUploadWeights = async (file: File | null) => {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_URL}/training/load-weights`, {
        method: "POST",
        headers: { "X-API-Key": API_KEY },
        body: form,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }
      const data = await res.json();
      setUploadedModelPath(data.path || "");
      setUploadInfo(data.info || null);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg || "Upload error");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-blue-500/10 flex items-center justify-center text-blue-500">
              <FastForward size={18} />
            </div>
            <div>
              <h3 className="text-[12px] font-bold text-white">Test Set Evaluator</h3>
              <p className="text-[10px] text-zinc-500">
                Evaluate saved runs against their hold-out test split
              </p>
            </div>
          </div>
          {resolvedRunId ? (
            <div className="flex items-center gap-2 px-3 py-1 bg-green-500/10 border border-green-500/30 rounded text-[10px] font-bold text-green-400 uppercase">
              <Check size={12} /> Ready
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1 bg-red-500/10 border border-red-500/30 rounded text-[10px] font-bold text-red-400 uppercase">
              <AlertCircle size={12} /> Run Missing
            </div>
          )}
        </div>

        <div className="space-y-2 border-y border-zinc-800/50 py-4">
          <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
            <HardDrive size={12} /> Saved Models (Completed Runs)
          </label>
          <div className="flex gap-2 items-center">
            <select
              value={selectedRunId}
              onChange={(e) => {
                const next = e.target.value;
                setSelectedRunId(next);
                setEvaluation(null);
                if (next) setLoadedModelPath(next);
              }}
              className="flex-1 bg-black border border-zinc-800 rounded px-3 py-1.5 text-[11px] text-white focus:outline-none focus:border-blue-500"
              data-testid="select-completed-model"
            >
              <option value="">
                {completedRuns.length > 0 ? "Select a completed run..." : "No completed runs found"}
              </option>
              {completedRuns.map((run) => (
                <option key={String(run.run_id)} value={String(run.run_id)}>
                  {String(run.run_id)}
                </option>
              ))}
            </select>
          </div>
        </div>

        {isAdvancedMode && (
          <div className="space-y-2">
            <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
              <HardDrive size={12} /> Upload .pt (Advanced)
            </label>
            <div className="flex items-center gap-3">
              <input
                type="file"
                accept=".pt"
                onChange={(e) => handleUploadWeights(e.target.files?.[0] || null)}
                className="text-[11px] text-zinc-400"
              />
              {uploading && (
                <span className="text-[10px] text-zinc-500">Uploading...</span>
              )}
              {uploadedModelPath && (
                <span className="text-[10px] text-blue-400 font-mono">
                  {uploadedModelPath}
                </span>
              )}
            </div>
            {uploadInfo && (
              <div className="text-[10px] text-zinc-500">
                Uploaded model info: R² {uploadInfo.r2 || "N/A"} • MAE{" "}
                {uploadInfo.mae || "N/A"} • {uploadInfo.model || "Unknown"}
              </div>
            )}
          </div>
        )}

        <div className="space-y-4">
          <button
            onClick={runEvaluation}
            data-testid="btn-execute-inference"
            disabled={loading || !resolvedRunId}
            className="w-full py-2.5 bg-blue-500 hover:bg-blue-600 text-white text-[11px] font-bold rounded-lg transition-all disabled:opacity-50 disabled:grayscale shadow-lg shadow-blue-500/10"
          >
            {loading ? "COMPUTING..." : "EVALUATE TEST SET"}
          </button>
        </div>

        {error && (
          <div className="text-[11px] text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3 flex items-center gap-3">
            <AlertCircle size={14} />
            {error}
          </div>
        )}

        {evaluation && (
          <div className="space-y-4">
            <div className="bg-[hsl(var(--panel-lighter))] border border-[hsl(var(--border-muted))] rounded-lg p-5">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold text-[#52525b] uppercase tracking-widest">
                  Accuracy Summary
                </span>
                <span className="text-[10px] text-[#3b82f6] font-mono">
                  {evaluation.count} samples
                </span>
              </div>
              <div className="text-[28px] font-mono text-[#3b82f6] tracking-tighter tabular-nums leading-none">
                {evaluation.accuracy_percent.toFixed(2)}%
              </div>
              <div className="text-[10px] text-zinc-500 mt-2 flex justify-between">
                <span>MAPE: {evaluation.mape_percent.toFixed(2)}%</span>
                <span>R²: {evaluation.r2_score.toFixed(4)}</span>
              </div>
            </div>
            <div className="border border-zinc-800 rounded-lg overflow-hidden">
              <div className="bg-zinc-900/60 px-4 py-2 text-[10px] uppercase font-bold text-zinc-500">
                Sample Predictions
              </div>
              <div className="max-h-64 overflow-y-auto custom-scrollbar">
                <table className="w-full text-left text-[11px] font-mono">
                  <thead className="bg-zinc-900 text-zinc-500 font-bold">
                    <tr>
                      <th className="px-4 py-2 border-b border-zinc-800">#</th>
                      <th className="px-4 py-2 border-b border-zinc-800">
                        Actual
                      </th>
                      <th className="px-4 py-2 border-b border-zinc-800">
                        Predicted
                      </th>
                      <th className="px-4 py-2 border-b border-zinc-800">
                        Error %
                      </th>
                    </tr>
                  </thead>
                  <tbody className="text-white">
                    {evaluation.comparisons.map((row) => (
                      <tr
                        key={row.index}
                        className="border-b border-zinc-800/50"
                      >
                        <td className="px-4 py-2 text-zinc-500">{row.index}</td>
                        <td className="px-4 py-2">
                          {row.actual.toFixed(6)}
                        </td>
                        <td className="px-4 py-2">
                          {row.predicted.toFixed(6)}
                        </td>
                        <td className="px-4 py-2 text-zinc-400">
                          {row.percent_error.toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

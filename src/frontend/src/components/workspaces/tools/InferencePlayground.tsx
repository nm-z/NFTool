import React, { useState } from "react";
import { FastForward, Check, AlertCircle, HardDrive } from "lucide-react";
import { useTrainingStore } from "@/store/useTrainingStore";
import { useApi } from "@/components/ApiProvider";
import type { EvalResult } from "@/components/workspaces/LibraryWorkspace";

export function InferencePlayground({
  loadedPath,
}: {
  loadedPath: string | null;
}) {
  const { apiUrl: API_URL } = useApi();
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
      <div className="bg-[hsl(var(--panel-lighter)/0.5)] border border-[hsl(var(--border))] rounded-xl p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-[hsl(var(--primary)/0.1)] flex items-center justify-center text-[hsl(var(--primary))]">
              <FastForward size={18} />
            </div>
            <div>
              <h3 className="text-[12px] font-bold text-[hsl(var(--foreground-active))]">
                Test Set Evaluator
              </h3>
              <p className="text-[10px] text-[hsl(var(--foreground-muted))]">
                Evaluate saved runs against their hold-out test split
              </p>
            </div>
          </div>
          {resolvedRunId ? (
            <div className="flex items-center gap-2 px-3 py-1 bg-[hsl(var(--success)/0.1)] border border-[hsl(var(--success)/0.3)] rounded text-[10px] font-bold text-[hsl(var(--success))] uppercase">
              <Check size={12} /> Ready
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1 bg-[hsl(var(--danger)/0.1)] border border-[hsl(var(--danger)/0.3)] rounded text-[10px] font-bold text-[hsl(var(--danger))] uppercase">
              <AlertCircle size={12} /> Run Missing
            </div>
          )}
        </div>

        <div className="space-y-2 border-y border-[hsl(var(--border)/0.5)] py-4">
          <label className="text-[10px] font-bold text-[hsl(var(--foreground-muted))] uppercase tracking-widest flex items-center gap-2">
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
              className="flex-1 bg-[hsl(var(--input))] border border-[hsl(var(--border))] rounded px-3 py-1.5 text-[11px] text-[hsl(var(--foreground-active))] focus:outline-none focus:border-[hsl(var(--primary))]"
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
            <label className="text-[10px] font-bold text-[hsl(var(--foreground-muted))] uppercase tracking-widest flex items-center gap-2">
              <HardDrive size={12} /> Upload .pt (Advanced)
            </label>
            <div className="flex items-center gap-3">
              <input
                type="file"
                accept=".pt"
                onChange={(e) => handleUploadWeights(e.target.files?.[0] || null)}
                className="text-[11px] text-[hsl(var(--foreground))]"
              />
              {uploading && (
                <span className="text-[10px] text-[hsl(var(--foreground-muted))]">
                  Uploading...
                </span>
              )}
              {uploadedModelPath && (
                <span className="text-[10px] text-[hsl(var(--primary-soft))] font-mono">
                  {uploadedModelPath}
                </span>
              )}
            </div>
            {uploadInfo && (
              <div className="text-[10px] text-[hsl(var(--foreground-muted))]">
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
            className="w-full py-2.5 bg-[hsl(var(--primary))] hover:bg-[hsl(var(--primary-strong))] text-[hsl(var(--foreground-active))] text-[11px] font-bold rounded-lg transition-all disabled:opacity-50 disabled:grayscale shadow-[0_10px_15px_-3px_hsl(var(--primary)/0.12),0_4px_6px_-4px_hsl(var(--primary)/0.12)]"
          >
            {loading ? "COMPUTING..." : "EVALUATE TEST SET"}
          </button>
        </div>

        {error && (
          <div className="text-[11px] text-[hsl(var(--danger))] bg-[hsl(var(--danger)/0.1)] border border-[hsl(var(--danger)/0.2)] rounded-lg px-4 py-3 flex items-center gap-3">
            <AlertCircle size={14} />
            {error}
          </div>
        )}

        {evaluation && (
          <div className="space-y-4">
            <div className="bg-[hsl(var(--panel-lighter))] border border-[hsl(var(--border-muted))] rounded-lg p-5">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold text-[hsl(var(--foreground-dim))] uppercase tracking-widest">
                  Accuracy Summary
                </span>
                <span className="text-[10px] text-[hsl(var(--primary))] font-mono">
                  {evaluation.count} samples
                </span>
              </div>
              <div className="text-[28px] font-mono text-[hsl(var(--primary))] tracking-tighter tabular-nums leading-none">
                {evaluation.accuracy_percent.toFixed(2)}%
              </div>
              <div className="text-[10px] text-[hsl(var(--foreground-muted))] mt-2 space-y-1">
                <div className="flex justify-between">
                  <span>MAE: {evaluation.mae.toFixed(6)}</span>
                  <span>RMSE: {evaluation.rmse.toFixed(6)}</span>
                </div>
                <div className="flex justify-between">
                  <span>R²: {evaluation.r2_score.toFixed(4)}</span>
                </div>
              </div>
            </div>
            <div className="border border-[hsl(var(--border))] rounded-lg overflow-hidden">
              <div className="bg-[hsl(var(--panel-lighter)/0.6)] px-4 py-2 text-[10px] uppercase font-bold text-[hsl(var(--foreground-muted))]">
                Sample Predictions
              </div>
              <div className="max-h-64 overflow-y-auto custom-scrollbar">
                <table className="w-full text-left text-[11px] font-mono">
                  <thead className="bg-[hsl(var(--panel-lighter))] text-[hsl(var(--foreground-muted))] font-bold">
                    <tr>
                      <th className="px-4 py-2 border-b border-[hsl(var(--border))]">#</th>
                      <th className="px-4 py-2 border-b border-[hsl(var(--border))]">
                        Actual
                      </th>
                      <th className="px-4 py-2 border-b border-[hsl(var(--border))]">
                        Predicted
                      </th>
                      <th className="px-4 py-2 border-b border-[hsl(var(--border))]">
                        Error %
                      </th>
                    </tr>
                  </thead>
                  <tbody className="text-[hsl(var(--foreground-active))]">
                    {evaluation.comparisons.map((row) => (
                      <tr
                        key={row.index}
                        className="border-b border-[hsl(var(--border)/0.5)]"
                      >
                        <td className="px-4 py-2 text-[hsl(var(--foreground-muted))]">
                          {row.index}
                        </td>
                        <td className="px-4 py-2">
                          {row.actual.toFixed(6)}
                        </td>
                        <td className="px-4 py-2">
                          {row.predicted.toFixed(6)}
                        </td>
                        <td className="px-4 py-2 text-[hsl(var(--foreground))]">
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

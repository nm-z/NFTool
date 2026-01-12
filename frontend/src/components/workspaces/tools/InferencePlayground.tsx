import React, { useState } from "react";
import { FastForward, Check, AlertCircle, HardDrive } from "lucide-react";
import { useTrainingStore } from "@/store/useTrainingStore";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

export function InferencePlayground({
  loadedPath,
}: {
  loadedPath: string | null;
}) {
  const { isAdvancedMode, setLoadedModelPath } = useTrainingStore();
  const [features, setFeatures] = useState("");
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [manualPath, setManualPath] = useState(loadedPath || "");

  const runInference = async () => {
    const targetPath = isAdvancedMode ? manualPath : loadedPath;
    if (!targetPath || !features) {
      setError("Please load model weights first");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const featureArray = features.split(",").map((f) => parseFloat(f.trim()));
      const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";

      const res = await fetch(`${API_URL}/inference`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": API_KEY,
        },
        body: JSON.stringify({
          model_path: targetPath,
          features: featureArray,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Inference failed");
      }
      const data = await res.json();
      setPrediction(data.prediction);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg || "Inference error");
    } finally {
      setLoading(false);
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
              <h3 className="text-[12px] font-bold text-white">
                Prediction Engine
              </h3>
              <p className="text-[10px] text-zinc-500">
                Execute single-row inference on active weights
              </p>
            </div>
          </div>
          {(isAdvancedMode ? manualPath : loadedPath) ? (
            <div className="flex items-center gap-2 px-3 py-1 bg-green-500/10 border border-green-500/30 rounded text-[10px] font-bold text-green-400 uppercase">
              <Check size={12} /> Live
            </div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1 bg-red-500/10 border border-red-500/30 rounded text-[10px] font-bold text-red-400 uppercase">
              <AlertCircle size={12} /> Weights Missing
            </div>
          )}
        </div>

        {isAdvancedMode && (
          <div className="space-y-2 border-y border-zinc-800/50 py-4">
            <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
              <HardDrive size={12} /> Model Path (Advanced/Automation)
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={manualPath}
                onChange={(e) => setManualPath(e.target.value)}
                placeholder="e.g. workspace/runs/reports/PASS_.../best_model.pt"
                className="flex-1 bg-black border border-zinc-800 rounded px-3 py-1.5 text-[11px] text-blue-400 font-mono focus:outline-none focus:border-blue-500"
                data-testid="input-manual-model-path"
              />
              <button
                onClick={() => setLoadedModelPath(manualPath)}
                className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-white text-[10px] font-bold rounded transition-colors"
              >
                SET
              </button>
            </div>
          </div>
        )}

        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">
              Input Features (CSV Vector)
            </label>
            <textarea
              value={features}
              onChange={(e) => setFeatures(e.target.value)}
              placeholder="Enter comma-separated features (e.g., 0.12, 0.45, -0.11...)"
              rows={4}
              className="w-full bg-black border border-zinc-800 rounded-lg px-4 py-3 text-[12px] text-white font-mono focus:outline-none focus:border-blue-500 placeholder-zinc-700 transition-all"
              data-testid="textarea-inference-features"
            />
          </div>
          <button
            onClick={runInference}
            data-testid="btn-execute-inference"
            disabled={loading || !(isAdvancedMode ? manualPath : loadedPath)}
            className="w-full py-2.5 bg-blue-500 hover:bg-blue-600 text-white text-[11px] font-bold rounded-lg transition-all disabled:opacity-50 disabled:grayscale shadow-lg shadow-blue-500/10"
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
          <div className="bg-[hsl(var(--panel-lighter))] border border-[hsl(var(--border-muted))] rounded-lg p-5 transition-all animate-in fade-in slide-in-from-bottom-2">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-bold text-[#52525b] uppercase tracking-widest">
                Inference Output
              </span>
              <span className="text-[10px] text-[#3b82f6] font-mono">
                FLOAT64
              </span>
            </div>
            <div className="text-[32px] font-mono text-[#3b82f6] tracking-tighter tabular-nums leading-none">
              {prediction?.toFixed(8) || "0.00000000"}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

import React, { useState } from "react";
import { FastForward, Check, AlertCircle } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

export function InferencePlayground({ loadedPath }: { loadedPath: string | null }) {
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
      const res = await fetch(`${API_URL}/inference`, {
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
      <div className="bg-[hsl(var(--panel))]/50 border border-[hsl(var(--border))] rounded-xl p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-[#3b82f6]/10 flex items-center justify-center text-[#3b82f6]">
              <FastForward size={18} />
            </div>
            <div>
              <h3 className="text-[12px] font-bold text-[hsl(var(--foreground-active))]">Prediction Engine</h3>
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
              className="w-full bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded-lg px-4 py-3 text-[12px] text-[hsl(var(--foreground-active))] font-mono focus:outline-none focus:border-[#3b82f6] placeholder-[#3f3f46] transition-all"
            />
          </div>
          <button
            onClick={runInference}
            disabled={loading || !loadedPath}
            className="w-full py-2.5 bg-[#3b82f6] hover:bg-[#2563eb] text-[hsl(var(--foreground-active))] text-[11px] font-bold rounded-lg transition-all disabled:opacity-50 disabled:grayscale shadow-lg shadow-[#3b82f6]/10"
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
              <span className="text-[10px] font-bold text-[#52525b] uppercase tracking-widest">Inference Output</span>
              <span className="text-[10px] text-[#3b82f6] font-mono">FLOAT64</span>
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

import React, { useState, useEffect } from "react";
import { FolderOpen, AlertCircle, Database, LayoutGrid, Info } from "lucide-react";
import { SummaryCard } from "../../common/Cards";

const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

export function DatasetPreview({ initialPath }: { initialPath?: string }) {
  const [previewData, setPreviewData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filePath, setFilePath] = useState(initialPath || "");

  useEffect(() => {
    if (initialPath) setFilePath(initialPath);
  }, [initialPath]);

  const loadPreview = async () => {
    if (!filePath) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/dataset/preview?path=${encodeURIComponent(filePath)}&rows=20`, {
        headers: { "X-API-Key": API_KEY }
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || `Preview failed: ${res.status}`);
      }
      const data = await res.json();
      setPreviewData(data);
    } catch (e: any) {
      console.error("Failed to load preview:", e);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (filePath) loadPreview();
  }, [filePath]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 bg-[hsl(var(--panel))] p-3 rounded-lg border border-[hsl(var(--border))]">
        <FolderOpen size={16} className="text-[#3b82f6]" />
        <input
          type="text"
          value={filePath}
          onChange={(e) => setFilePath(e.target.value)}
          placeholder="Dataset path..."
          className="flex-1 bg-transparent border-none outline-none text-[11px] text-[hsl(var(--foreground-active))] placeholder-[#3f3f46]"
        />
        <button
          onClick={loadPreview}
          data-testid="btn-preview-dataset"
          disabled={loading}
          className="px-4 py-1.5 bg-[#3b82f6]/10 text-[#3b82f6] border border-[#3b82f6]/30 hover:bg-[#3b82f6] hover:text-[hsl(var(--foreground-active))] text-[11px] font-bold rounded transition-all disabled:opacity-50"
        >
          {loading ? "Loading..." : "Preview"}
        </button>
      </div>

      {previewData && (
        <div className="grid grid-cols-4 gap-4">
          <SummaryCard icon={Database} label="Total Samples" value={previewData.total_rows.toLocaleString()} subValue="Row Count" />
          <SummaryCard icon={LayoutGrid} label="Features" value={previewData.shape[1].toString()} subValue="Column Count" />
          <SummaryCard icon={Info} label="Missing Values" value={previewData.stats.missing.toString()} subValue="Data Integrity" />
          <SummaryCard icon={LayoutGrid} label="Memory Size" value={`${((previewData.total_rows * previewData.shape[1] * 8) / 1024 / 1024).toFixed(2)} MB`} subValue="Estimated Heap" />
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 flex items-center gap-3 text-red-400 text-[11px] font-medium">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      {previewData && (
        <div className="border border-[hsl(var(--border))] rounded-lg overflow-hidden bg-[hsl(var(--panel))]/30">
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full text-left border-collapse" suppressHydrationWarning>
              <thead className="bg-[hsl(var(--panel))] text-[10px] text-[#52525b] uppercase font-bold">
                <tr>
                  {previewData.headers.map((h: string, i: number) => (
                    <th key={i} className="px-3 py-2 border-b border-[hsl(var(--border))] font-mono whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="text-[11px] text-[hsl(var(--foreground))] font-mono">
                {previewData.rows.map((row: any[], i: number) => (
                  <tr key={i} className="hover:bg-[hsl(var(--panel-lighter))]/30 group transition-colors">
                    {row.map((cell: any, j: number) => (
                      <td key={j} className="px-3 py-1.5 border-b border-[hsl(var(--border))]/50 group-hover:text-[hsl(var(--foreground-active))] whitespace-nowrap">{typeof cell === 'number' ? cell.toFixed(4) : cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
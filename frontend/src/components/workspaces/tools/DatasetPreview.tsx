import React, { useState, useEffect, useCallback } from "react";
import {
  FolderOpen,
  AlertCircle,
  Database,
  LayoutGrid,
  Info,
} from "lucide-react";
import { SummaryCard } from "../../common/Cards";
import { useApi } from "@/components/ApiProvider";

const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";

export function DatasetPreview({ initialPath }: { initialPath?: string }) {
  const { apiUrl } = useApi();
  type PreviewData = {
    headers: string[];
    rows: Array<Array<string | number | null>>;
    total_rows: number;
    shape: [number, number];
    stats: { missing: number };
  };
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filePath, setFilePath] = useState(initialPath || "");

  useEffect(() => {
    if (initialPath) setFilePath(initialPath);
  }, [initialPath]);

  const loadPreview = useCallback(
    async (path?: string) => {
      const targetPath = path ?? filePath;
      if (!targetPath) return;
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `${apiUrl}/data/dataset/preview?path=${encodeURIComponent(targetPath)}&rows=20`,
          {
            headers: { "X-API-Key": API_KEY },
          },
        );

        if (!res.ok) {
          // Try to parse JSON error body, otherwise fall back to text or status
          let errText = `Preview failed: ${res.status}`;
          try {
            const errBody = await res.json();
            if (errBody && (errBody.detail || errBody.message)) {
              errText = String(errBody.detail || errBody.message);
            } else if (typeof errBody === "string" && errBody) {
              errText = errBody;
            }
          } catch {
            // Not JSON — try text
            const textBody = await res.text().catch(() => "");
            if (textBody) errText = textBody;
          }

          // Surface friendly message in UI instead of throwing (avoids console stack traces)
          setError(errText);
          setPreviewData(null);
          return;
        }

        const data = await res.json();
        setPreviewData(data);
      } catch (e: unknown) {
        // Log minimal info for debugging and show a friendly message to the user
        const msg = e instanceof Error ? e.message : String(e);
        console.error("Failed to load preview:", msg);
        setError(msg || "Failed to load preview");
      } finally {
        setLoading(false);
      }
    },
    [apiUrl, filePath],
  );

  useEffect(() => {
    if (filePath) loadPreview(filePath);
  }, [filePath, loadPreview]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 bg-[hsl(var(--panel))] p-3 rounded-lg border border-[hsl(var(--border))]">
        <FolderOpen size={16} className="text-[hsl(var(--primary))]" />
        <input
          type="text"
          value={filePath}
          onChange={(e) => setFilePath(e.target.value)}
          placeholder="Dataset path..."
          data-testid="input-dataset-path"
          className="flex-1 bg-transparent border-none outline-none text-[11px] text-[hsl(var(--foreground-active))] placeholder-[hsl(var(--foreground-subtle))]"
        />
        <button
          onClick={() => loadPreview()}
          data-testid="btn-preview-dataset"
          disabled={loading}
          className="px-4 py-1.5 bg-[hsl(var(--primary)/0.1)] text-[hsl(var(--primary))] border border-[hsl(var(--primary)/0.3)] hover:bg-[hsl(var(--primary))] hover:text-[hsl(var(--foreground-active))] text-[11px] font-bold rounded transition-all disabled:opacity-50"
        >
          {loading ? "Loading..." : "Preview"}
        </button>
      </div>

      {previewData && (
        <div className="grid grid-cols-4 gap-4">
          <SummaryCard
            icon={Database}
            label="Total Samples"
            value={previewData.total_rows.toLocaleString()}
            subValue="Row Count"
          />
          <SummaryCard
            icon={LayoutGrid}
            label="Features"
            value={previewData.shape[1].toString()}
            subValue="Column Count"
          />
          <SummaryCard
            icon={Info}
            label="Missing Values"
            value={previewData.stats.missing.toString()}
            subValue="Data Integrity"
          />
          <SummaryCard
            icon={LayoutGrid}
            label="Memory Size"
            value={`${((previewData.total_rows * previewData.shape[1] * 8) / 1024 / 1024).toFixed(2)} MB`}
            subValue="Estimated Heap"
          />
        </div>
      )}

      {error && (
        <div className="bg-[hsl(var(--danger)/0.1)] border border-[hsl(var(--danger)/0.2)] rounded-lg p-3 flex items-center gap-3 text-[hsl(var(--danger))] text-[11px] font-medium">
          <AlertCircle size={14} />
          <span className="flex-1">{error}</span>
          <button
            onClick={() => loadPreview(filePath)}
            disabled={loading}
            className="px-3 py-1 rounded text-[11px] font-medium bg-[hsl(var(--danger)/0.1)] border border-[hsl(var(--danger)/0.2)] hover:bg-[hsl(var(--danger)/0.2)] disabled:opacity-50"
          >
            Retry
          </button>
        </div>
      )}

      {previewData && (
        <div className="border border-[hsl(var(--border))] rounded-lg overflow-hidden bg-[hsl(var(--panel))]/30">
          <div className="overflow-x-auto custom-scrollbar">
            <table
              className="w-full text-left border-collapse"
              suppressHydrationWarning
            >
              <thead className="bg-[hsl(var(--panel))] text-[10px] text-[hsl(var(--foreground-dim))] uppercase font-bold">
                <tr>
                  {previewData.headers.map((h: string, i: number) => (
                    <th
                      key={i}
                      className="px-3 py-2 border-b border-[hsl(var(--border))] font-mono whitespace-nowrap"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="text-[11px] text-[hsl(var(--foreground))] font-mono">
                {previewData.rows.map(
                  (row: (string | number | null)[], i: number) => (
                    <tr
                      key={i}
                      className="hover:bg-[hsl(var(--panel-lighter))]/30 group transition-colors"
                    >
                      {row.map((cell: string | number | null, j: number) => (
                        <td
                          key={j}
                          className="px-3 py-1.5 border-b border-[hsl(var(--border))]/50 group-hover:text-[hsl(var(--foreground-active))] whitespace-nowrap"
                        >
                          {typeof cell === "number" ? cell.toFixed(4) : cell}
                        </td>
                      ))}
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

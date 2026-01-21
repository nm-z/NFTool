import React from "react";
import * as Tabs from "@radix-ui/react-tabs";
import archy from "archy";
import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Activity,
  Table,
  History as HistoryIcon,
  Terminal,
  Trash2,
  Copy,
} from "lucide-react";
import {
  Panel,
  Group as PanelGroup,
  Separator as PanelResizeHandle,
} from "react-resizable-panels";

import { SummaryCard } from "../common/Cards";
import { TabTrigger } from "../common/UIComponents";
import { InferencePlayground } from "./tools/InferencePlayground";
import { useTrainingStore } from "@/store/useTrainingStore";

export function TrainWorkspace() {
  const {
    metricsHistory,
    isRunning,
    currentTrial,
    totalTrials,
    result,
    logs,
    clearLogs,
    loadedModelPath,
    split,
    setSplit,
    selectedPredictor,
    setMetricsHistory,
    setResult,
    setProgress,
    setTrialInfo,
    setIsRunning,
    runs,
    seed,
  } = useTrainingStore();

  const [isMounted, setIsMounted] = React.useState(false);
  const [copiedLogs, setCopiedLogs] = React.useState(false);
  const [streamView, setStreamView] = React.useState<"log" | "tree">("log");
  const logEndRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    setIsMounted(true);
  }, []);

  // Reset transient UI state when opening the Train workspace unless a model or dataset
  // was intentionally selected (e.g., deploying from Library). This avoids showing
  // leftover logs/metrics from previous sessions while preserving deliberate deploy behavior.
  React.useEffect(() => {
    if (!loadedModelPath && !selectedPredictor) {
      clearLogs();
      setMetricsHistory([]);
      setResult(null);
      setProgress(0);
      setTrialInfo(0, 0);
      setIsRunning(false);
      console.debug("TrainWorkspace: transient UI reset", {
        loadedModelPath,
        selectedPredictor,
      });
    }
  }, [
    loadedModelPath,
    selectedPredictor,
    clearLogs,
    setMetricsHistory,
    setResult,
    setProgress,
    setTrialInfo,
    setIsRunning,
  ]);

  React.useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
    console.debug("TrainWorkspace: logs updated", {
      length: logs?.length ?? 0,
    });
  }, [logs]);


  const handleCopyLogs = async () => {
    if (streamView === "tree") {
      if (!archyText) return;
      try {
        await navigator.clipboard.writeText(archyText.trimEnd());
        setCopiedLogs(true);
        window.setTimeout(() => setCopiedLogs(false), 1500);
      } catch (err) {
        console.warn("Failed to copy logs:", err);
      }
      return;
    }
    if (!logs || logs.length === 0) return;
    const payload = logs
      .map((log) => {
        const ts = log.time ? `[${log.time}]` : "";
        const epoch =
          log.epoch !== null && log.epoch !== undefined
            ? ` (Epoch ${log.epoch})`
            : "";
        return `${ts}${epoch} ${log.msg}`.trim();
      })
      .join("\n");
    try {
      await navigator.clipboard.writeText(payload);
      setCopiedLogs(true);
      window.setTimeout(() => setCopiedLogs(false), 1500);
    } catch (err) {
      console.warn("Failed to copy logs:", err);
    }
  };

  const handleStreamMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    if (e.target !== e.currentTarget) return;
    const selection = window.getSelection();
    if (selection && selection.type === "Range") {
      selection.removeAllRanges();
    }
  };

  const renderTreeLine = (line: string, idx: number) => {
    const match = line.match(/^([│├└─┬\s]*)(.*)$/);
    const prefix = match?.[1] ?? "";
    const content = match?.[2] ?? line;

    const renderContent = () => {
      if (content.startsWith("Run ")) {
        return <span className="text-cyan-300">{content}</span>;
      }
      if (content.startsWith("Config")) {
        return <span className="text-amber-300">{content}</span>;
      }
      if (content.startsWith("Trials")) {
        return <span className="text-emerald-300">{content}</span>;
      }
      if (content.startsWith("Trial ")) {
        return <span className="text-fuchsia-300">{content}</span>;
      }
      if (content.startsWith("Params:")) {
        const [, rest = ""] = content.split("Params:");
        return (
          <>
            <span className="text-sky-300">Params:</span>
            <span className="text-zinc-200">{rest}</span>
          </>
        );
      }
      if (content.startsWith("Summary:")) {
        const [, rest = ""] = content.split("Summary:");
        return (
          <>
            <span className="text-orange-300">Summary:</span>
            <span className="text-zinc-200">{rest}</span>
          </>
        );
      }
      if (content.startsWith("Epoch ")) {
        const epochMatch = content.match(/^(Epoch\s+\d+)(.*)$/);
        const epochLabel = epochMatch?.[1] ?? content;
        const rest = epochMatch?.[2] ?? "";
        return (
          <>
            <span className="text-cyan-200">{epochLabel}</span>
            {rest && (
              <span className="text-zinc-100">
                {rest
                  .replace("R²", "R²")
                  .replace("val", "val")
                  .replace("MAE", "MAE")
                  .split(/(R²|val|MAE)/g)
                  .filter(Boolean)
                  .map((part, i) => {
                    if (part === "R²") {
                      return (
                        <span key={`${idx}-r2-${i}`} className="text-purple-300">
                          {" "}
                          {part}
                        </span>
                      );
                    }
                    if (part === "val") {
                      return (
                        <span key={`${idx}-val-${i}`} className="text-rose-300">
                          {" "}
                          {part}
                        </span>
                      );
                    }
                    if (part === "MAE") {
                      return (
                        <span key={`${idx}-mae-${i}`} className="text-amber-200">
                          {" "}
                          {part}
                        </span>
                      );
                    }
                    return (
                      <span key={`${idx}-seg-${i}`} className="text-zinc-100">
                        {part}
                      </span>
                    );
                  })}
              </span>
            )}
          </>
        );
      }
      return <span className="text-zinc-200">{content}</span>;
    };

    return (
      <div key={`tree-${idx}`} className="leading-snug">
        <span className="text-zinc-600">{prefix}</span>
        {renderContent()}
      </div>
    );
  };

  const datasetName = selectedPredictor
    ? selectedPredictor.split("/").pop()
    : "None Selected";
  const displayTotalTrials = totalTrials;
  const displayCurrentTrial =
    displayTotalTrials > 0 ? Math.min(currentTrial + 1, displayTotalTrials) : currentTrial;
  const chartData = React.useMemo(
    () =>
      (metricsHistory ?? []).map((point, index) => ({
        ...point,
        step: index,
      })),
    [metricsHistory],
  );

  const activeRun = React.useMemo(() => {
    if (!Array.isArray(runs) || runs.length === 0) return undefined;
    return (
      runs.find((r) => (r.status as string) === "running") ||
      runs.find((r) => (r.status as string) === "queued") ||
      runs[0]
    );
  }, [runs]);

  const trialParamLogs = React.useMemo(() => {
    const map = new Map<number, string>();
    const regex = /Trial\s+(\d+)\/\d+\s+params:\s+(.*)$/i;
    logs?.forEach((log) => {
      const msg = log.msg ?? "";
      const match = msg.match(regex);
      if (!match) return;
      const trialNum = Math.max(1, parseInt(match[1] || "1", 10));
      map.set(trialNum, match[2] || "");
    });
    return map;
  }, [logs]);

  const trialSummaryLogs = React.useMemo(() => {
    const map = new Map<number, string>();
    const regex = /Trial\s+(\d+)\/\d+\s+summary:\s+(.*)$/i;
    logs?.forEach((log) => {
      const msg = log.msg ?? "";
      const match = msg.match(regex);
      if (!match) return;
      const trialNum = Math.max(1, parseInt(match[1] || "1", 10));
      map.set(trialNum, match[2] || "");
    });
    return map;
  }, [logs]);

  const archyText = React.useMemo(() => {
    type ArchyNode = { label: string; nodes?: Array<ArchyNode | string> };

    type MetricPoint = (typeof metricsHistory)[number];
    type TrialPoint = { point: MetricPoint; index: number };
    const trialGroups = new Map<number, TrialPoint[]>();
    (metricsHistory ?? []).forEach((point, index) => {
      const trialNum = typeof point.trial === "number" ? point.trial + 1 : 1;
      const existing = trialGroups.get(trialNum) || [];
      existing.push({ point, index });
      trialGroups.set(trialNum, existing);
    });

    const trialNodes: ArchyNode[] = Array.from(trialGroups.entries())
      .sort(([a], [b]) => a - b)
      .map(([trialNum, points]) => {
        const epochMap = new Map<number, TrialPoint>();
        points.forEach((entry) => {
          const epochVal =
            typeof entry.point.epoch === "number" ? entry.point.epoch : entry.index;
          if (!epochMap.has(epochVal)) {
            epochMap.set(epochVal, entry);
          }
        });

        const epochLines = Array.from(epochMap.entries())
          .sort(([a], [b]) => a - b)
          .map(([epochVal, { point }]) => {
            const r2 = typeof point.r2 === "number" ? point.r2.toFixed(4) : "—";
            const vloss =
              typeof point.val_loss === "number" ? point.val_loss.toFixed(6) : "—";
            const mae = typeof point.mae === "number" ? point.mae.toFixed(4) : "—";
            return `Epoch ${epochVal} R² ${r2} • val ${vloss} • MAE ${mae}`;
          });

        const params = trialParamLogs.get(trialNum);
        const summary = trialSummaryLogs.get(trialNum);

        const nodes: Array<ArchyNode | string> = [];
        if (params) nodes.push(`Params: ${params}`);
        if (summary) nodes.push(`Summary: ${summary}`);
        if (epochLines.length > 0) {
          nodes.push({
            label: `Epochs (${epochLines.length})`,
            nodes: epochLines,
          });
        }

        return {
          label: `Trial ${trialNum}`,
          nodes,
        };
      });

    const runConfig = (activeRun as Record<string, unknown> | undefined)?.config as
      | Record<string, unknown>
      | undefined;
    const configSeed =
      typeof runConfig?.seed === "number" ? runConfig.seed : parseInt(seed) || 0;
    const configTrain =
      typeof runConfig?.train_ratio === "number"
        ? runConfig.train_ratio
        : split / 100;
    const configVal =
      typeof runConfig?.val_ratio === "number"
        ? runConfig.val_ratio
        : (100 - split) / 200;
    const configTest =
      typeof runConfig?.test_ratio === "number"
        ? runConfig.test_ratio
        : (100 - split) / 200;
    const splitId =
      typeof runConfig?.split_id === "string" && runConfig.split_id.trim()
        ? runConfig.split_id
        : null;
    const foldIndex =
      typeof runConfig?.fold_index === "number" ? runConfig.fold_index : null;

    const configNodes: Array<ArchyNode | string> = [
      `Seed: ${configSeed}`,
      `Split: train ${configTrain.toFixed(2)}, val ${configVal.toFixed(
        2,
      )}, test ${configTest.toFixed(2)}`,
    ];
    if (splitId) configNodes.push(`Split ID: ${splitId}`);
    if (foldIndex !== null) configNodes.push(`Fold: ${foldIndex}`);

    const runLabel = activeRun?.run_id ? `Run ${activeRun.run_id}` : "Run";
    const tree: ArchyNode = {
      label: runLabel,
      nodes: [
        { label: "Config", nodes: configNodes },
        {
          label: `Trials (${trialNodes.length})`,
          nodes: trialNodes.length > 0 ? trialNodes : ["No trials yet"],
        },
      ],
    };

    try {
      return archy(tree);
    } catch {
      return "";
    }
  }, [activeRun, logs, metricsHistory, seed, split, trialParamLogs, trialSummaryLogs]);


  return (
    <div
      className="flex flex-col h-full bg-zinc-950 outline-none"
      tabIndex={-1}
    >
      <Tabs.Root defaultValue="optimization" className="flex flex-col h-full">
        <div className="h-12 border-b border-zinc-800 flex items-center justify-between px-6 bg-zinc-900/50 shrink-0">
          <Tabs.List className="flex gap-8">
            <TabTrigger value="optimization" label="Optimization" />
            <TabTrigger value="inference" label="Inference Playground" />
          </Tabs.List>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
              <span className="text-[10px] font-bold text-zinc-500 uppercase">
                Split
              </span>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  value={split}
                  onChange={(e) => setSplit(parseInt(e.target.value))}
                  min="50"
                  max="95"
                  step="5"
                  className="w-24 h-1 bg-zinc-800 rounded-full appearance-none accent-[hsl(var(--primary))] cursor-pointer"
                />
                <span className="text-[10px] font-mono text-[hsl(var(--primary))] w-8">
                  {split}%
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-hidden relative">
          <Tabs.Content
            value="optimization"
            className="h-full flex flex-col data-[state=inactive]:hidden"
          >
            <PanelGroup orientation="vertical">
              <Panel defaultSize={70} minSize={30}>
                <div className="h-full overflow-y-auto custom-scrollbar p-6 space-y-6">
                  <div className="grid grid-cols-3 gap-4">
                    <SummaryCard
                      icon={Table}
                      label="Active Dataset"
                      value={datasetName || "—"}
                      subValue="Predictor Source"
                    />
                    <SummaryCard
                      icon={Activity}
                      label="Best R² Score"
                      value={result?.best_r2?.toFixed(4) || "0.0000"}
                      subValue="Optimization Peak"
                    />
                    <SummaryCard
                      icon={HistoryIcon}
                      label="Engine Status"
                      value={isRunning ? "Active" : "Idle"}
                      subValue={
                        isRunning
                          ? `Trial ${displayCurrentTrial}/${displayTotalTrials}`
                          : "Awaiting Run"
                      }
                    />
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-[11px] uppercase font-bold tracking-widest text-zinc-500">
                        Live Metrics History
                      </h3>
                    <div className="flex items-center gap-4 text-[10px] font-mono">
                        <div data-testid="metrics-count" className="text-[10px] text-zinc-400">
                          {metricsHistory?.length ?? 0} points
                        </div>
                        <span className="text-white flex items-center gap-1.5">
                          <div className="w-2 h-2 rounded-full bg-[hsl(var(--danger))]"></div>{" "}
                          Loss
                        </span>
                        <span className="text-white flex items-center gap-1.5">
                          <div className="w-2 h-2 rounded-full bg-[hsl(var(--primary))]"></div>{" "}
                          R²
                        </span>
                      </div>
                    </div>
                    <div data-testid="metrics-chart-container" className="border border-zinc-800 rounded-lg p-4 bg-black min-h-[240px] flex items-center justify-center">
                      {isMounted ? (
                        <ResponsiveContainer width="100%" height={240}>
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
                                const entry = Array.isArray(payload) ? payload[0]?.payload : undefined;
                                if (!entry) return "Step";
                                const trialLabel =
                                  entry.trial != null ? `Trial ${entry.trial + 1}` : "Trial ?";
                                const epochLabel =
                                  entry.epoch != null ? `Epoch ${entry.epoch}` : "Epoch ?";
                                return `${trialLabel} • ${epochLabel}`;
                              }}
                              formatter={(value: number, name: string) => {
                                const precision = name === "loss" || name === "r2" ? 6 : 4;
                                return [value?.toFixed(precision) ?? "—", name === "loss" ? "Loss" : "R²"];
                              }}
                            />
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="loss"
                              stroke="hsl(var(--danger))"
                              strokeWidth={2}
                              dot={false}
                              isAnimationActive={false}
                              activeDot={{ r: 4, strokeWidth: 0 }}
                              connectNulls={true}
                            />
                            <Line
                              yAxisId="right"
                              type="monotone"
                              dataKey="r2"
                              stroke="hsl(var(--primary))"
                              strokeWidth={2}
                              dot={false}
                              isAnimationActive={false}
                              activeDot={{ r: 4, strokeWidth: 0 }}
                              connectNulls={true}
                            />
                          </RechartsLineChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="text-zinc-800 font-mono text-[10px]">
                          Initializing Charts...
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h3 className="text-[11px] uppercase font-bold tracking-widest text-zinc-500">
                      Optimization Results
                    </h3>
                    <div className="border border-zinc-800 rounded-lg overflow-hidden bg-zinc-900/30">
                      <table
                        className="w-full text-left text-[11px]"
                        suppressHydrationWarning
                      >
                        <thead className="bg-zinc-900 text-zinc-500 font-bold">
                          <tr>
                            <th className="px-4 py-2 border-b border-zinc-800">
                              TRIAL
                            </th>
                            <th className="px-4 py-2 border-b border-zinc-800">
                              EPOCH
                            </th>
                            <th className="px-4 py-2 border-b border-zinc-800">
                              R²
                            </th>
                            <th className="px-4 py-2 border-b border-zinc-800">
                              VAL_LOSS
                            </th>
                            <th className="px-4 py-2 border-b border-zinc-800">
                              MAE
                            </th>
                          </tr>
                        </thead>
                        <tbody className="text-white font-mono">
                          {metricsHistory
                            ?.slice(-10)
                            .reverse()
                            .map((m, i: number) => (
                              <tr
                                key={i}
                                className="border-b border-zinc-800/50 hover:bg-zinc-800/30"
                              >
                                <td className="px-4 py-2 text-zinc-500">
                                  #{m.trial + 1}
                                </td>
                                <td className="px-4 py-2 text-zinc-500">
                                  {m.epoch ?? "-"}
                                </td>
                                <td className="px-4 py-2 text-white">
                                  {m.r2?.toFixed(4) || "0.0000"}
                                </td>
                                <td className="px-4 py-2">
                                  {m.val_loss?.toFixed(6) || "0.000000"}
                                </td>
                                <td className="px-4 py-2">
                                  {m.mae?.toFixed(4) || "0.0000"}
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </Panel>

              <PanelResizeHandle className="h-px bg-zinc-800 hover:bg-[hsl(var(--primary)/0.5)] transition-colors cursor-row-resize" />

              <Panel defaultSize={30} minSize={10}>
                <div className="h-full bg-zinc-900/50 flex flex-col overflow-hidden">
                  <div className="h-8 border-b border-zinc-800 flex items-center justify-between px-4 bg-black shrink-0">
                    <div className="flex items-center gap-2">
                      <Terminal size={12} className="text-[hsl(var(--primary))]" />
                      <span className="text-[10px] font-bold uppercase tracking-wider text-white">
                        Process Stream
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <button
                        data-testid="btn-toggle-stream"
                        onClick={() =>
                          setStreamView((prev) => (prev === "tree" ? "log" : "tree"))
                        }
                        className="text-[10px] text-zinc-500 hover:text-white flex items-center gap-1.5 transition-colors"
                      >
                        {streamView === "tree" ? "Logs" : "Tree"}
                      </button>
                      <button
                        data-testid="btn-copy-logs"
                        onClick={handleCopyLogs}
                        className="text-[10px] text-zinc-500 hover:text-white flex items-center gap-1.5 transition-colors"
                      >
                        <Copy size={10} /> {copiedLogs ? "Copied" : "Copy"}
                      </button>
                      <button
                        data-testid="btn-clear-logs"
                        onClick={clearLogs}
                        className="text-[10px] text-zinc-500 hover:text-white flex items-center gap-1.5 transition-colors"
                      >
                        <Trash2 size={10} /> Clear
                      </button>
                    </div>
                  </div>
                  <div
                    className="flex-1 overflow-y-auto p-3 font-mono text-[11px] leading-snug custom-scrollbar bg-black"
                    onMouseDown={handleStreamMouseDown}
                  >
                    {streamView === "tree" ? (
                      archyText ? (
                        <div className="whitespace-pre text-zinc-500">
                          {archyText.split("\n").map((line, idx) =>
                            renderTreeLine(line, idx),
                          )}
                        </div>
                      ) : (
                        <div className="text-[10px] text-zinc-600">
                          No tree data yet.
                        </div>
                      )
                    ) : (
                      logs?.map((log, i: number) => (
                        <div key={i} className="flex gap-3 mb-0.5 group">
                          <span className="text-zinc-800 shrink-0 tabular-nums">
                            [{log.time}]
                            {log.epoch !== null && log.epoch !== undefined
                              ? ` (Epoch ${log.epoch})`
                              : ""}
                          </span>
                          <span
                            className={`break-words flex-1 ${
                              log.type === "success"
                                ? "text-green-500"
                                : log.type === "warn"
                                  ? "text-yellow-500"
                                  : log.type === "info"
                                    ? "text-blue-500"
                                    : log.type === "optuna"
                                      ? "text-purple-500"
                                      : "text-zinc-400"
                            }`}
                          >
                            {log.msg}
                          </span>
                        </div>
                      ))
                    )}
                    <div ref={logEndRef} />
                  </div>
                </div>
              </Panel>
            </PanelGroup>
          </Tabs.Content>

          <Tabs.Content
            value="inference"
            className="h-full overflow-y-auto custom-scrollbar p-10 data-[state=inactive]:hidden"
          >
            <div className="max-w-3xl mx-auto">
              <InferencePlayground loadedPath={loadedModelPath} />
            </div>
          </Tabs.Content>

        </div>
      </Tabs.Root>
    </div>
  );
}

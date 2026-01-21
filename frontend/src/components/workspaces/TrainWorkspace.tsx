import React from "react";
import * as Tabs from "@radix-ui/react-tabs";
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
  } = useTrainingStore();

  const [isMounted, setIsMounted] = React.useState(false);
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
                  className="w-24 h-1 bg-zinc-800 rounded-full appearance-none accent-blue-500 cursor-pointer"
                />
                <span className="text-[10px] font-mono text-blue-500 w-8">
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
                          <div className="w-2 h-2 rounded-full bg-red-500"></div>{" "}
                          Loss
                        </span>
                        <span className="text-white flex items-center gap-1.5">
                          <div className="w-2 h-2 rounded-full bg-blue-500"></div>{" "}
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
                              stroke="#27272a"
                              vertical={false}
                              horizontal={true}
                            />
                            <XAxis dataKey="step" hide />
                            <YAxis
                              yAxisId="left"
                              stroke="#3f3f46"
                              tick={{
                                fill: "#52525b",
                                fontSize: 9,
                                fontWeight: 600,
                              }}
                              tickLine={false}
                              axisLine={false}
                            />
                            <YAxis
                              yAxisId="right"
                              orientation="right"
                              stroke="#3f3f46"
                              tick={{
                                fill: "#52525b",
                                fontSize: 9,
                                fontWeight: 600,
                              }}
                              tickLine={false}
                              axisLine={false}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#09090b",
                                border: "1px solid #27272a",
                                borderRadius: "6px",
                                fontSize: "10px",
                                color: "#fafafa",
                              }}
                              itemStyle={{ padding: "2px 0" }}
                              cursor={{ stroke: "#27272a", strokeWidth: 1 }}
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
                              stroke="#ef4444"
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
                              stroke="#3b82f6"
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

              <PanelResizeHandle className="h-px bg-zinc-800 hover:bg-blue-500/50 transition-colors cursor-row-resize" />

              <Panel defaultSize={30} minSize={10}>
                <div className="h-full bg-zinc-900/50 flex flex-col overflow-hidden">
                  <div className="h-8 border-b border-zinc-800 flex items-center justify-between px-4 bg-black shrink-0">
                    <div className="flex items-center gap-2">
                      <Terminal size={12} className="text-blue-500" />
                      <span className="text-[10px] font-bold uppercase tracking-wider text-white">
                        Process Stream
                      </span>
                    </div>
                    <button
                      data-testid="btn-clear-logs"
                      onClick={clearLogs}
                      className="text-[10px] text-zinc-500 hover:text-white flex items-center gap-1.5 transition-colors"
                    >
                      <Trash2 size={10} /> Clear
                    </button>
                  </div>
                  <div className="flex-1 overflow-y-auto p-3 font-mono text-[11px] leading-snug custom-scrollbar bg-black">
                    {logs?.map((log, i: number) => (
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
                    ))}
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

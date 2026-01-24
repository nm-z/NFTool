"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Panel,
  Group as PanelGroup,
  Separator as PanelResizeHandle,
} from "react-resizable-panels";
import { Settings, AlertCircle } from "lucide-react";
import * as Dialog from "@radix-ui/react-dialog";
import { useTrainingStore } from "@/store/useTrainingStore";
import { useApi } from "@/components/ApiProvider";

// Components
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Sidebar } from "@/components/layout/Sidebar";
import { TrainWorkspace } from "@/components/workspaces/TrainWorkspace";
import { LibraryWorkspace } from "@/components/workspaces/LibraryWorkspace";
import { Inspector } from "@/components/inspector/Inspector";
import { ErrorBoundary } from "@/components/error/ErrorBoundary";

const waitForBackend = async (apiUrl: string, attempts = 5, delayMs = 1000) => {
  for (let i = 0; i < attempts; i += 1) {
    try {
      const res = await fetch(`${apiUrl}/health`);
      if (res.ok) return true;
    } catch {
      // Swallow network errors; we'll retry below.
    }
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
  return false;
};

export default function Dashboard() {
  // Get dynamic API URLs from context (configured for both dev and Tauri)
  const { apiUrl: API_URL, wsUrl: WS_URL_BASE } = useApi();
  const WS_URL = WS_URL_BASE; // WebSockets are handled at root, and getWsUrl already handles this
  const {
    activeWorkspace,
    setActiveWorkspace,
    isRunning,
    setIsRunning,
    isStarting,
    setIsStarting,
    isAborting,
    setIsAborting,
    progress,
    setProgress,
    currentTrial,
    totalTrials,
    setTrialInfo,
    addLog,
    setLogs,
    setResult,
    setMetricsHistory,
    addMetric,
    setGpuList,
    setDatasets,
    setSelectedPredictor,
    setSelectedTarget,
    setRuns,
    setHardwareStats,
    resetTrainingUi,
  } = useTrainingStore();

  const [isMounted, setIsMounted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const startTimeoutRef = useRef<number | null>(null);
  const startRequestTsRef = useRef<number | null>(null);
  const fetchRetryRef = useRef<number | null>(null);
  const [wsStatus, setWsStatus] = useState<number>(3);

  const refreshRuns = React.useCallback(async () => {
    try {
      const runsRes = await fetch(`${API_URL}/training/runs`);
      if (runsRes.ok) {
        const runsJson = await runsRes.json();
        setRuns(runsJson);
      }
    } catch (e) {
      console.debug("Runs refresh error:", e);
    }
  }, [setRuns, API_URL]);

  const getConnectionStatus = () => {
    switch (wsStatus) {
      case 0:
        return { label: "CONNECTING", color: "bg-[hsl(var(--warning))]" };
      case 1:
        return { label: "CONNECTED", color: "bg-[hsl(var(--success))]" };
      case 2:
        return { label: "CLOSING", color: "bg-[hsl(var(--warning))]" };
      case 3:
        return { label: "CLOSED", color: "bg-[hsl(var(--danger))]" };
      default:
        return { label: "DISCONNECTED", color: "bg-[hsl(var(--danger))]" };
    }
  };

  useEffect(() => {
    setIsMounted(true);
  }, []);

  useEffect(() => {
    if (!isMounted) return;
    const fetchData = async () => {
      const backendReady = await waitForBackend(API_URL, 8, 1000);
      if (!backendReady) {
        console.warn("Backend not ready; retrying initial fetch.");
        if (fetchRetryRef.current) {
          clearTimeout(fetchRetryRef.current);
        }
        fetchRetryRef.current = window.setTimeout(fetchData, 2000);
        return;
      }
      try {
        const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
        const dsRes = await fetch(`${API_URL}/data/datasets`, {
          headers: { "X-API-Key": API_KEY },
        });
        if (dsRes.ok) {
          const data = await dsRes.json();
          if (Array.isArray(data)) {
            setDatasets(data);
            if (data.length > 0) {
              // Prefer a predictor/target if named accordingly, otherwise don't force empty selection.
              const pFile =
                data.find((d: { name?: string; path?: string }) =>
                  d.name?.toLowerCase().includes("predictor"),
                ) || data[0];
              const tFile =
                data.find((d: { name?: string; path?: string }) =>
                  d.name?.toLowerCase().includes("target"),
                ) || (data.length > 1 ? data[1] : undefined);
              if (pFile && pFile.path) setSelectedPredictor(pFile.path);
              if (tFile && tFile.path) setSelectedTarget(tFile.path);
            }
          }
        }
      } catch (e) {
        console.warn("Dataset fetch error:", e);
      }

      try {
        const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
        const runsRes = await fetch(`${API_URL}/training/runs`, {
          headers: { "X-API-Key": API_KEY },
        });
        if (runsRes.ok) {
          const runsJson = await runsRes.json();
          setRuns(runsJson);
          // If the backend has persisted logs for the latest run, populate them
          // as the canonical source of truth so the UI doesn't show stale or
          // duplicate entries. This replaces the logs state with the stored
          // run logs on load.
          if (Array.isArray(runsJson) && runsJson.length > 0) {
            const latest = runsJson[0] as unknown;

            type BackendLog = { time: string; msg: string; type?: string; epoch?: number | null };
            const isLog = (v: unknown): v is BackendLog =>
              !!v && typeof v === "object" && "time" in (v as Record<string, unknown>) && "msg" in (v as Record<string, unknown>);
            const isLogArray = (arr: unknown): arr is BackendLog[] =>
              Array.isArray(arr) && arr.every((it) => isLog(it));

            const maybeLogs = (latest as Record<string, unknown>)?.logs;

            // Only apply persisted logs from the backend when the latest run contains
            // server-provided state. This prevents showing previous-run logs before
            // the user starts a new test.
            const latestStatus = (latest as Record<string, unknown>)?.status as string | undefined;
            const hasServerState =
              latestStatus === "running" ||
              latestStatus === "queued";

            if (isLogArray(maybeLogs) && hasServerState) {
              type LogEntryLocal = {
                time: string;
                msg: string;
                type: "default" | "info" | "success" | "warn" | "optuna";
                epoch?: number;
              };
              const allowed = ["default", "info", "success", "warn", "optuna"];
              const normalized: LogEntryLocal[] = maybeLogs.map((l) => ({
                time: String(l.time),
                msg: String(l.msg),
                type: allowed.includes(String(l.type))
                  ? (String(l.type) as LogEntryLocal["type"])
                  : "default",
                epoch: typeof l.epoch === "number" ? (l.epoch as number) : undefined,
              }));
              setLogs(normalized);
            }
          }
        }
      } catch (e) {
        console.warn("Runs fetch error:", e);
      }

      try {
        const gpuRes = await fetch(`${API_URL}/hardware/gpus`);
        if (gpuRes.ok) setGpuList(await gpuRes.json());
      } catch (e) {
        console.warn("GPU fetch error:", e);
      }
    };
    fetchData();
    return () => {
      if (fetchRetryRef.current) {
        clearTimeout(fetchRetryRef.current);
        fetchRetryRef.current = null;
      }
    };
  }, [isMounted, setDatasets, setRuns, setGpuList, setSelectedPredictor, setSelectedTarget, setLogs, API_URL]);

  useEffect(() => {
    if (!isMounted) return;
    let active = true;
    let ws: WebSocket | null = null;

    const connect = async () => {
      if (!active) return;
      const backendReady = await waitForBackend(API_URL, 3, 1000);
      if (!backendReady) {
        if (active) {
          setWsStatus(3);
          setTimeout(connect, 2000);
        }
        return;
      }
      try {
        // No API key subprotocol needed for Tauri (local auth removed)
        ws = new WebSocket(`${WS_URL}/ws`);
        socketRef.current = ws;
        setWsStatus(ws.readyState);

        ws.onopen = () => {
          if (active) {
            console.debug("WebSocket connected to:", WS_URL, {
              ts: Date.now(),
            });
            setWsStatus(1);
          }
        };

        ws.onmessage = (event) => {
          if (!active) return;
          try {
            console.debug("WS raw message:", event.data);
            const msg = JSON.parse(event.data);
            console.debug("WS parsed message:", {
              type: msg.type,
              data: msg.data,
            });
            if (msg.type === "init") {
              setIsRunning(msg.data.is_running);
              if (msg.data.is_running) setIsStarting(false);
              setIsAborting(msg.data.is_aborting || false);
              setProgress(msg.data.progress);
              setTrialInfo(msg.data.current_trial, msg.data.total_trials);
              // Only apply server-provided result/logs/metrics when:
              // - the engine is currently running, OR
              // - a result/metrics are explicitly present, OR
              // - a model or dataset has been intentionally selected (deploy behavior).
              // This prevents showing leftover logs/metrics on fresh UI open.
              const hasServerState = Boolean(
                msg.data.is_running || msg.data.result,
              );
              if (hasServerState) {
                setResult(msg.data.result);
                if (msg.data.metrics_history)
                  setMetricsHistory(msg.data.metrics_history);
                if (msg.data.hardware_stats)
                  setHardwareStats(msg.data.hardware_stats);
                if (msg.data.logs) setLogs(msg.data.logs);
              }
            } else if (msg.type === "status") {
              // Backends may emit either boolean flags (is_running) or a status
              // string (e.g. "queued", "running"). Handle both formats gracefully.
              const statusStr =
                typeof msg.data?.status === "string" ? String(msg.data.status) : undefined;
              const isTerminalStatus =
                statusStr === "completed" || statusStr === "failed" || statusStr === "aborted";
              const isStopped = msg.data?.is_running === false;
              const hasResult = Boolean(msg.data?.result);
              const progressVal =
                typeof msg.data?.progress === "number" ? msg.data.progress : null;
              if (statusStr) {
                if (statusStr === "running") {
                  setIsRunning(true);
                  setIsStarting(false);
                } else if (statusStr === "queued" || statusStr === "pending") {
                  setIsRunning(false);
                  setIsStarting(true);
                } else if (statusStr === "completed" || statusStr === "failed" || statusStr === "aborted") {
                  // Training finished - reset running state but preserve results
                  setIsRunning(false);
                  setIsStarting(false);
                  setIsAborting(false);
                  setTrialInfo(0, 0);
                } else {
                  setIsRunning(false);
                  setIsStarting(false);
                }
              } else {
                setIsRunning(Boolean(msg.data.is_running));
                if (msg.data.is_running) setIsStarting(false);
              }

              setIsAborting(msg.data.is_aborting || false);
              setProgress(msg.data.progress ?? 0);
              // Accept both snake_case and camelCase trial keys
              setTrialInfo(msg.data.current_trial ?? msg.data.currentTrial ?? 0, msg.data.total_trials ?? msg.data.totalTrials ?? 0);
              if (msg.data.result) setResult(msg.data.result);

              if (isTerminalStatus || (isStopped && (hasResult || progressVal === 100))) {
                refreshRuns();
              }
            } else if (msg.type === "log") {
              addLog(msg.data);
            } else if (msg.type === "metrics") {
              addMetric(msg.data);
            } else if (msg.type === "hardware") {
              setHardwareStats(msg.data);
            }
          } catch (e) {
            console.error("WS message parse error:", e);
          }
        };

        ws.onclose = () => {
          if (active) {
            console.debug("WebSocket closed, retrying...", { ts: Date.now() });
            setWsStatus(3);
            setTimeout(connect, 5000);
          }
        };

        ws.onerror = (e) => {
          if (active) {
            console.error("WebSocket error:", e);
            setWsStatus(3);
          }
        };
      } catch (e) {
        console.error("WS connection instantiation error:", e);
      }
    };
    connect();
    return () => {
      active = false;
      if (ws) ws.close();
    };
  }, [
    isMounted,
    setIsRunning,
    setProgress,
    setTrialInfo,
    setResult,
    setMetricsHistory,
    setHardwareStats,
    setLogs,
    addLog,
    addMetric,
    setIsAborting,
    setIsStarting,
    refreshRuns,
    API_URL,
    WS_URL,
  ]);

  // Polling fallback: when WebSocket is disconnected or while starting, poll runs
  useEffect(() => {
    if (!isMounted) return;
    let intervalId: number | null = null;
    const shouldPoll = wsStatus === 3 || isStarting;
    const poll = async () => {
      try {
        const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
        const runsRes = await fetch(`${API_URL}/training/runs`, {
          headers: { "X-API-Key": API_KEY },
        });
        if (runsRes.ok) {
          const runsJson = await runsRes.json();
          setRuns(runsJson);
          const active = Array.isArray(runsJson)
            ? runsJson.find(
                (r: Record<string, unknown>) =>
                  (r["status"] as string | undefined) === "running",
              )
            : undefined;
          const queued = Array.isArray(runsJson)
            ? runsJson.find(
                (r: Record<string, unknown>) =>
                  (r["status"] as string | undefined) === "queued",
              )
            : undefined;
          const completed = Array.isArray(runsJson)
            ? runsJson.find(
                (r: Record<string, unknown>) => {
                  const status = r["status"] as string | undefined;
                  return status === "completed" || status === "failed" || status === "aborted";
                }
              )
            : undefined;
          if (active) {
            setIsRunning(true);
            setIsStarting(false);
          } else if (queued) {
            setIsRunning(false);
            setIsStarting(true);
          } else if (completed) {
            // Training finished - reset running state but preserve results
            setIsRunning(false);
            setIsStarting(false);
            setIsAborting(false);
            setTrialInfo(0, 0);
          } else {
            setIsRunning(false);
            setIsStarting(false);
          }
        }
      } catch {
        // ignore transient errors
      }
    };

    if (shouldPoll) {
      // initial immediate poll
      poll();
      intervalId = window.setInterval(poll, 2000);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    };
  }, [isMounted, wsStatus, isStarting, setRuns, setIsRunning, setIsStarting, setIsAborting, setTrialInfo, API_URL]);

  const parseRange = (val: string): [number, number] => {
    const parts = val.split("→").map((part) => part.trim());
    const min = parseFloat(parts[0]) || 0;
    const max = parseFloat(parts[parts.length - 1]) || min;
    return [min, max];
  };

  // Clear the start timeout if the engine reports it is running.
  useEffect(() => {
    if (isRunning && startTimeoutRef.current) {
      clearTimeout(startTimeoutRef.current);
      startTimeoutRef.current = null;
    }
  }, [isRunning]);

  // Ensure timeout is cleared on unmount.
  useEffect(() => {
    return () => {
      if (startTimeoutRef.current) {
        clearTimeout(startTimeoutRef.current);
        startTimeoutRef.current = null;
      }
    };
  }, []);

  const handleStartTraining = async () => {
    const backendReady = await waitForBackend(API_URL);
    if (!backendReady) {
      const msg = "Backend is not ready. Please wait and try again.";
      console.warn(msg);
      setError(msg);
      addLog({
        time: new Date().toLocaleTimeString(),
        msg: `Execution Error: ${msg}`,
        type: "warn",
      });
      setIsStarting(false);
      return;
    }
    const s = useTrainingStore.getState();

    const [lrMin, lrMax] = parseRange(s.lrRange);
    const [layersMin, layersMax] = parseRange(s.layersRange);
    const [lSizeMin, lSizeMax] = parseRange(s.layerSizeRange);
    const [dropMin, dropMax] = parseRange(s.dropoutRange);
    const [hDimMin, hDimMax] = parseRange(s.hDimRange);
    const [convMin, convMax] = parseRange(s.convBlocks);
    const [cnnFilterMin, cnnFilterMax] = parseRange(s.cnnFilterCapRange);

    const rawModelChoice = s.modelType && s.modelType.trim() ? s.modelType.trim() : "NN";
    const modelChoice =
      rawModelChoice === "NN" || rawModelChoice === "CNN" ? rawModelChoice : "NN";
    const config = {
      model_choice: modelChoice,
      seed: parseInt(s.seed) || 42,
      patience: parseInt(s.patience) || 100,
      train_ratio: s.split / 100,
      val_ratio: (100 - s.split) / 200,
      test_ratio: (100 - s.split) / 200,
      optuna_trials: parseInt(s.trials) || 10,
      optimizers: ["AdamW"],
      n_layers_min: Math.round(layersMin),
      n_layers_max: Math.round(layersMax),
      l_size_min: Math.round(lSizeMin),
      l_size_max: Math.round(lSizeMax),
      lr_min: lrMin,
      lr_max: lrMax,
      drop_min: dropMin,
      drop_max: dropMax,
      h_dim_min: hDimMin,
      h_dim_max: hDimMax,
      conv_blocks_min: Math.round(convMin),
      conv_blocks_max: Math.round(convMax),
      kernel_size: parseInt(s.kernelSize) || 3,
      gpu_throttle_sleep: parseFloat(s.gpuThrottle) || 0.1,
      cnn_filter_cap_min: Math.round(cnnFilterMin),
      cnn_filter_cap_max: Math.round(cnnFilterMax),
      max_epochs: parseInt(s.maxEpochs) || 200,
      batch_size: parseInt(s.batchSize) || 32,
      device: s.deviceChoice,
      gpu_id: s.gpuChoice,
      predictor_path: s.selectedPredictor,
      target_path: s.selectedTarget,
    };
    // Debug: ensure selections are visible in console
    console.debug("Starting training with selectedPredictor:", s.selectedPredictor, "selectedTarget:", s.selectedTarget, "datasets:", s.datasets?.length);

    // If the UI didn't set predictor/target but datasets exist, auto-fill sensible defaults.
    if ((!s.selectedPredictor || !s.selectedPredictor.trim()) && Array.isArray(s.datasets) && s.datasets.length > 0) {
      const p = s.datasets.find((d: { name?: string; path?: string }) => d.name?.toLowerCase().includes("predictor")) || s.datasets[0];
      if ((p as { path?: string }).path) s.setSelectedPredictor((p as { path?: string }).path as string);
    }
    if ((!s.selectedTarget || !s.selectedTarget.trim()) && Array.isArray(s.datasets) && s.datasets.length > 1) {
      const t =
        s.datasets.find((d: { name?: string; path?: string }) => d.name?.toLowerCase().includes("target")) ||
        (s.datasets.length > 1 ? s.datasets[1] : s.datasets[0]);
      if ((t as { path?: string }).path) s.setSelectedTarget((t as { path?: string }).path as string);
    }

    // Validate final selections (re-read from store since we may have autofilled above)
    const finalState = useTrainingStore.getState();
    const finalPredictor = finalState.selectedPredictor?.trim();
    const finalTarget = finalState.selectedTarget?.trim();
    if (!finalPredictor || !finalTarget) {
      const msg = "Please select both predictor and target dataset files before starting training.";
      console.error(msg, { finalPredictor, finalTarget });
      setError(msg);
      addLog({
        time: new Date().toLocaleTimeString(),
        msg: `Execution Error: ${msg}`,
        type: "warn",
      });
      setIsStarting(false);
      return;
    }
    // Ensure each run starts from a clean UI baseline.
    setLogs([]);
    setMetricsHistory([]);
    setResult(null);
    setProgress(0);
    setTrialInfo(0, config.optuna_trials);
    setIsStarting(true);
    console.debug("handleStartTraining: initiating training request", {
      config,
      apiUrl: API_URL,
      finalUrl: `${API_URL}/training/train`,
    });
    startRequestTsRef.current = Date.now();
    // Start a 10s watchdog: if the engine hasn't reported `is_running` within
    // 10 do not make this no 30s, it should not take even 5s.
    if (startTimeoutRef.current) {
      clearTimeout(startTimeoutRef.current);
      startTimeoutRef.current = null;
    }
    startTimeoutRef.current = window.setTimeout(() => {
      const runningNow = useTrainingStore.getState().isRunning;
      if (!runningNow) {
        const msg = "Training failed to start within 10 seconds.";
        console.error(msg);
        setError(msg);
        addLog({
          time: new Date().toLocaleTimeString(),
          msg: `Execution Error: ${msg}`,
          type: "warn",
        });
        setIsStarting(false);
      }
      startTimeoutRef.current = null;
    }, 10000);
    try {
      const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
      const res = await fetch(`${API_URL}/training/train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": API_KEY,
        },
        body: JSON.stringify(config),
      });

      console.debug("handleStartTraining: POST /train returned", {
        status: res.status,
        ok: res.ok,
        elapsedMs:
          startRequestTsRef.current != null
            ? Date.now() - startRequestTsRef.current
            : null,
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText || `Server responded with ${res.status}`);
      }

      const data = await res.json();
      // On successful submission, clear the short watchdog so we don't show a
      // false-positive "failed to start" while the backend queues/starts the run.
      if (startTimeoutRef.current) {
        clearTimeout(startTimeoutRef.current);
        startTimeoutRef.current = null;
      }
      console.debug("handleStartTraining: train response json", data);
      addLog({
        time: new Date().toLocaleTimeString(),
        msg: `Training pipeline initiated: ${data.run_id}`,
        type: "info",
      });

      // Immediately poll the runs endpoint once to update UI state so the
      // dashboard doesn't stay stuck on "Starting..." if the WebSocket is
      // briefly disconnected. We'll still keep the longer watchdog below.
      const runId = data.run_id;
      (async function immediateRunSync(id: string) {
        try {
          const runsRes = await fetch(`${API_URL}/training/runs`, {
            headers: { "X-API-Key": API_KEY },
          });
          if (runsRes.ok) {
            const runsJson = await runsRes.json();
            setRuns(runsJson);
            const found =
              Array.isArray(runsJson) &&
              runsJson.find((r: Record<string, unknown>) => r["run_id"] === id);
            if (found) {
              const foundStatus = (found as Record<string, unknown>)["status"] as
                | string
                | undefined;
              if (foundStatus === "running") {
                setIsRunning(true);
                setIsStarting(false);
              } else if (foundStatus === "queued") {
                // remain in starting state but ensure UI has the latest run list
                setIsRunning(false);
                setIsStarting(true);
              }
            }
          }
        } catch (err) {
          console.debug("Immediate run sync failed:", err);
        }
      })(runId).catch(() => {});

      // Begin a longer watchdog and poll the runs endpoint for this run id so
      // we don't show a false "failed to start" when the backend is queuing.
      const watchdogMs = 30000;
      const pollIntervalMs = 1000;

      // Start a 30s watchdog; it will be cleared if the run is observed.
      if (startTimeoutRef.current) {
        clearTimeout(startTimeoutRef.current);
        startTimeoutRef.current = null;
      }
      startTimeoutRef.current = window.setTimeout(() => {
        const runningNow = useTrainingStore.getState().isRunning;
        if (!runningNow) {
          const msg = "Training failed to start within 30 seconds.";
          console.error(msg);
          setError(msg);
          addLog({
            time: new Date().toLocaleTimeString(),
            msg: `Execution Error: ${msg}`,
            type: "warn",
          });
          setIsStarting(false);
        }
        startTimeoutRef.current = null;
      }, watchdogMs);

      (async function pollRunUntilRunning(id: string) {
        const deadline = Date.now() + watchdogMs;
        while (Date.now() < deadline) {
          try {
            const runsRes = await fetch(`${API_URL}/training/runs`);
            if (runsRes.ok) {
              const runs = await runsRes.json();
              const found =
                Array.isArray(runs) &&
                runs.find(
                  (r: Record<string, unknown>) =>
                    (r["run_id"] as string | undefined) === id,
                );
              if (found) {
                // If backend has acknowledged the run (queued/running), clear watchdog.
                const foundStatus = (found as Record<string, unknown>)["status"] as
                  | string
                  | undefined;
                if (foundStatus === "queued" || foundStatus === "running") {
                  if (startTimeoutRef.current) {
                    clearTimeout(startTimeoutRef.current);
                    startTimeoutRef.current = null;
                  }
                  return;
                }
              }
            }
          } catch {
            // ignore transient poll errors
          }
          await new Promise((res) => setTimeout(res, pollIntervalMs));
        }
      })(runId).catch(() => {});
    } catch (e: unknown) {
      console.error("Training error:", e);
      const msg = e instanceof Error ? e.message : String(e);
      setError(`Execution Failed: ${msg}`);
      addLog({
        time: new Date().toLocaleTimeString(),
        msg: `Execution Error: ${msg}`,
        type: "warn",
      });
      setIsStarting(false);
      if (startTimeoutRef.current) {
        clearTimeout(startTimeoutRef.current);
        startTimeoutRef.current = null;
      }
    }
  };

  const handleAbortTraining = async () => {
    try {
      const backendReady = await waitForBackend(API_URL);
      if (!backendReady) {
        const msg = "Backend is not ready. Please wait and try again.";
        console.warn(msg);
        setError(msg);
        return;
      }
      const res = await fetch(`${API_URL}/training/abort`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(await res.text());
      addLog({
        time: new Date().toLocaleTimeString(),
        msg: "Termination signal dispatched to core engine",
        type: "warn",
      });
      try {
        const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
        const runsRes = await fetch(`${API_URL}/training/runs`, {
          headers: { "X-API-Key": API_KEY },
        });
        if (runsRes.ok) {
          const runsJson = await runsRes.json();
          setRuns(runsJson);
        }
      } catch {
        // ignore refresh errors
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(`Abort Failed: ${msg}`);
    }
  };

  const handleResetTraining = () => {
    resetTrainingUi();
    setError(null);
    startRequestTsRef.current = null;
    if (startTimeoutRef.current) {
      clearTimeout(startTimeoutRef.current);
      startTimeoutRef.current = null;
    }
  };

  if (!isMounted) return <div className="h-screen bg-black" />;

  const connStatus = getConnectionStatus();

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-white selection:bg-[hsl(var(--primary)/0.2)] font-sans overflow-hidden">
      <Header
        isRunning={isRunning}
        isStarting={isStarting}
        isAborting={isAborting}
        currentTrial={currentTrial}
        totalTrials={totalTrials}
        handleStartTraining={handleStartTraining}
        handleAbortTraining={handleAbortTraining}
        handleResetTraining={handleResetTraining}
        setActiveWorkspace={setActiveWorkspace}
      />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          activeWorkspace={activeWorkspace}
          setActiveWorkspace={setActiveWorkspace}
          onOpenSettings={() => setIsSettingsOpen(true)}
        />

        <div className="flex-1 flex flex-col min-w-0">
          <PanelGroup orientation="horizontal">
            <Panel defaultSize={75} minSize={40}>
              <div className="flex flex-col h-full overflow-hidden bg-zinc-950">
                {activeWorkspace === "Train" && (
                  <ErrorBoundary>
                    <TrainWorkspace />
                  </ErrorBoundary>
                )}
                {activeWorkspace === "Library" && (
                  <ErrorBoundary>
                    <LibraryWorkspace />
                  </ErrorBoundary>
                )}
              </div>
            </Panel>

            {activeWorkspace === "Train" && (
              <>
                <PanelResizeHandle className="w-px bg-zinc-800 hover:bg-[hsl(var(--primary)/0.5)] transition-colors" />
                <Panel defaultSize={25} minSize={20}>
                  <ErrorBoundary>
                    <Inspector setError={setError} />
                  </ErrorBoundary>
                </Panel>
              </>
            )}
          </PanelGroup>
        </div>
      </div>

      <Footer
        progress={progress}
        wsStatusLabel={connStatus.label}
        wsStatusColor={connStatus.color}
      />

      {error && (
        <div className="fixed bottom-12 right-6 z-[100] max-w-md animate-in fade-in slide-in-from-right-4 pointer-events-none">
          <div className="bg-[hsl(var(--danger)/0.1)] border border-[hsl(var(--danger)/0.2)] rounded-lg p-4 flex gap-3 shadow-2xl backdrop-blur-md">
            <AlertCircle className="text-[hsl(var(--danger))] shrink-0" size={18} />
            <div className="flex-1">
              <h4 className="text-[11px] font-bold text-[hsl(var(--danger))] uppercase tracking-widest mb-1">
                System Error
              </h4>
              <p className="text-[11px] text-[hsl(var(--danger)/0.8)] font-mono leading-relaxed">
                {error}
              </p>
              <button
                onClick={() => setError(null)}
                className="mt-3 text-[10px] font-bold text-[hsl(var(--danger))] hover:text-[hsl(var(--danger)/0.85)] uppercase pointer-events-auto"
                data-testid="btn-dismiss-error"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}

      <Dialog.Root open={isSettingsOpen} onOpenChange={setIsSettingsOpen}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[100]" />
          <Dialog.Content className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 bg-zinc-900 border border-zinc-800 rounded-xl p-6 shadow-2xl z-[101]">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 rounded bg-[hsl(var(--primary)/0.1)] flex items-center justify-center text-[hsl(var(--primary))]">
                <Settings size={18} />
              </div>
              <Dialog.Title className="text-sm font-bold uppercase tracking-widest">
                Global Settings
              </Dialog.Title>
            </div>
            <Dialog.Description className="sr-only">
              System configuration and preference persistence settings.
            </Dialog.Description>
            <div className="space-y-4 py-4 border-y border-zinc-800/50">
              <p className="text-[11px] text-zinc-400 font-mono leading-relaxed">
                System configuration module is currently under development.
                Preference persistence and API endpoint configuration will be
                available in V3.1.
              </p>
            </div>
            <div className="mt-6 flex justify-end">
              <button
                type="button"
                data-testid="settings-close"
                onClick={(e) => {
                  e.stopPropagation();
                  setIsSettingsOpen(false);
                }}
                className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white text-[11px] font-bold rounded-lg transition-all"
              >
                CLOSE
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}

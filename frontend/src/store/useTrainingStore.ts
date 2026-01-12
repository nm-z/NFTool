import { create } from "zustand";

interface LogEntry {
  time: string;
  msg: string;
  type: "default" | "info" | "success" | "warn" | "optuna";
  epoch?: number; // Add epoch as an optional field
}

interface TrainingResult {
  best_r2: number;
  best_params: Record<string, unknown>;
}

interface MetricPoint {
  trial: number;
  loss: number;
  r2: number;
  mae: number;
  val_loss: number;
}

interface HardwareStats {
  vram_total_gb: number;
  vram_used_gb: number;
  vram_percent: number;
  gpu_use_percent: number;
  gpu_temp_c: number;
  cpu_percent: number;
  ram_total_gb: number;
  ram_used_gb: number;
  ram_percent: number;
}

export type WorkspaceType = "Train" | "Inference" | "Runs" | "Library";

interface TrainingState {
  // Global App State
  activeWorkspace: WorkspaceType;
  isAdvancedMode: boolean;

  // Training Engine State
  isRunning: boolean;
  isStarting: boolean;
  isAborting: boolean;
  progress: number;
  currentTrial: number;
  totalTrials: number;
  logs: LogEntry[];
  result: TrainingResult | null;
  metricsHistory: MetricPoint[];

  // Hardware/System State
  hardwareStats: HardwareStats | null;

  // Model/Asset State
  deviceChoice: "cuda" | "cpu";
  gpuChoice: number;
  gpuList: { id: number; name: string }[];
  loadedModelPath: string | null;
  activePreset: string;
  hasUnsavedChanges: boolean;
  selectedPredictor: string;
  selectedTarget: string;
  datasets: { name: string; path: string }[];
  runs: Record<string, unknown>[];

  // Training Configuration (Inspector States)
  seed: string;
  patience: string;
  trials: string;
  lrRange: string;
  split: number;
  modelType: string;
  convBlocks: string;
  kernelSize: string;
  layersRange: string;
  layerSizeRange: string;
  dropoutRange: string;
  hDimRange: string;
  maxEpochs: string;
  gpuThrottle: string;
  cnnFilterCapRange: string;

  // Actions
  setActiveWorkspace: (ws: WorkspaceType) => void;
  setIsAdvancedMode: (mode: boolean) => void;
  setIsRunning: (isRunning: boolean) => void;
  setIsStarting: (isStarting: boolean) => void;
  setIsAborting: (isAborting: boolean) => void;
  setProgress: (progress: number) => void;
  setTrialInfo: (current: number, total: number) => void;
  addLog: (log: LogEntry) => void;
  setLogs: (logs: LogEntry[]) => void;
  setResult: (result: TrainingResult | null) => void;
  clearLogs: () => void;
  addMetric: (metric: MetricPoint) => void;
  setMetricsHistory: (metrics: MetricPoint[]) => void;
  setHardwareStats: (stats: HardwareStats) => void;
  setDeviceChoice: (device: "cuda" | "cpu") => void;
  setGpuChoice: (id: number) => void;
  setGpuList: (gpus: { id: number; name: string }[]) => void;
  setLoadedModelPath: (path: string | null) => void;
  setActivePreset: (preset: string) => void;
  setHasUnsavedChanges: (hasUnsavedChanges: boolean) => void;
  setSelectedPredictor: (path: string) => void;
  setSelectedTarget: (path: string) => void;
  setDatasets: (datasets: { name: string; path: string }[]) => void;
  setRuns: (runs: Record<string, unknown>[]) => void;

  // Config Setters
  setSeed: (v: string) => void;
  setPatience: (v: string) => void;
  setTrials: (v: string) => void;
  setLrRange: (v: string) => void;
  setSplit: (v: number) => void;
  setModelType: (v: string) => void;
  setConvBlocks: (v: string) => void;
  setKernelSize: (v: string) => void;
  setLayersRange: (v: string) => void;
  setLayerSizeRange: (v: string) => void;
  setDropoutRange: (v: string) => void;
  setHDimRange: (v: string) => void;
  setMaxEpochs: (v: string) => void;
  setGpuThrottle: (v: string) => void;
  setCnnFilterCapRange: (v: string) => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  activeWorkspace: "Train",
  isAdvancedMode: true,
  isRunning: false,
  isStarting: false,
  isAborting: false,
  progress: 0,
  currentTrial: 0,
  totalTrials: 0,
  logs: [],
  result: null,
  metricsHistory: [],
  hardwareStats: null,
  deviceChoice: "cuda",
  gpuChoice: 0,
  gpuList: [],
  loadedModelPath: null,
  activePreset: "Gold Standard MLP",
  hasUnsavedChanges: false,
  selectedPredictor: "",
  selectedTarget: "",
  datasets: [],
  runs: [],

  // Default Config
  seed: "42",
  patience: "100",
  trials: "10",
  lrRange: "1e-4 → 1e-3",
  split: 70,
  modelType: "NN",
  convBlocks: "1 → 3",
  kernelSize: "3",
  layersRange: "1 → 8",
  layerSizeRange: "128 → 1024",
  dropoutRange: "0.0 → 0.0",
  hDimRange: "32 → 256",
  maxEpochs: "200",
  gpuThrottle: "0.1",
  cnnFilterCapRange: "512 → 1024",

  setActiveWorkspace: (activeWorkspace) => set({ activeWorkspace }),
  setIsAdvancedMode: (isAdvancedMode) => set({ isAdvancedMode }),
  setIsRunning: (isRunning) => set({ isRunning }),
  setIsStarting: (isStarting) => set({ isStarting }),
  setIsAborting: (isAborting) => set({ isAborting }),
  setProgress: (progress) => set({ progress: progress ?? 0 }),
  setTrialInfo: (current, total) =>
    set({
      currentTrial: current ?? 0,
      totalTrials: total ?? 0,
    }),
  addLog: (log) => set((state) => ({ logs: [...state.logs, log] })),
  setLogs: (logs) => set({ logs }),
  setResult: (result) => set({ result }),
  clearLogs: () => set({ logs: [] }),
  addMetric: (metric) =>
    set((state) => ({ metricsHistory: [...state.metricsHistory, metric] })),
  setMetricsHistory: (metrics) => set({ metricsHistory: metrics }),
  setHardwareStats: (stats) => set({ hardwareStats: stats }),
  setDeviceChoice: (deviceChoice) => set({ deviceChoice }),
  setGpuChoice: (gpuChoice) => set({ gpuChoice }),
  setGpuList: (gpuList) => set({ gpuList }),
  setLoadedModelPath: (path) => set({ loadedModelPath: path }),
  setActivePreset: (activePreset) => set({ activePreset }),
  setHasUnsavedChanges: (hasUnsavedChanges) => set({ hasUnsavedChanges }),
  setSelectedPredictor: (selectedPredictor) => set({ selectedPredictor }),
  setSelectedTarget: (selectedTarget) => set({ selectedTarget }),
  setDatasets: (datasets) => set({ datasets }),
  setRuns: (runs) => set({ runs }),

  // Config Setters
  setSeed: (seed) => set({ seed }),
  setPatience: (patience) => set({ patience }),
  setTrials: (trials) => set({ trials }),
  setLrRange: (lrRange) => set({ lrRange }),
  setSplit: (split) => set({ split }),
  setModelType: (modelType) => set({ modelType }),
  setConvBlocks: (convBlocks) => set({ convBlocks }),
  setKernelSize: (kernelSize) => set({ kernelSize }),
  setLayersRange: (layersRange) => set({ layersRange }),
  setLayerSizeRange: (layerSizeRange) => set({ layerSizeRange }),
  setDropoutRange: (dropoutRange) => set({ dropoutRange }),
  setHDimRange: (hDimRange) => set({ hDimRange }),
  setMaxEpochs: (maxEpochs) => set({ maxEpochs }),
  setGpuThrottle: (gpuThrottle) => set({ gpuThrottle }),
  setCnnFilterCapRange: (cnnFilterCapRange) => set({ cnnFilterCapRange }),
}));

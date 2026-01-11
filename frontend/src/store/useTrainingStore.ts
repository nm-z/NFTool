import { create } from 'zustand';

interface LogEntry {
  time: string;
  msg: string;
  type: 'default' | 'info' | 'success' | 'warn' | 'optuna';
}

interface TrainingResult {
  best_r2: number;
  best_params: Record<string, any>;
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
  gpuList: {id: number, name: string}[];
  loadedModelPath: string | null;
  activePreset: string;
  hasUnsavedChanges: boolean;
  
  // Actions
  setActiveWorkspace: (ws: WorkspaceType) => void;
  setIsAdvancedMode: (mode: boolean) => void;
  setIsRunning: (isRunning: boolean) => void;
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
  setGpuList: (gpus: {id: number, name: string}[]) => void;
  setLoadedModelPath: (path: string | null) => void;
  setActivePreset: (preset: string) => void;
  setHasUnsavedChanges: (unsaved: boolean) => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  activeWorkspace: "Train",
  isAdvancedMode: false,
  isRunning: false,
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

  setActiveWorkspace: (activeWorkspace) => set({ activeWorkspace }),
  setIsAdvancedMode: (isAdvancedMode) => set({ isAdvancedMode }),
  setIsRunning: (isRunning) => set({ isRunning }),
  setIsAborting: (isAborting) => set({ isAborting }),
  setProgress: (progress) => set({ progress }),
  setTrialInfo: (current, total) => set({ currentTrial: current, totalTrials: total }),
  addLog: (log) => set((state) => ({ logs: [...state.logs, log] })),
  setLogs: (logs) => set({ logs }),
  setResult: (result) => set({ result }),
  clearLogs: () => set({ logs: [] }),
  addMetric: (metric) => set((state) => ({ metricsHistory: [...state.metricsHistory, metric] })),
  setMetricsHistory: (metrics) => set({ metricsHistory: metrics }),
  setHardwareStats: (stats) => set({ hardwareStats: stats }),
  setDeviceChoice: (deviceChoice) => set({ deviceChoice }),
  setGpuChoice: (gpuChoice) => set({ gpuChoice }),
  setGpuList: (gpuList) => set({ gpuList }),
  setLoadedModelPath: (path) => set({ loadedModelPath: path }),
  setActivePreset: (activePreset) => set({ activePreset }),
  setHasUnsavedChanges: (hasUnsavedChanges) => set({ hasUnsavedChanges }),
}));

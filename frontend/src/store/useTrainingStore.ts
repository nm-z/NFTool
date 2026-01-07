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

interface TrainingState {
  isRunning: boolean;
  progress: number;
  logs: LogEntry[];
  result: TrainingResult | null;
  
  // Actions
  setIsRunning: (isRunning: boolean) => void;
  setProgress: (progress: number) => void;
  addLog: (log: LogEntry) => void;
  setLogs: (logs: LogEntry[]) => void;
  setResult: (result: TrainingResult | null) => void;
  clearLogs: () => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  isRunning: false,
  progress: 0,
  logs: [],
  result: null,

  setIsRunning: (isRunning) => set({ isRunning }),
  setProgress: (progress) => set({ progress }),
  addLog: (log) => set((state) => ({ logs: [...state.logs, log] })),
  setLogs: (logs) => set({ logs }),
  setResult: (result) => set({ result }),
  clearLogs: () => set({ logs: [] }),
}));


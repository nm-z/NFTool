import React from "react";
import * as Tabs from "@radix-ui/react-tabs";
import * as Accordion from "@radix-ui/react-accordion";
import { Info, RefreshCw, AlertCircle } from "lucide-react";

import { InspectorSection, ControlInput, RangeControl } from "../common/Inputs";
import { TabTrigger } from "../common/UIComponents";
import { HardwarePanel } from "../common/Cards";

const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

export function Inspector({ 
  activeWorkspace, modelType, setModelType, seed, setSeed, patience, setPatience,
  trials, setTrials, lrRange, setLrRange, convBlocks, setConvBlocks, kernelSize, setKernelSize,
  layersRange, setLayersRange, layerSizeRange, setLayerSizeRange, dropoutRange, setDropoutRange,
  hDimRange, setHDimRange, maxEpochs, setMaxEpochs, gpuThrottle, setGpuThrottle, cnnFilterCapRange, setCnnFilterCapRange,
  datasets, selectedPredictor, setSelectedPredictor, selectedTarget, setSelectedTarget,
  isAdvancedMode, deviceChoice, setDeviceChoice, gpuChoice, setGpuChoice, gpuList, setGpuList,
  hardwareStats, setError
}: any) {
  return (
    <aside className="h-full flex flex-col bg-[hsl(var(--panel))] border-l border-[hsl(var(--border))] outline-none" tabIndex={-1}>
      <Tabs.Root defaultValue="model" className="flex flex-col flex-1 overflow-hidden">
        <div className="px-4 pt-4 shrink-0 bg-[hsl(var(--panel))]">
          <Tabs.List className="flex border-b border-[hsl(var(--border))] gap-6">
            <TabTrigger value="model" label="Model" />
            <TabTrigger value="perf" label="Performance" />
          </Tabs.List>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
          <Tabs.Content value="model" className="space-y-1">
            <Accordion.Root type="multiple" defaultValue={["dataset", "arch", "optuna"]} className="w-full">
              <InspectorSection value="dataset" title="Dataset Assets">
                <div className="space-y-4 pt-2">
                  <div className="space-y-1.5">
                    <label className="text-[9px] font-bold text-[#52525b] uppercase">Predictors (X)</label>
                    <select 
                      value={selectedPredictor} 
                      onChange={(e) => setSelectedPredictor(e.target.value)}
                      className="w-full bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded px-2.5 py-1.5 text-[11px] text-[hsl(var(--foreground-active))] focus:outline-none focus:border-[#3b82f6] transition-colors"
                    >
                      {datasets.map((d: any) => <option key={d.path} value={d.path}>{d.name}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-[9px] font-bold text-[#52525b] uppercase">Targets (y)</label>
                    <select 
                      value={selectedTarget} 
                      onChange={(e) => setSelectedTarget(e.target.value)}
                      className="w-full bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded px-2.5 py-1.5 text-[11px] text-[hsl(var(--foreground-active))] focus:outline-none focus:border-[#3b82f6] transition-colors"
                    >
                      {datasets.map((d: any) => <option key={d.path} value={d.path}>{d.name}</option>)}
                    </select>
                  </div>
                </div>
              </InspectorSection>

              <InspectorSection value="arch" title="Architecture">
                <div className="space-y-4 pt-2">
                  <div className="space-y-1.5 group relative">
                    <label className="text-[9px] font-bold text-[#52525b] uppercase flex items-center gap-1.5 cursor-help">
                      Base Class
                      <Info size={10} className="text-[#3f3f46] group-hover:text-[#3b82f6] transition-colors" />
                    </label>
                    <div className="grid grid-cols-2 bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded p-0.5">
                      <button onClick={() => setModelType("NN")} className={`py-1 rounded text-[10px] font-bold transition-all ${modelType === "NN" ? "bg-[hsl(var(--panel-lighter))] text-[hsl(var(--foreground-active))]" : "text-[#52525b]"}`}>NN</button>
                      <button onClick={() => setModelType("CNN")} className={`py-1 rounded text-[10px] font-bold transition-all ${modelType === "CNN" ? "bg-[hsl(var(--panel-lighter))] text-[hsl(var(--foreground-active))]" : "text-[#52525b]"}`}>CNN</button>
                    </div>
                    <div className="fixed -translate-x-full ml-[-20px] w-44 bg-[hsl(var(--panel-lighter))]/95 backdrop-blur-sm text-[hsl(var(--foreground-active))] text-[10px] p-2.5 rounded-md opacity-0 group-hover:opacity-100 transition-all duration-300 delay-[1500ms] pointer-events-none border border-[hsl(var(--border-muted))] shadow-[0_0_30px_rgba(0,0,0,0.5)] z-[9999] leading-snug">
                      <div className="font-bold text-[#3b82f6] mb-1 uppercase text-[8px] tracking-widest border-b border-[hsl(var(--border-muted))] pb-1">Base Class</div>
                      Choose between standard Multi-Layer Perceptron (NN) or Convolutional Neural Network (CNN).
                    </div>
                  </div>
                  
                  <ControlInput 
                    label="Random Seed" value={seed} onChange={(v: string) => setSeed(v)} 
                    tooltip="Sets the reproducibility of the weights initialization."
                  />
                  
                  {modelType === "CNN" && (
                    <ControlInput 
                      label="Kernel Size" value={kernelSize} onChange={(v: string) => setKernelSize(v)} min={1} max={15} step={2} 
                      tooltip="Defines the 'window' size for convolutional filters (typically 3, 5, or 7)."
                    />
                  )}
                </div>
              </InspectorSection>

              <InspectorSection value="optuna" title="Optuna Settings">
                <div className="space-y-4 pt-2">
                  <ControlInput 
                    label="Trial Budget" value={trials} onChange={(v: string) => setTrials(v)} min={1} max={500} 
                    tooltip="How many different configurations Optuna should try before picking the best one."
                  />
                  {isAdvancedMode && (
                    <RangeControl 
                      label="LR Bounds" value={lrRange} onChange={(v: string) => setLrRange(v)} min={0.00001} max={0.1} step={0.00001} 
                      tooltip="The range of 'step sizes' (learning rate) the optimizer can explore."
                    />
                  )}
                  
                  {modelType === "NN" ? (
                    <>
                      <RangeControl 
                        label="Layers (Range)" value={layersRange} onChange={(v: string) => setLayersRange(v)} min={1} max={20} 
                        tooltip="The minimum and maximum depth (number of hidden layers) of the network."
                      />
                      <RangeControl 
                        label="Layer Size (Range)" value={layerSizeRange} onChange={(v: string) => setLayerSizeRange(v)} min={8} max={2048} step={8} 
                        tooltip="The range of neurons in each hidden layer."
                      />
                    </>
                  ) : (
                    <>
                      <RangeControl 
                        label="Conv Blocks (Range)" value={convBlocks} onChange={(v: string) => setConvBlocks(v)} min={1} max={10} 
                        tooltip="The range of convolutional blocks in the CNN architecture."
                      />
                      <RangeControl 
                        label="Hidden Dim (Range)" value={hDimRange} onChange={(v: string) => setHDimRange(v)} min={8} max={1024} step={8} 
                        tooltip="The range of units in the final dense layer before output."
                      />
                    </>
                  )}
                  {isAdvancedMode && (
                    <RangeControl 
                      label="Dropout (Range)" value={dropoutRange} onChange={(v: string) => setDropoutRange(v)} min={0.0} max={0.9} step={0.05} 
                      tooltip="The range of regularization to prevent overfitting by randomly disabling neurons."
                    />
                  )}
                </div>
              </InspectorSection>
            </Accordion.Root>
          </Tabs.Content>

          <Tabs.Content value="perf" className="space-y-1">
            <Accordion.Root type="multiple" defaultValue={["system", "hardware"]} className="w-full">
              <InspectorSection value="hardware" title="Hardware Monitoring">
                <div className="grid grid-cols-2 gap-3 pt-2">
                  <HardwarePanel 
                    label="CPU Usage" 
                    util={(hardwareStats?.cpu_percent ?? 0).toFixed(1)} 
                    extra="System" 
                  />
                  <HardwarePanel 
                    label="RAM Usage" 
                    util={(hardwareStats?.ram_percent ?? 0).toFixed(0)} 
                    extra={`${hardwareStats?.ram_used_gb || 0} / ${hardwareStats?.ram_total_gb || 0} GB`} 
                  />
                  <HardwarePanel 
                    label="GPU Usage" 
                    util={hardwareStats?.gpu_use_percent || 0} 
                    extra={`Temp: ${hardwareStats?.gpu_temp_c || 0}°C`} 
                  />
                  <HardwarePanel 
                    label="VRAM Usage" 
                    util={hardwareStats?.vram_percent || 0} 
                    extra={`${hardwareStats?.vram_used_gb || 0} / ${hardwareStats?.vram_total_gb || 0} GB`} 
                  />
                </div>
              </InspectorSection>

              <InspectorSection value="system" title="Runtime Performance">
                <div className="space-y-4 pt-2">
                  <div className="space-y-1.5 group relative">
                    <label className="text-[9px] font-bold text-[#52525b] uppercase flex items-center gap-1.5 cursor-help">
                      Compute Device
                      <Info size={10} className="text-[#3f3f46] group-hover:text-[#3b82f6] transition-colors" />
                    </label>
                    <div className="grid grid-cols-2 bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded p-0.5">
                      <button onClick={() => setDeviceChoice("cuda")} className={`py-1 rounded text-[10px] font-bold transition-all ${deviceChoice === "cuda" ? "bg-[#3b82f6] text-[hsl(var(--foreground-active))]" : "text-[#52525b] hover:text-[hsl(var(--foreground))]"}`}>GPU (CUDA)</button>
                      <button onClick={() => setDeviceChoice("cpu")} className={`py-1 rounded text-[10px] font-bold transition-all ${deviceChoice === "cpu" ? "bg-[hsl(var(--panel-lighter))] text-[hsl(var(--foreground-active))]" : "text-[#52525b] hover:text-[hsl(var(--foreground))]"}`}>CPU</button>
                    </div>
                    <div className="fixed -translate-x-full ml-[-20px] w-44 bg-[hsl(var(--panel-lighter))]/95 backdrop-blur-sm text-[hsl(var(--foreground-active))] text-[10px] p-2.5 rounded-md opacity-0 group-hover:opacity-100 transition-all duration-300 delay-[1500ms] pointer-events-none border border-[hsl(var(--border-muted))] shadow-[0_0_30px_rgba(0,0,0,0.5)] z-[9999] leading-snug">
                      <div className="font-bold text-[#3b82f6] mb-1 uppercase text-[8px] tracking-widest border-b border-[hsl(var(--border-muted))] pb-1">Compute Device</div>
                      Force training on GPU (CUDA) or CPU. CPU is often faster for very small models.
                    </div>
                  </div>

                  {deviceChoice === "cuda" && (
                    <div className="space-y-1.5 pt-1">
                      <div className="flex items-center justify-between">
                        <label className="text-[9px] font-bold text-[#52525b] uppercase">Target GPU</label>
                        <button 
                          onClick={async () => {
                            try {
                              const res = await fetch(`${API_URL}/gpus`, {
                                headers: { "X-API-Key": API_KEY }
                              });
                              if (!res.ok) throw new Error("Failed to fetch GPUs");
                              const data = await res.json();
                              setGpuList(data);
                            } catch (e: any) {
                              setError(`GPU Refresh Failed: ${e.message}`);
                            }
                          }}
                          className="text-[8px] font-bold text-[#3b82f6] hover:text-[#60a5fa] uppercase flex items-center gap-1"
                        >
                          <RefreshCw size={8} /> Refresh
                        </button>
                      </div>
                      
                      {gpuList.length > 0 ? (
                        <select 
                          value={gpuChoice} 
                          onChange={(e) => setGpuChoice(parseInt(e.target.value))}
                          className="w-full bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded px-2.5 py-1.5 text-[11px] text-[hsl(var(--foreground-active))] focus:outline-none focus:border-[#3b82f6] transition-colors"
                        >
                          {gpuList.map((gpu: any) => (
                            <option key={gpu.id} value={gpu.id}>
                              {gpu.name}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <div className="w-full bg-[hsl(var(--background))] border border-red-500/20 rounded px-2.5 py-1.5 text-[10px] text-red-400 font-bold italic flex items-center gap-2">
                          <AlertCircle size={10} />
                          No CUDA GPUs Detected
                        </div>
                      )}
                    </div>
                  )}

                  <ControlInput 
                    label="Max Epochs" value={maxEpochs} onChange={(v: string) => setMaxEpochs(v)} min={0} max={100} step={1} 
                    tooltip="The maximum number of passes through the data for each trial."
                  />
                  <ControlInput 
                    label="Early Stop Patience" value={patience} onChange={(v: string) => setPatience(v)} min={1} max={1000} 
                    tooltip="How many epochs to wait for improvement before giving up on a trial."
                  />
                  <ControlInput 
                    label="GPU Throttle (s)" value={gpuThrottle} onChange={(v: string) => setGpuThrottle(v)} min={0} max={0.25} step={0.01} 
                    tooltip="Introduces a small sleep between epochs to lower GPU utilization and heat."
                  />
                  <RangeControl 
                    label="CNN Filter Cap (Range)" value={cnnFilterCapRange} onChange={(v: string) => setCnnFilterCapRange(v)} min={16} max={4096} step={16} 
                    tooltip="Limits the maximum number of filters in deep layers. Optuna will explore caps in this range to prevent VRAM crashes while finding the best architecture."
                  />
                </div>
              </InspectorSection>
            </Accordion.Root>
          </Tabs.Content>
        </div>
      </Tabs.Root>
    </aside>
  );
}

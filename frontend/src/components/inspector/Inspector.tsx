import React from "react";
import * as Tabs from "@radix-ui/react-tabs";
import * as Accordion from "@radix-ui/react-accordion";
import { Info, RefreshCw, AlertCircle } from "lucide-react";

import { InspectorSection, ControlInput, RangeControl } from "../common/Inputs";
import { TabTrigger } from "../common/UIComponents";
import { HardwarePanel } from "../common/Cards";
import { useTrainingStore } from "@/store/useTrainingStore";
import { useApi } from "../ApiProvider";

const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "nftool-dev-key";

export function Inspector({
  setError,
}: {
  setError: (err: string | null) => void;
}) {
  const { apiUrl: API_URL } = useApi();
  // Config States from Store
  const {
    modelType,
    setModelType,
    seed,
    setSeed,
    patience,
    setPatience,
    trials,
    setTrials,
    lrRange,
    setLrRange,
    convBlocks,
    setConvBlocks,
    kernelSize,
    setKernelSize,
    layersRange,
    setLayersRange,
    layerSizeRange,
    setLayerSizeRange,
    dropoutRange,
    setDropoutRange,
    hDimRange,
    setHDimRange,
    maxEpochs,
    setMaxEpochs,
    gpuThrottle,
    setGpuThrottle,
    cnnFilterCapRange,
    setCnnFilterCapRange,
    batchSize,
    setBatchSize,

    // Dataset States
    datasets,
    selectedPredictor,
    setSelectedPredictor,
    selectedTarget,
    setSelectedTarget,

    // System States
    isAdvancedMode,
    deviceChoice,
    setDeviceChoice,
    gpuChoice,
    setGpuChoice,
    gpuList,
    setGpuList,
    hardwareStats,
  } = useTrainingStore();

  return (
    <aside
      className="h-full flex flex-col bg-zinc-950 border-l border-zinc-800 outline-none"
      tabIndex={-1}
    >
      <Tabs.Root
        defaultValue="model"
        className="flex flex-col flex-1 overflow-hidden"
      >
        <div className="px-4 pt-4 shrink-0 bg-zinc-950">
          <Tabs.List className="flex border-b border-zinc-800 gap-6">
            <TabTrigger value="model" label="Model" />
            <TabTrigger value="perf" label="Performance" />
          </Tabs.List>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
          <Tabs.Content value="model" className="space-y-1">
            <Accordion.Root
              type="multiple"
              defaultValue={["dataset", "arch", "optuna"]}
              className="w-full"
            >
              <InspectorSection value="dataset" title="Dataset Assets">
                <div className="space-y-4 pt-2">
                  <div className="space-y-1.5">
                    <label className="text-[9px] font-bold text-zinc-500 uppercase">
                      Predictors (X)
                    </label>
                    <select
                      value={selectedPredictor}
                      onChange={(e) => setSelectedPredictor(e.target.value)}
                      data-testid="select-predictors"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1.5 text-[11px] text-white focus:outline-none focus:border-[hsl(var(--primary))] transition-colors"
                    >
                      {datasets?.map((d: { name?: string; path?: string }) => (
                        <option key={d.path} value={d.path}>
                          {d.name ? d.name.split("/").pop() : "Unknown"}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-[9px] font-bold text-zinc-500 uppercase">
                      Targets (y)
                    </label>
                    <select
                      value={selectedTarget}
                      onChange={(e) => setSelectedTarget(e.target.value)}
                      data-testid="select-targets"
                      className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1.5 text-[11px] text-white focus:outline-none focus:border-[hsl(var(--primary))] transition-colors"
                    >
                      {datasets?.map((d: { name?: string; path?: string }) => (
                        <option key={d.path} value={d.path}>
                          {d.name ? d.name.split("/").pop() : "Unknown"}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </InspectorSection>

              <InspectorSection value="arch" title="Architecture">
                <div className="space-y-4 pt-2">
                  <div className="space-y-1.5 group relative">
                    <label className="text-[9px] font-bold text-zinc-500 uppercase flex items-center gap-1.5 cursor-help">
                      Base Class
                      <Info
                        size={10}
                        className="text-zinc-600 group-hover:text-[hsl(var(--primary))] transition-colors"
                      />
                    </label>
                    <div className="grid grid-cols-2 bg-zinc-900 border border-zinc-800 rounded p-0.5">
                      <button
                        onClick={() => {
                          try {
                            setModelType("NN");
                          } catch (err) {
                            console.warn("Failed to set model type NN:", err);
                          }
                        }}
                        className={`py-1 rounded text-[10px] font-bold transition-all ${modelType === "NN" ? "bg-zinc-800 text-white" : "text-zinc-500"}`}
                      >
                        NN
                      </button>
                      <button
                        onClick={() => {
                          try {
                            setModelType("CNN");
                          } catch (err) {
                            console.warn("Failed to set model type CNN:", err);
                          }
                        }}
                        className={`py-1 rounded text-[10px] font-bold transition-all ${modelType === "CNN" ? "bg-zinc-800 text-white" : "text-zinc-500"}`}
                      >
                        CNN
                      </button>
                    </div>
                  </div>

                  <ControlInput
                    label="Random Seed"
                    value={seed}
                    onChange={setSeed}
                    tooltip="Sets the reproducibility of the weights initialization."
                  />

                  {modelType === "CNN" && (
                    <ControlInput
                      label="Kernel Size"
                      value={kernelSize}
                      onChange={setKernelSize}
                      min={1}
                      max={15}
                      step={2}
                      tooltip="Defines the 'window' size for convolutional filters (typically 3, 5, or 7)."
                    />
                  )}
                </div>
              </InspectorSection>

              <InspectorSection value="optuna" title="Optuna Settings">
                <div className="space-y-4 pt-2">
                  <ControlInput
                    label="Trial Budget"
                    value={trials}
                    onChange={setTrials}
                    min={1}
                    max={500}
                    tooltip="How many different configurations Optuna should try before picking the best one."
                  />
                  {isAdvancedMode && (
                    <RangeControl
                      label="LR Bounds"
                      value={lrRange}
                      onChange={setLrRange}
                      min={0.00001}
                      max={0.1}
                      step={0.00001}
                      tooltip="The range of 'step sizes' (learning rate) the optimizer can explore."
                    />
                  )}

                  {modelType === "NN" ? (
                    <>
                      <RangeControl
                        label="Layers (Range)"
                        value={layersRange}
                        onChange={setLayersRange}
                        min={1}
                        max={20}
                        tooltip="The minimum and maximum depth (number of hidden layers) of the network."
                      />
                      <RangeControl
                        label="Layer Size (Range)"
                        value={layerSizeRange}
                        onChange={setLayerSizeRange}
                        min={8}
                        max={2048}
                        step={8}
                        tooltip="The range of neurons in each hidden layer."
                      />
                    </>
                  ) : (
                    <>
                      <RangeControl
                        label="Conv Blocks (Range)"
                        value={convBlocks}
                        onChange={setConvBlocks}
                        min={1}
                        max={10}
                        tooltip="The range of convolutional blocks in the CNN architecture."
                      />
                      <RangeControl
                        label="Hidden Dim (Range)"
                        value={hDimRange}
                        onChange={setHDimRange}
                        min={8}
                        max={1024}
                        step={8}
                        tooltip="The range of units in the final dense layer before output."
                      />
                    </>
                  )}
                  {isAdvancedMode && (
                    <RangeControl
                      label="Dropout (Range)"
                      value={dropoutRange}
                      onChange={setDropoutRange}
                      min={0.0}
                      max={0.9}
                      step={0.05}
                      tooltip="The range of regularization to prevent overfitting by randomly disabling neurons."
                    />
                  )}
                </div>
              </InspectorSection>
            </Accordion.Root>
          </Tabs.Content>

          <Tabs.Content value="perf" className="space-y-1">
            <Accordion.Root
              type="multiple"
              defaultValue={["system", "hardware"]}
              className="w-full"
            >
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
                    <label className="text-[9px] font-bold text-zinc-500 uppercase flex items-center gap-1.5 cursor-help">
                      Compute Device
                    </label>
                    <div className="grid grid-cols-2 bg-zinc-900 border border-zinc-800 rounded p-0.5">
                      <button
                        onClick={() => setDeviceChoice("cuda")}
                        className={`py-1 rounded text-[10px] font-bold transition-all ${deviceChoice === "cuda" ? "bg-[hsl(var(--primary))] text-[hsl(var(--foreground-active))]" : "text-zinc-500 hover:text-zinc-300"}`}
                      >
                        GPU (CUDA)
                      </button>
                      <button
                        onClick={() => setDeviceChoice("cpu")}
                        className={`py-1 rounded text-[10px] font-bold transition-all ${deviceChoice === "cpu" ? "bg-zinc-800 text-white" : "text-zinc-500 hover:text-zinc-300"}`}
                      >
                        CPU
                      </button>
                    </div>
                  </div>

                  {deviceChoice === "cuda" && (
                    <div className="space-y-1.5 pt-1">
                      <div className="flex items-center justify-between">
                        <label className="text-[9px] font-bold text-zinc-500 uppercase">
                          Target GPU
                        </label>
                        <button
                          onClick={async () => {
                            try {
                              const res = await fetch(`${API_URL}/hardware/gpus`, {
                                headers: { "X-API-Key": API_KEY },
                              });
                              if (!res.ok)
                                throw new Error("Failed to fetch GPUs");
                              const data = await res.json();
                              setGpuList(data);
                            } catch (e: unknown) {
                              const msg =
                                e instanceof Error ? e.message : String(e);
                              setError(`GPU Refresh Failed: ${msg}`);
                            }
                          }}
                          className="text-[8px] font-bold text-[hsl(var(--primary))] hover:text-[hsl(var(--primary-soft))] uppercase flex items-center gap-1"
                        >
                          <RefreshCw size={8} /> Refresh
                        </button>
                      </div>

                      {gpuList.length > 0 ? (
                        <select
                          value={gpuChoice}
                          onChange={(e) =>
                            setGpuChoice(parseInt(e.target.value))
                          }
                          className="w-full bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1.5 text-[11px] text-white focus:outline-none focus:border-[hsl(var(--primary))] transition-colors"
                        >
                          {gpuList.map((gpu: { id: number; name: string }) => (
                            <option key={gpu.id} value={gpu.id}>
                              {gpu.name}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <div className="w-full bg-zinc-900 border border-[hsl(var(--danger)/0.2)] rounded px-2.5 py-1.5 text-[10px] text-[hsl(var(--danger))] font-bold italic flex items-center gap-2">
                          <AlertCircle size={10} />
                          No CUDA GPUs Detected
                        </div>
                      )}
                    </div>
                  )}

                  <ControlInput
                    label="Max Epochs"
                    value={maxEpochs}
                    onChange={setMaxEpochs}
                    min={1}
                    max={10000}
                    step={1}
                    rangeScale="log"
                    tooltip="The maximum number of passes through the data for each trial."
                  />
                  <ControlInput
                    label="Early Stop Patience"
                    value={patience}
                    onChange={setPatience}
                    min={1}
                    max={1000}
                    rangeScale="log"
                    tooltip="How many epochs to wait for improvement before giving up on a trial."
                  />
                  <ControlInput
                    label="Batch Size"
                    value={batchSize}
                    onChange={setBatchSize}
                    min={2}
                    max={4096}
                    step={1}
                    rangeScale="log"
                    tooltip="The number of training samples used in one iteration to update weights."
                  />
                  <ControlInput
                    label="GPU Throttle (s)"
                    value={gpuThrottle}
                    onChange={setGpuThrottle}
                    min={0}
                    max={0.25}
                    step={0.01}
                    tooltip="Introduces a small sleep between epochs to lower GPU utilization and heat."
                  />
                  <RangeControl
                    label="CNN Filter Cap (Range)"
                    value={cnnFilterCapRange}
                    onChange={setCnnFilterCapRange}
                    min={16}
                    max={4096}
                    step={16}
                    tooltip="Limits the maximum number of filters in deep layers."
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

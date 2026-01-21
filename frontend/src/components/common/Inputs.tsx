import React from "react";
import { Info } from "lucide-react";
import * as Accordion from "@radix-ui/react-accordion";
import { ChevronDown } from "lucide-react";
import { Tooltip } from "./Tooltip";

export function InspectorSection({
  value,
  title,
  children,
}: {
  value: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Accordion.Item
      value={value}
      className="border-b border-[hsl(var(--border))]"
    >
      <Accordion.Header className="flex">
        <Accordion.Trigger className="flex flex-1 items-center justify-between py-3 text-[11px] font-bold text-[hsl(var(--foreground-active))] hover:bg-[hsl(var(--panel-lighter))/0.3] px-4 transition-colors group outline-none uppercase tracking-widest">
          {title}
          <ChevronDown
            size={14}
            className="text-[hsl(var(--foreground-subtle))] group-data-[state=open]:rotate-180 transition-transform"
          />
        </Accordion.Trigger>
      </Accordion.Header>
      <Accordion.Content className="pb-4 px-4 overflow-hidden data-[state=closed]:animate-slideUp data-[state=open]:animate-slideDown bg-[hsl(var(--panel))]/20">
        {children}
      </Accordion.Content>
    </Accordion.Item>
  );
}

export function ControlInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  tooltip,
  rangeScale = "linear",
}: {
  label: string;
  value: string;
  onChange?: (v: string) => void;
  min?: number;
  max?: number;
  step?: number;
  tooltip?: string;
  rangeScale?: "linear" | "log";
}) {
  const numValue = parseFloat(value);
  const isLogScale =
    rangeScale === "log" &&
    min !== undefined &&
    max !== undefined &&
    min > 0 &&
    max > 0 &&
    max > min;
  const sliderResolution = 1000;
  const clamp = (v: number, minValue: number, maxValue: number) =>
    Math.min(maxValue, Math.max(minValue, v));
  const logMin = min ?? 1;
  const logMax = max ?? logMin + 1;
  const ratio = isLogScale ? logMax / logMin : 1;
  const toLogValue = (t: number) => logMin * Math.pow(ratio, t);
  const toLogPosition = (v: number) =>
    Math.log(v / logMin) / Math.log(ratio);
  const formatValue = (v: number) => {
    const clamped = clamp(v, min ?? v, max ?? v);
    if (!Number.isFinite(step) || step <= 0) {
      return `${clamped}`;
    }
    const rounded = Math.round(clamped / step) * step;
    const decimals = `${step}`.split(".")[1]?.length ?? 0;
    return decimals > 0 ? rounded.toFixed(decimals) : `${Math.round(rounded)}`;
  };

  return (
    <div className="space-y-3 py-3 border-b border-[hsl(var(--border))/0.5] last:border-0 group relative">
      <div className="flex items-center justify-between">
        <label className="text-[11px] font-medium text-[hsl(var(--foreground))] tracking-tight flex items-center gap-2">
          {label}
          {tooltip && (
            <Tooltip content={tooltip} side="top" align="start" sideOffset={6}>
              <button
                type="button"
                className="text-[hsl(var(--foreground-subtle))] transition-colors hover:text-[hsl(var(--foreground-muted))] focus-visible:text-[hsl(var(--primary))]"
                aria-label={`${label} info`}
              >
                <Info size={12} />
              </button>
            </Tooltip>
          )}
        </label>
        <input
          type="text"
          value={value}
          data-testid={`input-${label.toLowerCase().replace(/\s+/g, "-")}`}
          onChange={(e) => onChange?.(e.target.value)}
          className="bg-[hsl(var(--input))] border border-[hsl(var(--border-strong))] rounded px-2 py-1 text-right text-[11px] text-[hsl(var(--foreground-active))] font-mono focus:outline-none focus:border-[hsl(var(--primary)/0.5)] w-20 transition-colors"
        />
      </div>
      {min !== undefined && max !== undefined && (
        <input
          type="range"
          min={isLogScale ? 0 : min}
          max={isLogScale ? sliderResolution : max}
          step={isLogScale ? 1 : step}
          value={
            isLogScale
              ? Math.round(
                  clamp(
                    toLogPosition(
                      clamp(
                        isNaN(numValue) ? min : numValue,
                        min,
                        max,
                      ),
                    ),
                    0,
                    1,
                  ) * sliderResolution,
                )
              : isNaN(numValue)
                ? min
                : numValue
          }
          onChange={(e) => {
            if (!isLogScale) {
              onChange?.(e.target.value);
              return;
            }
            const t = clamp(
              parseFloat(e.target.value) / sliderResolution,
              0,
              1,
            );
            const mappedValue = toLogValue(t);
            onChange?.(formatValue(mappedValue));
          }}
          className="w-full h-[2px] bg-[hsl(var(--border-strong))] rounded-full appearance-none accent-[hsl(var(--primary))] cursor-pointer hover:accent-[hsl(var(--primary-soft))] transition-all"
        />
      )}
    </div>
  );
}

export function RangeControl({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  tooltip,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  min: number;
  max: number;
  step?: number;
  tooltip?: string;
}) {
  const parts =
    typeof value === "string" ? value.split("→").map((p) => p.trim()) : [];
  const vMin = parseFloat(parts[0]) || min;
  const vMax = parseFloat(parts[1]) || max;
  const updateMin = (newMin: string) => onChange(`${newMin} → ${vMax}`);
  const updateMax = (newMax: string) => onChange(`${vMin} → ${newMax}`);

  return (
    <div className="space-y-4 py-4 border-b border-[hsl(var(--border))/0.5] last:border-0 group relative">
      <div className="flex items-center justify-between">
        <label className="text-[11px] font-medium text-[hsl(var(--foreground))] tracking-tight flex items-center gap-2">
          {label}
          {tooltip && (
            <Tooltip content={tooltip} side="top" align="start" sideOffset={6}>
              <button
                type="button"
                className="text-[hsl(var(--foreground-subtle))] transition-colors hover:text-[hsl(var(--foreground-muted))] focus-visible:text-[hsl(var(--primary))]"
                aria-label={`${label} info`}
              >
                <Info size={12} />
              </button>
            </Tooltip>
          )}
        </label>
        <div className="flex items-center gap-1">
          <input
            type="text"
            value={vMin}
            onChange={(e) => updateMin(e.target.value)}
            data-testid={`range-min-${label.toLowerCase().replace(/\s+/g, "-")}`}
            className="bg-[hsl(var(--input))] border border-[hsl(var(--border-strong))] rounded px-2 py-1 text-center text-[10px] text-[hsl(var(--foreground-active))] font-mono focus:outline-none w-14"
          />
          <span className="text-[hsl(var(--foreground-subtle))] text-[10px]">
            →
          </span>
          <input
            type="text"
            value={vMax}
            onChange={(e) => updateMax(e.target.value)}
            data-testid={`range-max-${label.toLowerCase().replace(/\s+/g, "-")}`}
            className="bg-[hsl(var(--input))] border border-[hsl(var(--border-strong))] rounded px-2 py-1 text-center text-[10px] text-[hsl(var(--foreground-active))] font-mono focus:outline-none w-14"
          />
        </div>
      </div>
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <span className="text-[9px] text-[hsl(var(--foreground-dim))] w-6 font-bold">
            MIN
          </span>
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={vMin}
            onChange={(e) => updateMin(e.target.value)}
            className="flex-1 h-[2px] bg-[hsl(var(--border-strong))] rounded-full appearance-none accent-[hsl(var(--primary))] cursor-pointer"
          />
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[9px] text-[hsl(var(--foreground-dim))] w-6 font-bold">
            MAX
          </span>
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={vMax}
            onChange={(e) => updateMax(e.target.value)}
            className="flex-1 h-[2px] bg-[hsl(var(--border-strong))] rounded-full appearance-none accent-[hsl(var(--primary))] cursor-pointer"
          />
        </div>
      </div>
    </div>
  );
}

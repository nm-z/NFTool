import React from "react";
import { LucideIcon } from "lucide-react";

export function SummaryCard({
  icon: Icon,
  label,
  value,
  subValue,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  subValue: string;
}) {
  return (
    <div className="bg-[hsl(var(--panel))] border border-[hsl(var(--border))] rounded-lg p-4 flex items-center gap-4">
      <div className="w-10 h-10 rounded-lg bg-[hsl(var(--panel-lighter))] flex items-center justify-center text-[hsl(var(--primary))]">
        <Icon size={20} />
      </div>
      <div>
        <div className="text-[10px] font-bold text-[hsl(var(--foreground-dim))] uppercase">
          {label}
        </div>
        <div className="text-[16px] font-bold text-[hsl(var(--foreground-active))] leading-tight mt-0.5">
          {value}
        </div>
        <div className="text-[10px] text-[hsl(var(--foreground-dim))] font-mono mt-0.5">
          {subValue}
        </div>
      </div>
    </div>
  );
}

export function HardwarePanel({
  label,
  util,
  extra,
}: {
  label: string;
  util: number | string;
  extra: string;
}) {
  const utilNum = typeof util === "string" ? parseFloat(util) : util;

  return (
    <div className="bg-[hsl(var(--panel))] border border-[hsl(var(--border))] rounded-lg p-3">
      <div className="text-[9px] font-bold text-[hsl(var(--foreground-dim))] uppercase mb-2">
        {label}
      </div>
      <div className="flex items-end justify-between mb-1">
        <div className="text-[14px] font-bold text-[hsl(var(--foreground-active))] font-mono">
          {util}%
        </div>
        <div className="text-[9px] text-[hsl(var(--foreground-dim))] mb-0.5">
          {extra}
        </div>
      </div>
      <div className="h-1 w-full bg-[hsl(var(--panel-lighter))] rounded-full overflow-hidden">
        <div
          className="h-full bg-[hsl(var(--primary))] transition-all duration-1000"
          style={{ width: `${utilNum}%` }}
        ></div>
      </div>
    </div>
  );
}

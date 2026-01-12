import React from "react";
import { LucideIcon } from "lucide-react";

export function SummaryCard({ icon: Icon, label, value, subValue }: { icon: LucideIcon, label: string, value: string, subValue: string }) {
  return (
    <div className="bg-[hsl(var(--panel))] border border-[hsl(var(--border))] rounded-lg p-4 flex items-center gap-4">
      <div className="w-10 h-10 rounded-lg bg-[hsl(var(--panel-lighter))] flex items-center justify-center text-[#3b82f6]">
        <Icon size={20} />
      </div>
      <div>
        <div className="text-[10px] font-bold text-[#52525b] uppercase tracking-wider">{label}</div>
        <div className="text-[16px] font-bold text-[hsl(var(--foreground-active))] leading-tight mt-0.5">{value}</div>
        <div className="text-[10px] text-[#52525b] font-mono mt-0.5">{subValue}</div>
      </div>
    </div>
  );
}

export function HardwarePanel({ label, util, extra }: { label: string, util: number | string, extra: string }) {
  const utilNum = typeof util === 'string' ? parseFloat(util) : util;
  
  return (
    <div className="bg-[hsl(var(--panel))] border border-[hsl(var(--border))] rounded-lg p-3">
      <div className="text-[9px] font-bold text-[#52525b] uppercase mb-2">{label}</div>
      <div className="flex items-end justify-between mb-1">
        <div className="text-[14px] font-bold text-[hsl(var(--foreground-active))] font-mono">{util}%</div>
        <div className="text-[9px] text-[#52525b] mb-0.5">{extra}</div>
      </div>
      <div className="h-1 w-full bg-[hsl(var(--panel-lighter))] rounded-full overflow-hidden">
        <div className="h-full bg-[#3b82f6] transition-all duration-1000" style={{ width: `${utilNum}%` }}></div>
      </div>
    </div>
  );
}

export function PlotCard({ title, src }: { title: string, src: string }) {
  return (
    <div className="border border-[hsl(var(--border))] rounded-xl overflow-hidden bg-[hsl(var(--panel))]/50 flex flex-col">
      <div className="px-4 py-2 border-b border-[hsl(var(--border))] bg-[hsl(var(--panel))]">
        <span className="text-[10px] font-bold uppercase tracking-widest text-[#52525b]">{title}</span>
      </div>
      <div className="flex-1 p-2 flex items-center justify-center min-h-[300px]">
        <img 
          src={src} 
          alt={title} 
          className="max-h-full max-w-full object-contain"
          onError={(e: any) => {
            e.target.src = "https://placehold.co/600x400/0c0c0e/52525b?text=Plot+Not+Found";
          }}
        />
      </div>
    </div>
  );
}

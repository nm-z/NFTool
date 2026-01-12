import React from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { LucideIcon } from "lucide-react";

export function NavIcon({ icon: Icon, active, onClick, tooltip }: { icon: LucideIcon, active: boolean, onClick: () => void, tooltip: string }) {
  return (
    <button 
      type="button"
      onClick={onClick}
      className={`p-2.5 rounded-md transition-all relative group ${active ? "text-[hsl(var(--foreground-active))] bg-[hsl(var(--panel-lighter))]" : "text-[#52525b] hover:text-[hsl(var(--foreground))] hover:bg-[hsl(var(--panel-lighter))]/50"}`}
    >
      <Icon size={20} strokeWidth={active ? 2.5 : 2} />
      {active && <div className="absolute left-[-4px] top-2 bottom-2 w-[3px] bg-[#3b82f6] rounded-r-full"></div>}
      <div className="absolute left-[64px] bg-[hsl(var(--panel-lighter))] text-[hsl(var(--foreground-active))] text-[10px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity delay-[1500ms] whitespace-nowrap pointer-events-none border border-[hsl(var(--border-muted))] shadow-xl z-50">
        {tooltip}
      </div>
    </button>
  );
}

export function IconButton({ icon: Icon, tooltip, onClick }: { icon: LucideIcon, tooltip: string, onClick: () => void }) {
  return (
    <button 
      type="button"
      onClick={onClick} 
      className="p-1.5 rounded hover:bg-[hsl(var(--panel-lighter))] text-[hsl(var(--foreground))] hover:text-[hsl(var(--foreground-active))] transition-all relative group"
    >
      <Icon size={16} />
      <div className="absolute top-[32px] left-1/2 -translate-x-1/2 bg-[hsl(var(--panel-lighter))] text-[hsl(var(--foreground-active))] text-[9px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity delay-[1500ms] whitespace-nowrap pointer-events-none border border-[hsl(var(--border-muted))] shadow-xl z-50">
        {tooltip}
      </div>
    </button>
  );
}

export function ResourceBadge({ label, value, color }: { label: string, value: string, color?: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[#52525b] font-bold">{label}</span>
      <span className={`font-mono ${color || 'text-[hsl(var(--foreground-active))]'}`}>{value}</span>
    </div>
  );
}

export function TabTrigger({ value, label }: { value: string, label: string }) {
  return (
    <Tabs.Trigger 
      value={value} 
      className="flex-1 text-[11px] font-medium py-1.5 rounded-md transition-all data-[state=active]:bg-[#3b82f6] data-[state=active]:text-[hsl(var(--foreground-active))] text-[hsl(var(--foreground))] hover:text-[hsl(var(--foreground-active))] outline-none"
    >
      {label}
    </Tabs.Trigger>
  );
}

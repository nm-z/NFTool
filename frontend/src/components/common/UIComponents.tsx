import React from "react";
import * as Tabs from "@radix-ui/react-tabs";
import { LucideIcon } from "lucide-react";
import { Tooltip } from "./Tooltip";

export function NavIcon({
  icon: Icon,
  active,
  onClick,
  tooltip,
}: {
  icon: LucideIcon;
  active: boolean;
  onClick: () => void;
  tooltip: string;
}) {
  return (
    <Tooltip content={tooltip} side="right" align="center" sideOffset={12}>
      <button
        type="button"
        onClick={onClick}
        aria-label={tooltip}
        className={`p-2.5 rounded-md transition-all relative ${active ? "text-[hsl(var(--foreground-active))] bg-[hsl(var(--panel-lighter))]" : "text-[hsl(var(--foreground-dim))] hover:text-[hsl(var(--foreground))] hover:bg-[hsl(var(--panel-lighter))]/50"}`}
      >
        <Icon size={20} strokeWidth={active ? 2.5 : 2} />
        {active && (
          <div className="absolute left-[-4px] top-2 bottom-2 w-[3px] bg-[hsl(var(--primary))] rounded-r-full"></div>
        )}
      </button>
    </Tooltip>
  );
}

export function IconButton({
  icon: Icon,
  tooltip,
  onClick,
}: {
  icon: LucideIcon;
  tooltip: string;
  onClick: () => void;
}) {
  return (
    <Tooltip content={tooltip} side="bottom" align="center" sideOffset={10}>
      <button
        type="button"
        onClick={onClick}
        aria-label={tooltip}
        className="p-1.5 rounded hover:bg-[hsl(var(--panel-lighter))] text-[hsl(var(--foreground))] hover:text-[hsl(var(--foreground-active))] transition-all"
      >
        <Icon size={16} />
      </button>
    </Tooltip>
  );
}

export function ResourceBadge({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[hsl(var(--foreground-dim))] font-bold">
        {label}
      </span>
      <span
        className={`font-mono ${color || "text-[hsl(var(--foreground-active))]"}`}
      >
        {value}
      </span>
    </div>
  );
}

export function TabTrigger({ value, label }: { value: string; label: string }) {
  return (
    <Tabs.Trigger
      value={value}
      className="flex-1 text-[11px] font-medium py-1.5 rounded-md transition-all data-[state=active]:bg-[hsl(var(--primary))] data-[state=active]:text-[hsl(var(--foreground-active))] text-[hsl(var(--foreground))] hover:text-[hsl(var(--foreground-active))] outline-none"
    >
      {label}
    </Tabs.Trigger>
  );
}

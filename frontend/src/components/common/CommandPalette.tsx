import React from "react";
import { Command } from "cmdk";
import { Play, StopCircle, Layers, Database, LucideIcon } from "lucide-react";
import type { WorkspaceType } from "@/store/useTrainingStore";

function CommandItem({
  icon: Icon,
  label,
  onSelect,
}: {
  icon: LucideIcon;
  label: string;
  onSelect?: () => void;
}) {
  return (
    <Command.Item
      onSelect={() => onSelect?.()}
      className="flex items-center gap-3 px-4 py-2 text-[13px] cursor-default select-none aria-selected:bg-[hsl(var(--panel-lighter))] aria-selected:text-[hsl(var(--foreground-active))] rounded-md mx-2 transition-colors outline-none"
    >
      <Icon size={16} className="text-[hsl(var(--foreground-dim))]" />
      {label}
    </Command.Item>
  );
}

interface CommandPaletteProps {
  open: boolean;
  setOpen: (open: boolean) => void;
  handleStartTraining: () => void;
  handleAbortTraining: () => void;
  setActiveWorkspace: (ws: WorkspaceType) => void;
}

export function CommandPalette({
  open,
  setOpen,
  handleStartTraining,
  handleAbortTraining,
  setActiveWorkspace,
}: CommandPaletteProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh] bg-[hsl(var(--background)/0.6)] backdrop-blur-[2px]">
      <div className="w-[640px] max-h-[400px]">
        <Command
          label="Command Palette"
          onKeyDown={(e) => e.key === "Escape" && setOpen(false)}
        >
          <Command.Input
            placeholder="Search system actions (Ctrl+K)..."
            autoFocus
            className="w-full bg-[hsl(var(--panel-lighter))] border border-[hsl(var(--border-muted))] text-[hsl(var(--foreground-active))] px-4 py-3 rounded-t-lg outline-none"
          />
          <Command.List className="bg-[hsl(var(--panel))] border-x border-b border-[hsl(var(--border-muted))] rounded-b-lg overflow-y-auto max-h-[300px] custom-scrollbar p-2">
            <Command.Empty className="p-4 text-[12px] text-[hsl(var(--foreground-dim))]">
              No results found.
            </Command.Empty>
            <Command.Group
              heading="Execution"
              className="text-[10px] text-[hsl(var(--foreground-dim))] uppercase font-bold px-2 py-1"
            >
              <CommandItem
                icon={Play}
                label="Execute Active Core"
                onSelect={() => {
                  setOpen(false);
                  handleStartTraining();
                }}
              />
              <CommandItem
                icon={StopCircle}
                label="Terminate Running Pass"
                onSelect={() => {
                  setOpen(false);
                  handleAbortTraining();
                }}
              />
            </Command.Group>
            <Command.Group
              heading="Workspace"
              className="text-[10px] text-[hsl(var(--foreground-dim))] uppercase font-bold px-2 py-1 mt-2"
            >
              <CommandItem
                icon={Layers}
                label="Switch to Training"
                onSelect={() => {
                  setOpen(false);
                  setActiveWorkspace("Train");
                }}
              />
              <CommandItem
                icon={Database}
                label="Open Library"
                onSelect={() => {
                  setOpen(false);
                  setActiveWorkspace("Library");
                }}
              />
            </Command.Group>
          </Command.List>
        </Command>
      </div>
      <div
        className="absolute inset-0 -z-10"
        onClick={() => setOpen(false)}
      ></div>
    </div>
  );
}

import React from "react";
import { Layers, Database, Settings } from "lucide-react";
import { NavIcon } from "../common/UIComponents";
import { WorkspaceType } from "@/store/useTrainingStore";

interface SidebarProps {
  activeWorkspace: WorkspaceType;
  setActiveWorkspace: (ws: WorkspaceType) => void;
}

export function Sidebar({ activeWorkspace, setActiveWorkspace }: SidebarProps) {
  return (
    <aside className="w-[56px] border-r border-[hsl(var(--border))] bg-[hsl(var(--panel))] flex flex-col items-center py-4 gap-4 shrink-0 z-40">
      <div className="flex-1 w-full flex flex-col items-center gap-1">
        <NavIcon icon={Layers} active={activeWorkspace === "Train"} onClick={() => setActiveWorkspace("Train")} tooltip="Train" />
        <NavIcon icon={Database} active={activeWorkspace === "Library"} onClick={() => setActiveWorkspace("Library")} tooltip="Library" />
      </div>
      <NavIcon icon={Settings} active={false} onClick={() => {}} tooltip="Global Settings" />
    </aside>
  );
}

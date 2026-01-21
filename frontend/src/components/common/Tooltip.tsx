import React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";

const BASE_TOOLTIP_CLASSES =
  "z-[100] max-w-[260px] rounded border border-[hsl(var(--border-muted))] bg-[hsl(var(--panel-lighter))] px-2 py-1 text-[10px] leading-relaxed text-[hsl(var(--foreground-active))] shadow-xl opacity-0 transition-opacity data-[state=delayed-open]:opacity-100 data-[state=instant-open]:opacity-100";

type TooltipProps = {
  content: React.ReactNode;
  children: React.ReactNode;
  side?: TooltipPrimitive.TooltipContentProps["side"];
  align?: TooltipPrimitive.TooltipContentProps["align"];
  sideOffset?: number;
  delayDuration?: number;
  className?: string;
};

export function Tooltip({
  content,
  children,
  side = "top",
  align = "center",
  sideOffset = 8,
  delayDuration = 600,
  className,
}: TooltipProps) {
  if (!content) {
    return <>{children}</>;
  }

  return (
    <TooltipPrimitive.Provider
      delayDuration={delayDuration}
      skipDelayDuration={150}
      disableHoverableContent
    >
      <TooltipPrimitive.Root>
        <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
        <TooltipPrimitive.Portal>
          <TooltipPrimitive.Content
            side={side}
            align={align}
            sideOffset={sideOffset}
            className={
              className ? `${BASE_TOOLTIP_CLASSES} ${className}` : BASE_TOOLTIP_CLASSES
            }
          >
            {content}
            <TooltipPrimitive.Arrow className="fill-[hsl(var(--panel-lighter))]" />
          </TooltipPrimitive.Content>
        </TooltipPrimitive.Portal>
      </TooltipPrimitive.Root>
    </TooltipPrimitive.Provider>
  );
}

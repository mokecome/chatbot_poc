"use client";

import * as React from "react";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area@1.2.3";
import { cn } from "./utils";
import { HOTEL_PRIMARY } from "../../styles/hotelTheme";

/**
 * Enhanced ScrollArea with always-visible scrollbars
 * Fixes common issues with Radix ScrollArea hiding scrollbars
 */

const EnhancedScrollArea = React.forwardRef<
  React.ElementRef<typeof ScrollAreaPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.Root> & {
    showScrollbar?: boolean;
    scrollbarColor?: string;
  }
>(({ className, children, showScrollbar = true, scrollbarColor = HOTEL_PRIMARY, ...props }, ref) => (
  <ScrollAreaPrimitive.Root
    ref={ref}
    data-slot="enhanced-scroll-area"
    className={cn("relative", className)}
    style={{
      // Force scrollbar gutter space
      scrollbarGutter: "stable always"
    }}
    {...props}
  >
    <ScrollAreaPrimitive.Viewport
      data-slot="enhanced-scroll-area-viewport"
      className="focus-visible:ring-ring/50 size-full rounded-[inherit] transition-[color,box-shadow] outline-none focus-visible:ring-[3px] focus-visible:outline-1"
    >
      {children}
    </ScrollAreaPrimitive.Viewport>
    
    {showScrollbar && (
      <EnhancedScrollBar 
        scrollbarColor={scrollbarColor}
        className="opacity-100"
        style={{
          visibility: 'visible !important',
          opacity: '1 !important'
        }}
      />
    )}
    
    <ScrollAreaPrimitive.Corner />
  </ScrollAreaPrimitive.Root>
));
EnhancedScrollArea.displayName = "EnhancedScrollArea";

const EnhancedScrollBar = React.forwardRef<
  React.ElementRef<typeof ScrollAreaPrimitive.ScrollAreaScrollbar>,
  React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.ScrollAreaScrollbar> & {
    scrollbarColor?: string;
  }
>(({ className, orientation = "vertical", scrollbarColor = HOTEL_PRIMARY, ...props }, ref) => (
  <ScrollAreaPrimitive.ScrollAreaScrollbar
    ref={ref}
    data-slot="enhanced-scroll-area-scrollbar"
    orientation={orientation}
    className={cn(
      "flex touch-none select-none transition-colors",
      // Enhanced visibility and sizing
      orientation === "vertical" &&
        "h-full w-4 border-l-0 p-1 bg-gray-100/80 hover:bg-gray-200/80",
      orientation === "horizontal" &&
        "h-4 flex-col border-t-0 p-1 bg-gray-100/80 hover:bg-gray-200/80",
      // Force visibility
      "opacity-100 data-[state=hidden]:opacity-100 data-[state=visible]:opacity-100",
      className,
    )}
    style={{
      visibility: 'visible !important',
      opacity: '1 !important',
      ...props.style
    }}
    {...props}
  >
    <ScrollAreaPrimitive.ScrollAreaThumb
      data-slot="enhanced-scroll-area-thumb"
      className={cn(
        "relative flex-1 rounded-full transition-colors",
        // Custom scrollbar colors
        "min-h-[30px] min-w-[30px]",
        // Default colors that can be overridden
        "hover:opacity-80 active:opacity-90"
      )}
      style={{
        backgroundColor: `${scrollbarColor}60`, // 60% opacity
        '&:hover': {
          backgroundColor: `${scrollbarColor}80` // 80% opacity on hover
        }
      } as React.CSSProperties}
    />
  </ScrollAreaPrimitive.ScrollAreaScrollbar>
));
EnhancedScrollBar.displayName = "EnhancedScrollBar";

export { EnhancedScrollArea, EnhancedScrollBar };

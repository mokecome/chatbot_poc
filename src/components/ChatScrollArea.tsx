import { forwardRef, useEffect, useRef } from "react";
import { ScrollArea, ScrollBar } from "./ui/scroll-area";
import { cn } from "./ui/utils";

interface ChatScrollAreaProps {
  children: React.ReactNode;
  className?: string;
}

export const ChatScrollArea = forwardRef<
  React.ElementRef<typeof ScrollArea>,
  ChatScrollAreaProps
>(({ children, className }, ref) => {
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Force scrollbar visibility by adding CSS custom properties
  useEffect(() => {
    const scrollArea = scrollAreaRef.current;
    if (scrollArea) {
      // Add custom CSS variables to force scrollbar visibility
      scrollArea.style.setProperty('--scrollbar-visibility', 'visible');
      scrollArea.style.setProperty('--scrollbar-opacity', '1');
    }
  }, []);

  return (
    <ScrollArea 
      ref={ref}
      className={cn(
        "flex-1 relative",
        // 確保容器有正確的高度和溢出處理
        "min-h-0 h-full overflow-hidden",
        className
      )}
      style={{
        // 強制顯示滾動條空間
        scrollbarGutter: "stable always"
      }}
    >
      <div 
        ref={scrollAreaRef}
        className="p-4 pr-2 min-h-full"
        style={{
          // 確保內容區域有正確的高度
          minHeight: "100%"
        }}
      >
        {children}
      </div>
      
      {/* 自定義滾動條，強制顯示 */}
      <ScrollBar 
        orientation="vertical"
        className={cn(
          // 強制顯示滾動條
          "w-4 opacity-100 transition-opacity duration-200",
          "border-l-0 p-1",
          // 滾動條軌道背景
          "bg-gray-100/80 hover:bg-gray-200/80",
          // 自定義滾動條樣式
          "[&>div]:bg-[#B38844]/60 [&>div]:hover:bg-[#8F6A32]",
          "[&>div]:data-[state=active]:bg-[#B38844]",
          "[&>div]:rounded-full [&>div]:min-h-[30px]",
          // 確保滾動條總是可見
          "data-[state=hidden]:opacity-100 data-[state=visible]:opacity-100"
        )}
        style={{
          // 覆蓋 Radix 的隱藏邏輯
          visibility: 'visible !important',
          opacity: '1 !important'
        }}
      />
    </ScrollArea>
  );
});

ChatScrollArea.displayName = "ChatScrollArea";

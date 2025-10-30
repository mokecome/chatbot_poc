import { useState } from "react";
import { ScrollArea, ScrollBar } from "./ui/scroll-area";
import { Card, CardContent, CardHeader } from "./ui/card";
import { Button } from "./ui/button";

const ACCENT = "#B38844";

/**
 * 滾動條測試面板 - 用於診斷與展示不同滾動條情境
 */
export function ScrollBarTest() {
  const [messageCount, setMessageCount] = useState(10);

  const generateMessages = (count: number) =>
    Array.from({ length: count }, (_, i) => ({
      id: i,
      content: `測試訊息 ${i + 1}：這是一段用來檢視飯店智能客服訊息長度的範例內容，要確保無論是使用原生滾動還是客製化 ScrollArea，都能可靠顯示。`,
      isUser: i % 2 === 0,
    }));

  const messages = generateMessages(messageCount);

  const renderMessageCard = (message: { id: number; content: string; isUser: boolean }) => (
    <Card
      key={message.id}
      className={`p-3 ${message.isUser ? "ml-auto max-w-[80%] text-white border-0" : "max-w-[80%]"}`}
      style={message.isUser ? { backgroundColor: ACCENT } : undefined}
    >
      <p className="text-sm">{message.content}</p>
    </Card>
  );

  return (
    <div className="p-8 space-y-8">
      <div className="flex gap-4 items-center">
        <h2 className="text-2xl font-semibold">滾動條顯示測試</h2>
        <Button
          onClick={() => setMessageCount(5)}
          variant={messageCount === 5 ? "default" : "outline"}
          size="sm"
        >
          精簡訊息 (5)
        </Button>
        <Button
          onClick={() => setMessageCount(20)}
          variant={messageCount === 20 ? "default" : "outline"}
          size="sm"
        >
          大量訊息 (20)
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <Card className="h-[400px]">
          <CardHeader>
            <h3 className="font-medium">原生滾動區域 (overflow-y-auto)</h3>
            <p className="text-sm text-muted-foreground">使用瀏覽器原生滾動列</p>
          </CardHeader>
          <CardContent className="p-0 h-full">
            <div className="h-full overflow-y-auto p-4 space-y-3">
              {messages.map(renderMessageCard)}
            </div>
          </CardContent>
        </Card>

        <Card className="h-[400px]">
          <CardHeader>
            <h3 className="font-medium">Radix ScrollArea (預設)</h3>
            <p className="text-sm text-muted-foreground">採用 Radix UI 既有設定</p>
          </CardHeader>
          <CardContent className="p-0 h-full">
            <ScrollArea className="h-full">
              <div className="p-4 space-y-3">
                {messages.map(renderMessageCard)}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>

        <Card className="h-[400px]">
          <CardHeader>
            <h3 className="font-medium">Radix ScrollArea (強制顯示)</h3>
            <p className="text-sm text-muted-foreground">強制顯示並改造滾動軸樣式</p>
          </CardHeader>
          <CardContent className="p-0 h-full">
            <ScrollArea className="h-full">
              <div className="p-4 pr-2 space-y-3">
                {messages.map(renderMessageCard)}
              </div>
              <ScrollBar
                className="w-4 bg-gray-100/80 hover:bg-gray-200/80 opacity-100 [&>div]:bg-[#B38844]/60 [&>div]:hover:bg-[#8F6A32] [&>div]:rounded-full [&>div]:min-h-[30px]"
                style={{
                  visibility: "visible !important",
                  opacity: "1 !important",
                }}
              />
            </ScrollArea>
          </CardContent>
        </Card>

        <Card className="h-[400px]">
          <CardHeader>
            <h3 className="font-medium">進階 ChatScrollArea</h3>
            <p className="text-sm text-muted-foreground">使用客製化 ChatScrollArea 組件</p>
          </CardHeader>
          <CardContent className="p-0 h-full relative">
            <div className="absolute inset-0">
              <ScrollArea className="h-full">
                <div className="p-4 pr-2 space-y-3 min-h-full">
                  {messages.map(renderMessageCard)}
                </div>
                <ScrollBar
                  orientation="vertical"
                  className="w-4 opacity-100 border-l-0 p-1 bg-gray-100/80 hover:bg-gray-200/80 [&>div]:bg-[#B38844]/60 [&>div]:hover:bg-[#8F6A32] [&>div]:data-[state=active]:bg-[#B38844] [&>div]:rounded-full [&>div]:min-h-[30px]"
                  style={{
                    visibility: "visible !important",
                    opacity: "1 !important",
                  }}
                />
              </ScrollArea>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <h3 className="font-medium">滾動條常見問題診斷</h3>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 text-sm">
            <div>
              <h4 className="font-medium mb-2">常見問題與處理建議</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-3 bg-red-50 border border-red-200 rounded">
                  <h5 className="font-medium text-red-700 mb-2">
                    問題 1：Radix ScrollArea 預設不顯示
                  </h5>
                  <p className="text-red-600">
                    ScrollArea 需要使用者捲動才顯示滾動條
                  </p>
                  <p className="text-red-600 text-xs mt-1">
                    <strong>解法：</strong> 強制設定 style={{ visibility: 'visible !important' }}
                  </p>
                </div>

                <div className="p-3 bg-red-50 border border-red-200 rounded">
                  <h5 className="font-medium text-red-700 mb-2">
                    問題 2：Flex 容器高度未設定
                  </h5>
                  <p className="text-red-600">父層未設定 min-height，導致內容無法滾動</p>
                  <p className="text-red-600 text-xs mt-1">
                    <strong>解法：</strong> 為父層增加 min-h-0 或使用固定高度
                  </p>
                </div>

                <div className="p-3 bg-red-50 border border-red-200 rounded">
                  <h5 className="font-medium text-red-700 mb-2">
                    問題 3：overflow: hidden
                  </h5>
                  <p className="text-red-600">外層被 overflow: hidden 限制滾動</p>
                  <p className="text-red-600 text-xs mt-1">
                    <strong>解法：</strong> 移除或調整父層 overflow 設定
                  </p>
                </div>

                <div className="p-3 bg-red-50 border border-red-200 rounded">
                  <h5 className="font-medium text-red-700 mb-2">
                    問題 4：內容不足
                  </h5>
                  <p className="text-red-600">內容太少導致沒有滾動條</p>
                  <p className="text-red-600 text-xs mt-1">
                    <strong>解法：</strong> 增加測試資料或縮小容器高度
                  </p>
                </div>
              </div>
            </div>

            <div className="p-3 bg-emerald-50 border border-emerald-200 rounded">
              <h5 className="font-medium text-emerald-700 mb-2">最佳實務建議</h5>
              <ul className="text-emerald-600 space-y-1 list-disc list-inside">
                <li>在聊天介面中確保容器使用 min-h-0 配合 flex 佈局</li>
                <li>必要時強制顯示 ScrollBar 並調整樣式以符合品牌色</li>
                <li>設定 scrollbarGutter: "stable always" 讓滾動條保留空間</li>
                <li>使用水漾月明度假文旅主題色 {ACCENT} 打造一致的視覺體驗</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

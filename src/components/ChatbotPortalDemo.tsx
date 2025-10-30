import { useState } from "react";
import { Button } from "./ui/button";
import { FixedPositionPortal } from "./FixedPositionPortal";
import { MessageCircle, X, Minus, Maximize2, Bot } from "lucide-react";
import { Card, CardContent, CardHeader } from "./ui/card";
import { HOTEL_HEADER_GRADIENT, HOTEL_PRIMARY } from "../styles/hotelTheme";

/**
 * 使用 Portal 呈現的浮動聊天視窗範例
 * 展示如何透過 React Portal 正確定位 fixed 元素
 */
export function ChatbotPortalDemo() {
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState({
    x: window.innerWidth - 400 - 24,
    y: 100,
  });

  return (
    <>
      {!isOpen && (
        <FixedPositionPortal>
          <Button
            onClick={() => setIsOpen(true)}
            className="fixed bottom-6 right-6 w-14 h-14 rounded-full shadow-xl z-[9999] p-0 text-white border-0 transition-all duration-300 hover:scale-110 hover:shadow-2xl"
            size="sm"
            style={{
              backgroundImage: HOTEL_HEADER_GRADIENT,
              backgroundColor: HOTEL_PRIMARY,
            }}
          >
            <div className="absolute inset-0 bg-white/0 hover:bg-white/10 transition-colors duration-200 rounded-full" />
            <MessageCircle className="w-6 h-6 text-white relative z-10" />
          </Button>
        </FixedPositionPortal>
      )}

      {isOpen && (
        <FixedPositionPortal>
          <div
            className="fixed z-[9997] w-96 h-[500px] transition-all duration-300 ease-out"
            style={{
              left: `${position.x}px`,
              top: `${position.y}px`,
            }}
          >
            <Card className="h-full flex flex-col shadow-2xl border-2 bg-background">
              <CardHeader
                className="text-white rounded-t-lg"
                style={{
                  backgroundImage: HOTEL_HEADER_GRADIENT,
                  backgroundColor: HOTEL_PRIMARY,
                }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="p-1 bg-white/20 rounded">
                      <Bot className="w-4 h-4 text-white" />
                    </div>
                    <div>
                      <h3 className="text-lg font-medium text-white">
                        水漾月明度假文旅 · Portal 實驗室
                      </h3>
                      <p className="text-sm text-white/80">
                        看見智能客服的實際擺放效果
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="w-6 h-6 p-0 hover:bg-white/20 transition-colors text-white"
                      onClick={() => setIsOpen(false)}
                    >
                      <Minus className="w-3 h-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="w-6 h-6 p-0 hover:bg-white/20 transition-colors text-white"
                      onClick={() =>
                        setPosition({
                          x: Math.max((window.innerWidth - 384) / 2, 24),
                          y: 48,
                        })
                      }
                    >
                      <Maximize2 className="w-3 h-3" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="w-6 h-6 p-0 hover:bg-white/20 transition-colors text-white"
                      onClick={() => setIsOpen(false)}
                    >
                      <X className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </CardHeader>

              <CardContent className="flex-1 p-4 flex flex-col justify-center items-center space-y-6">
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 rounded-full flex items-center justify-center mx-auto shadow-lg"
                    style={{
                      backgroundImage: HOTEL_HEADER_GRADIENT,
                      backgroundColor: HOTEL_PRIMARY,
                    }}
                  >
                    <Bot className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <h4 className="font-medium mb-2 text-foreground">
                      Portal 讓視窗真正浮在畫面最上層
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      聊天視窗透過 React Portal 渲染到 document.body，
                      讓 fixed 定位不受父層 overflow 或 transform 影響。
                    </p>
                  </div>
                  <div className="p-3 bg-emerald-50 border border-emerald-200 rounded text-sm text-left">
                    <strong className="text-emerald-700">實作重點</strong>
                    <ul className="text-emerald-600 space-y-1 mt-2 list-disc list-inside">
                      <li>確保滾動與定位不受父層 CSS 限制</li>
                      <li>區分開啟按鈕與聊天視窗的 Portal</li>
                      <li>維持一致的品牌色與陰影效果</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </FixedPositionPortal>
      )}
    </>
  );
}

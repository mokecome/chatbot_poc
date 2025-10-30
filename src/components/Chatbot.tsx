import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { ScrollArea } from "./ui/scroll-area";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { Bot, ConciergeBell } from "lucide-react";
import { chatAPI } from "../services/api";
import { HOTEL_HEADER_GRADIENT, HOTEL_PRIMARY } from "../styles/hotelTheme";

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
}

export function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content:
        "您好！歡迎來到水漾月明度假文旅智能客服。我可以協助您查詢客房空房、安排湖畔餐飲與活動報名、預約接駁服務，以及提供館內設施與會員專屬優惠資訊。今天想先了解哪項服務呢？",
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]",
      );
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsTyping(true);

    const aiResponseId = (Date.now() + 1).toString();
    const initialAIResponse: Message = {
      id: aiResponseId,
      content: "",
      isUser: false,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, initialAIResponse]);

    try {
      await chatAPI.sendMessage(
        content,
        (chunk: string) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiResponseId
                ? { ...msg, content: msg.content + chunk }
                : msg,
            ),
          );
        },
        () => {
          setIsTyping(false);
        },
        (error: string) => {
          console.error("Chat API Error:", error);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === aiResponseId
                ? { ...msg, content: `很抱歉，發生錯誤：${error}` }
                : msg,
            ),
          );
          setIsTyping(false);
        },
      );
    } catch (error) {
      console.error("Failed to send message:", error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === aiResponseId
            ? {
                ...msg,
                content: "抱歉，目前無法連線至服務。可否稍後再試一次？",
              }
            : msg,
        ),
      );
      setIsTyping(false);
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto">
      <Card className="flex-1 flex flex-col shadow-xl border-0 overflow-hidden">
        <CardHeader
          className="border-b text-white"
          style={{
            backgroundImage: HOTEL_HEADER_GRADIENT,
            backgroundColor: HOTEL_PRIMARY,
          }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-white/15">
                <ConciergeBell className="w-6 h-6 text-white" />
              </div>
              <div>
                <CardTitle className="flex items-center gap-2 text-white">
                  水漾月明度假文旅智能客服
                  <Badge className="flex items-center gap-1 bg-white/20 text-white border-white/30">
                    <ConciergeBell className="w-3 h-3" />
                    禮賓
                  </Badge>
                </CardTitle>
                <p className="text-sm text-white/80">
                  24 小時旅宿管家 · 訂房、餐飲、會議與會員服務
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
              <span className="text-xs text-white/80">線上服務中</span>
            </div>
          </div>
        </CardHeader>

        <CardContent className="flex-1 p-0 flex flex-col bg-white">
          <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
            <div className="space-y-4">
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message.content}
                  isUser={message.isUser}
                  timestamp={message.timestamp}
                />
              ))}

              {isTyping && (
                <div className="flex gap-3 mb-4">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center bg-[#B38844]">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                  <Card
                    className="p-3 bg-card border"
                    style={{ userSelect: "text", WebkitUserSelect: "text" }}
                  >
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                      <div
                        className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                        style={{ animationDelay: "0.1s" }}
                      />
                      <div
                        className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"
                        style={{ animationDelay: "0.2s" }}
                      />
                    </div>
                  </Card>
                </div>
              )}
            </div>
          </ScrollArea>

          <ChatInput onSendMessage={handleSendMessage} disabled={isTyping} />
        </CardContent>
      </Card>
    </div>
  );
}

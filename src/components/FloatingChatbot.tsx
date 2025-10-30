import { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader } from "./ui/card";
import { Button } from "./ui/button";
import { ChatScrollArea } from "./ChatScrollArea";
import { ChatMessage } from "./ChatMessage";
import { ChatWelcomeMessage } from "./ChatWelcomeMessage";
import { ChatInput } from "./ChatInput";
import { FixedPositionPortal } from "./FixedPositionPortal";
import { Bot, Minus, X, MessageCircle, Maximize2, Minimize2, ChevronDown, ChevronUp } from "lucide-react";
import { chatAPI } from "../services/api";
import { HOTEL_HEADER_GRADIENT, HOTEL_PRIMARY } from "../styles/hotelTheme";

interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  isWelcome?: boolean;
}


export function FloatingChatbot() {
  const [isMinimized, setIsMinimized] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const [isClosed, setIsClosed] = useState(true); // 預設關閉
  const [position, setPosition] = useState({ x: 0, y: 0 }); // 改為 x, y 座標
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const dragAnimationRef = useRef<number | null>(null);
  
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: '',
      isUser: false,
      timestamp: new Date(),
      isWelcome: true
    }
  ]);
  
  const [isTyping, setIsTyping] = useState(false);
  const [showScrollIndicator, setShowScrollIndicator] = useState(false);
  const [showScrollToTop, setShowScrollToTop] = useState(false);
  const chatbotRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<React.ElementRef<typeof ChatScrollArea>>(null);

  const scrollToBottom = () => {
    // Use timeout to ensure DOM has updated
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ 
        behavior: 'smooth',
        block: 'end'
      });
    }, 50);
  };

  const scrollToTop = () => {
    const scrollElement = scrollAreaRef.current?.querySelector('[data-radix-scroll-area-viewport]');
    if (scrollElement) {
      scrollElement.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 改進的滾動監聽邏輯 - 避免按鈕位置飄移
  useEffect(() => {
    const scrollElement = scrollAreaRef.current?.querySelector('[data-radix-scroll-area-viewport]');
    if (!scrollElement) return;

    let scrollTimeout: NodeJS.Timeout;
    
    const handleScroll = () => {
      // 使用 debounce 避免頻繁更新
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        const { scrollTop, scrollHeight, clientHeight } = scrollElement;
        
        // 增加容差值，避免微小滾動導致按鈕閃爍
        const bottomThreshold = 120; // 增加底部容差
        const topThreshold = 120;    // 增加頂部容差
        
        const isNearBottom = scrollHeight - scrollTop - clientHeight < bottomThreshold;
        const isNearTop = scrollTop < topThreshold;
        const hasEnoughMessages = messages.length > 4; // 需要更多訊息才顯示按鈕
        
        // 更穩定的按鈕顯示邏輯 - 確保在適當條件下顯示
        // 當不在底部且有足夠訊息時，顯示「回到最新」按鈕
        setShowScrollIndicator(!isNearBottom && hasEnoughMessages);
        // 當不在頂部且有足夠訊息時，顯示「回到最舊」按鈕
        setShowScrollToTop(!isNearTop && hasEnoughMessages);
      }, 50); // 50ms debounce
    };

    scrollElement.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll(); // Check initial state

    return () => {
      scrollElement.removeEventListener('scroll', handleScroll);
      clearTimeout(scrollTimeout);
    };
  }, [messages.length]);

  // 計算聊天窗口的尺寸和位置
  const getChatbotDimensions = useCallback(() => {
    if (isMaximized) {
      return {
        width: window.innerWidth - 40,
        height: window.innerHeight - 40,
        x: 20,
        y: 20
      };
    } else if (isMinimized) {
      return {
        width: 320,
        height: 48,
        x: position.x || window.innerWidth - 320 - 24,
        y: position.y || window.innerHeight - 48 - 24
      };
    } else {
      return {
        width: 480,
        height: 500,
        x: position.x || window.innerWidth - 480 - 24,
        y: position.y || window.innerHeight - 500 - 24
      };
    }
  }, [isMaximized, isMinimized, position]);

  // 處理視窗大小變化
  useEffect(() => {
    const handleResize = () => {
      const dims = getChatbotDimensions();
      setPosition(prev => ({
        x: Math.max(0, Math.min(prev.x || dims.x, window.innerWidth - dims.width)),
        y: Math.max(0, Math.min(prev.y || dims.y, window.innerHeight - dims.height))
      }));
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [getChatbotDimensions]);

  // 添加錯誤狀態和當前正在流式接收的訊息
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState(false);

  const handleFAQClick = (question: string) => {
    handleSendMessage(question);
  };

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      isUser: true,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);
    setIsStreaming(true);
    setCurrentStreamingMessage('');

    // 創建一個 AI 回應訊息的佔位符
    const aiMessageId = (Date.now() + 1).toString();
    const placeholderMessage: Message = {
      id: aiMessageId,
      content: '',
      isUser: false,
      timestamp: new Date()
    };

    try {
      // 先添加佔位符訊息
      setMessages(prev => [...prev, placeholderMessage]);
      setIsTyping(false);

      let accumulatedContent = '';
      
      await chatAPI.sendMessage(
        content,
        // onChunk - 每次收到新的文字片段
        (chunk: string) => {
          accumulatedContent += chunk;
          setCurrentStreamingMessage(accumulatedContent);
          // 即時更新訊息內容
          setMessages(prev => 
            prev.map(msg => 
              msg.id === aiMessageId 
                ? { ...msg, content: accumulatedContent }
                : msg
            )
          );
        },
        // onComplete - 流式傳輸完成
        (fullMessage: string) => {
          setIsStreaming(false);
          setCurrentStreamingMessage('');
          // 最終更新訊息內容
          setMessages(prev => 
            prev.map(msg => 
              msg.id === aiMessageId 
                ? { ...msg, content: fullMessage }
                : msg
            )
          );
        },
        // onError - 發生錯誤
        (error: string) => {
          console.error('Chat API error:', error);
          setIsStreaming(false);
          setCurrentStreamingMessage('');
          setIsTyping(false);
          
          // 顯示錯誤訊息
          const errorMessage: Message = {
            id: (Date.now() + 2).toString(),
            content: `抱歉，發生了錯誤：${error}\\n\\n請稍後再試，或直接撥打服務專線：03-3322101`,
            isUser: false,
            timestamp: new Date()
          };
          
          // 移除佔位符並添加錯誤訊息
          setMessages(prev => 
            prev.filter(msg => msg.id !== aiMessageId).concat(errorMessage)
          );
        }
      );

    } catch (error) {
      console.error('Unexpected error:', error);
      setIsStreaming(false);
      setCurrentStreamingMessage('');
      setIsTyping(false);
      
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        content: '抱歉，系統發生錯誤，請稍後再試。\\n\\n如需立即協助，請撥打服務專線：03-3322101',
        isUser: false,
        timestamp: new Date()
      };
      
      // 移除佔位符並添加錯誤訊息
      setMessages(prev => 
        prev.filter(msg => msg.id !== aiMessageId).concat(errorMessage)
      );
    }
  };

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (isMaximized) return;
    
    e.preventDefault();
    setIsDragging(true);
    
    const dims = getChatbotDimensions();
    setDragOffset({
      x: e.clientX - (position.x || dims.x),
      y: e.clientY - (position.y || dims.y)
    });
  }, [isMaximized, position, getChatbotDimensions]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging || isMaximized) return;
    
    if (dragAnimationRef.current) {
      cancelAnimationFrame(dragAnimationRef.current);
    }
    
    dragAnimationRef.current = requestAnimationFrame(() => {
      const dims = getChatbotDimensions();
      const newX = e.clientX - dragOffset.x;
      const newY = e.clientY - dragOffset.y;
      
      setPosition({
        x: Math.max(0, Math.min(newX, window.innerWidth - dims.width)),
        y: Math.max(0, Math.min(newY, window.innerHeight - dims.height))
      });
    });
  }, [isDragging, isMaximized, dragOffset, getChatbotDimensions]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    if (dragAnimationRef.current) {
      cancelAnimationFrame(dragAnimationRef.current);
      dragAnimationRef.current = null;
    }
    document.body.style.userSelect = '';
    document.body.style.webkitUserSelect = '';
  }, []);

  useEffect(() => {
    if (isDragging) {
      document.body.style.userSelect = 'none';
      document.body.style.webkitUserSelect = 'none';
      
      document.addEventListener('mousemove', handleMouseMove, { passive: false });
      document.addEventListener('mouseup', handleMouseUp);
      
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
        if (dragAnimationRef.current) {
          cancelAnimationFrame(dragAnimationRef.current);
        }
        document.body.style.userSelect = '';
        document.body.style.webkitUserSelect = '';
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  const toggleMinimize = useCallback(() => {
    setIsMinimized(!isMinimized);
    setIsMaximized(false);
  }, [isMinimized]);

  const toggleMaximize = useCallback(() => {
    setIsMaximized(!isMaximized);
    setIsMinimized(false);
  }, [isMaximized]);

  const handleClose = useCallback(() => {
    setIsClosed(true);
  }, []);

  const handleReopen = useCallback(() => {
    setIsClosed(false);
    setIsMinimized(false);
    // 設定預設位置
    setPosition({
      x: window.innerWidth - 480 - 24,
      y: window.innerHeight - 500 - 24
    });
  }, []);

  // 觸發按鈕 - 使用 Portal 確保真正 fixed 定位
  const renderTriggerButton = () => (
    <FixedPositionPortal>
      <Button
        onClick={handleReopen}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full shadow-xl z-[9999] p-0 text-white border-0 transition-all duration-300 hover:scale-110 hover:shadow-2xl"
        size="sm"
        style={{
          backgroundImage: HOTEL_HEADER_GRADIENT,
          backgroundColor: HOTEL_PRIMARY
        }}
      >
        <div className="absolute inset-0 rounded-full bg-white/0 hover:bg-white/10 transition-colors duration-200"></div>
        <MessageCircle className="w-6 h-6 text-white relative z-10" />
      </Button>
    </FixedPositionPortal>
  );

  if (isClosed) {
    return renderTriggerButton();
  }

  // 主聊天窗口 - 使用 Portal 確保真正 fixed 定位
  const dims = getChatbotDimensions();
  
  // 穩定的滾動按鈕位置計算 - 兩個按鈕都固定在右下方，不會受到 ScrollArea 影響
  const scrollButtonBaseLeft = dims.x + dims.width - 56; // 距離窗口右邊 16px
  const scrollToBottomY = dims.y + dims.height - 128; // 距離底部 128px (88px 輸入框 + 40px 間距)
  const scrollToTopY = scrollToBottomY - 48; // 「回到最舊」按鈕位於「回到最新」按鈕上方 48px (按鈕高度 40px + 間距 8px)
  
  // 確保按鈕位置不會超出視窗邊界
  const safeScrollToTopY = Math.max(10, scrollToTopY);
  const safeScrollToBottomY = Math.max(safeScrollToTopY + 48, scrollToBottomY);

  const renderScrollButtons = () => {
    if (isMinimized) return null;
    
    return (
      <FixedPositionPortal>
        <>
          {/* 回到最舊按鈕 - 固定在右下角上方，不會飄移 */}
          {showScrollToTop && (
            <Button
              onClick={scrollToTop}
            className="fixed w-10 h-10 rounded-full shadow-lg z-[9998] p-0 bg-[#B38844] hover:bg-[#8F6A32] transition-all duration-200 text-white border-0"
              size="sm"
              style={{
                left: `${scrollButtonBaseLeft}px`,
                top: `${safeScrollToTopY}px`
              }}
              title="回到最舊訊息"
            >
              <ChevronUp className="w-4 h-4 text-white" />
            </Button>
          )}

          {/* 回到最新按鈕 - 固定在右下角下方，不會飄移 */}
          {showScrollIndicator && (
            <Button
              onClick={scrollToBottom}
            className="fixed w-10 h-10 rounded-full shadow-lg z-[9998] p-0 bg-[#B38844] hover:bg-[#8F6A32] transition-all duration-200 text-white border-0"
              size="sm"
              style={{
                left: `${scrollButtonBaseLeft}px`,
                top: `${safeScrollToBottomY}px`
              }}
              title="回到最新訊息"
            >
              <ChevronDown className="w-4 h-4 text-white" />
            </Button>
          )}
        </>
      </FixedPositionPortal>
    );
  };

  return (
    <FixedPositionPortal>
      <>
        {/* 滾動按鈕 */}
        {renderScrollButtons()}
        
        {/* 主聊天窗口 */}
        <div
          ref={chatbotRef}
          className={`fixed z-[9997] select-none ${
            isDragging 
              ? 'transition-none' 
              : 'transition-all duration-300 ease-out'
          } ${
            isDragging ? 'cursor-grabbing' : ''
          }`}
          style={{
            left: `${dims.x}px`,
            top: `${dims.y}px`,
            width: `${dims.width}px`,
            height: `${dims.height}px`
          }}
        >
          <Card className="h-full flex flex-col shadow-2xl border-2 bg-background">
            <CardHeader 
              className={`border-b ${isDragging ? 'cursor-grabbing' : 'cursor-grab'} text-white px-3 py-3 rounded-t-lg select-none`}
              style={{
                backgroundImage: HOTEL_HEADER_GRADIENT,
                backgroundColor: HOTEL_PRIMARY
              }}
              onMouseDown={handleMouseDown}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="p-1 bg-white/20 rounded">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-medium text-white">水漾月明度假文旅智能客服</h3>
                    {!isMinimized && (
                      <p className="text-sm text-white/80">
                        24 小時旅宿管家 · 訂房、餐飲、接駁即時回覆
                      </p>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-6 h-6 p-0 hover:bg-white/20 active:bg-white/30 transition-colors text-white"
                    onClick={toggleMinimize}
                  >
                    <Minus className="w-3 h-3" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-6 h-6 p-0 hover:bg-white/20 active:bg-white/30 transition-colors text-white"
                    onClick={toggleMaximize}
                  >
                    {isMaximized ? <Minimize2 className="w-3 h-3" /> : <Maximize2 className="w-3 h-3" />}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-6 h-6 p-0 hover:bg-white/20 active:bg-white/30 transition-colors text-white"
                    onClick={handleClose}
                  >
                    <X className="w-3 h-3" />
                  </Button>
                </div>
              </div>
            </CardHeader>
            
            {!isMinimized && (
              <CardContent className="flex-1 p-0 flex flex-col min-h-0">
                {/* 聊天訊息區域 - 確保有正確的高度約束 */}
                <div className="flex-1 min-h-0 relative">
                  <ChatScrollArea ref={scrollAreaRef} className="absolute inset-0">
                    <div className="space-y-4 min-h-full">
                      {messages.map((message) => (
                        message.isWelcome ? (
                          <ChatWelcomeMessage
                            key={message.id}
                            timestamp={message.timestamp}
                            onQuestionClick={handleFAQClick}
                          />
                        ) : (
                          <ChatMessage
                            key={message.id}
                            message={message.content}
                            isUser={message.isUser}
                            timestamp={message.timestamp}
                          />
                        )
                      ))}
                      
                      {(isTyping || isStreaming) && (
                        <div className="flex gap-3 mb-4">
                          <div className="w-8 h-8 bg-[#B38844] rounded-full flex items-center justify-center text-white">
                            <Bot className="w-4 h-4 text-white" />
                          </div>
                          <Card className="p-3 bg-card border">
                            {isStreaming ? (
                              <div className="text-sm text-muted-foreground">正在輸入回覆...</div>
                            ) : (
                              <div className="flex gap-1">
                                <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"></div>
                                <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                                <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                              </div>
                            )}
                          </Card>
                        </div>
                      )}
                      <div ref={messagesEndRef} className="h-1" />
                    </div>
                  </ChatScrollArea>
                </div>
                
                {/* 輸入區域 - 固定在底部 */}
                <div className="flex-shrink-0 border-t border-border/30 bg-background">
                  <ChatInput onSendMessage={handleSendMessage} disabled={isTyping || isStreaming} />
                </div>
              </CardContent>
            )}
          </Card>
        </div>
      </>
    </FixedPositionPortal>
  );
}

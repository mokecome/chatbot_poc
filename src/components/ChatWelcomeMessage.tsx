import { Avatar, AvatarFallback } from "./ui/avatar";
import { Card } from "./ui/card";
import { Bot } from "lucide-react";
import { ChatFAQSection } from "./ChatFAQSection";
import { HOTEL_PRIMARY, HOTEL_PRIMARY_SOFT } from "../styles/hotelTheme";

interface ChatWelcomeMessageProps {
  timestamp: Date;
  onQuestionClick: (question: string) => void;
}

export function ChatWelcomeMessage({ timestamp, onQuestionClick }: ChatWelcomeMessageProps) {
  return (
    <div className="flex gap-3 mb-4 justify-start">
      <Avatar className="w-8 h-8 mt-1">
        <AvatarFallback className="text-white" style={{ backgroundColor: HOTEL_PRIMARY }}>
          <Bot className="w-4 h-4" />
        </AvatarFallback>
      </Avatar>
      
      <div className="max-w-[70%]">
        <Card 
          className="p-3 bg-card border"
          style={{ backgroundColor: HOTEL_PRIMARY_SOFT }}
        >
          <p className="whitespace-pre-wrap mb-4 select-text cursor-text">
            您好！歡迎來到水漾月明度假文旅智能客服。我可以協助您查詢客房空房、安排湖畔餐飲與活動報名、預約接駁服務以及了解館內設施。今天想先從哪一項服務開始呢？
          </p>
          <ChatFAQSection onQuestionClick={onQuestionClick} />
        </Card>
        <p className="text-xs text-muted-foreground mt-1 text-left select-text cursor-text">
          {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </p>
      </div>
    </div>
  );
}

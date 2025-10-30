import { Avatar, AvatarFallback } from "./ui/avatar";
import { Card } from "./ui/card";
import { Bot, User } from "lucide-react";
import { MarkdownText } from "./MarkdownText";
import { HOTEL_PRIMARY } from "../styles/hotelTheme";

interface ChatMessageProps {
  message: string;
  isUser: boolean;
  timestamp: Date;
}

export function ChatMessage({ message, isUser, timestamp }: ChatMessageProps) {
  return (
    <div className={`flex gap-3 mb-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <Avatar className="w-8 h-8 mt-1">
          <AvatarFallback className="text-white" style={{ backgroundColor: HOTEL_PRIMARY }}>
            <Bot className="w-4 h-4" />
          </AvatarFallback>
        </Avatar>
      )}
      
      <div className={`max-w-[80%] ${isUser ? 'order-first' : ''}`} style={{ userSelect: 'text', WebkitUserSelect: 'text', WebkitTouchCallout: 'default' }}>
        <Card className={`p-3 ${
          isUser
            ? 'text-white ml-auto border-0'
            : 'bg-card border'
        }`}
        style={{
          ...(isUser ? { backgroundColor: HOTEL_PRIMARY } : {}),
          userSelect: 'text',
          WebkitUserSelect: 'text',
          wordBreak: 'break-word',
          overflowWrap: 'break-word'
        }}>
          <MarkdownText 
            className="whitespace-pre-wrap select-text cursor-text" 
            style={{ userSelect: 'text', WebkitUserSelect: 'text' }}
          >
            {message}
          </MarkdownText>
        </Card>
        <p className={`text-xs text-muted-foreground mt-1 select-text cursor-text ${
          isUser ? 'text-right' : 'text-left'
        }`}
        style={{ userSelect: 'text', WebkitUserSelect: 'text' }}>
          {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </p>
      </div>

      {isUser && (
        <Avatar className="w-8 h-8 mt-1">
          <AvatarFallback className="bg-secondary text-secondary-foreground">
            <User className="w-4 h-4" />
          </AvatarFallback>
        </Avatar>
      )}
    </div>
  );
}

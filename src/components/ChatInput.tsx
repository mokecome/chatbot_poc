import { useState, useRef, useEffect } from "react";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Send } from "lucide-react";
import { HOTEL_PRIMARY } from "../styles/hotelTheme";

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSendMessage, disabled = false }: ChatInputProps) {
  const [message, setMessage] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage("");
      // Refocus input after sending message
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 p-4">
      <Input
        ref={inputRef}
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="輸入訊息，讓我們安排您的住宿需求"
        disabled={disabled}
        className="flex-1 border-border/50 focus-visible:border-[#B38844] focus-visible:ring-[#B38844]/20 h-10"
        autoComplete="off"
      />
      <Button
        type="submit"
        disabled={!message.trim() || disabled}
        className="px-3 h-10 text-white border-0 bg-[#B38844] hover:bg-[#8F6A32] transition-colors disabled:opacity-50 shadow-md"
        style={{
          backgroundColor: HOTEL_PRIMARY
        }}
      >
        <Send className="w-4 h-4" />
      </Button>
    </form>
  );
}

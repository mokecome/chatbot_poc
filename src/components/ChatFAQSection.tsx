import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { HOTEL_PRIMARY, HOTEL_PRIMARY_SOFT } from "../styles/hotelTheme";

interface ChatFAQSectionProps {
  onQuestionClick: (question: string) => void;
}

export function ChatFAQSection({ onQuestionClick }: ChatFAQSectionProps) {
  const faqButtons = [
    {
      id: "room-availability",
      label: "查詢房型與價格",
      question: "我想了解豪華雙人房這週末是否還有空房？費用是多少？"
    },
    {
      id: "dining-reservation",
      label: "預約餐廳或酒吧",
      question: "可以幫我預約明天晚上七點的水漾月明湖畔餐廳兩人晚餐嗎？"
    },
    {
      id: "late-checkout",
      label: "延遲退房與接駁",
      question: "請問如何申請延遲退房？飯店是否提供機場接送服務？"
    }
  ];

  return (
    <Card
      className="p-4 border"
      style={{ backgroundColor: HOTEL_PRIMARY_SOFT, borderColor: `${HOTEL_PRIMARY}33` }}
    >
      <h4 className="text-[16px] font-medium mb-3 text-foreground select-text cursor-text">
        熱門服務快速選單
      </h4>
      <div className="flex flex-col gap-2">
        {faqButtons.map((button) => (
          <Button
            key={button.id}
            variant="outline"
            size="sm"
            className="w-full text-[13px] h-9 px-3 bg-white border border-[#E6D8C4] text-[#4A3A2A] hover:bg-[#B38844]/10 hover:border-[#B38844] hover:text-[#B38844] transition-colors text-center justify-center shadow-sm"
            onClick={() => onQuestionClick(button.question)}
          >
            <span className="select-text cursor-text">{button.label}</span>
          </Button>
        ))}
      </div>
    </Card>
  );
}

import React from 'react';

interface MarkdownTextProps {
  children: string;
  className?: string;
  style?: React.CSSProperties;
}

export function MarkdownText({ children, className, style }: MarkdownTextProps) {
  const parseMarkdown = (text: string): React.ReactNode => {
    // URL 正則表達式 - 匹配 http://, https://, www. 開頭的URL
    const urlRegex = /(https?:\/\/[^\s]+|www\.[^\s]+)/g;

    // 粗體語法正則表達式
    const boldRegex = /\*\*(.*?)\*\*/g;

    // 先處理粗體，將其轉換為特殊標記
    const processedText = text.replace(boldRegex, (match, p1) => `<BOLD>${p1}</BOLD>`);

    // 解析URL和文本
    const parseTextWithUrls = (str: string): React.ReactNode[] => {
      const parts = str.split(urlRegex);
      const elements: React.ReactNode[] = [];
      let urlIndex = 0;

      parts.forEach((part, index) => {
        if (!part) return;

        // 檢查是否為URL
        if (part.match(urlRegex)) {
          const href = part.startsWith('www.') ? `https://${part}` : part;
          elements.push(
            <a
              key={`url-${index}-${urlIndex++}`}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:text-blue-700 underline cursor-pointer"
              style={{ wordBreak: 'break-all', overflowWrap: 'break-word' }}
              onClick={(e) => e.stopPropagation()}
            >
              {part}
            </a>
          );
        } else {
          // 處理包含粗體標記的文本
          const boldParts = part.split(/(<BOLD>.*?<\/BOLD>)/g);
          boldParts.forEach((boldPart, boldIndex) => {
            if (!boldPart) return;

            const boldMatch = boldPart.match(/<BOLD>(.*?)<\/BOLD>/);
            if (boldMatch) {
              elements.push(
                <strong key={`bold-${index}-${boldIndex}`}>
                  {boldMatch[1]}
                </strong>
              );
            } else {
              elements.push(
                <span key={`text-${index}-${boldIndex}`}>
                  {boldPart}
                </span>
              );
            }
          });
        }
      });

      return elements;
    };

    return <>{parseTextWithUrls(processedText)}</>;
  };

  return (
    <div
      className={className}
      style={{
        ...style,
        wordBreak: 'break-word',
        overflowWrap: 'break-word'
      }}
    >
      {parseMarkdown(children)}
    </div>
  );
}
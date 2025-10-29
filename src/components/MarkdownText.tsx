import React from 'react';

interface MarkdownTextProps {
  children: string;
  className?: string;
  style?: React.CSSProperties;
}

export function MarkdownText({ children, className, style }: MarkdownTextProps) {
  const parseMarkdown = (text: string): React.ReactNode => {
    // 處理 **粗體** 語法
    const boldRegex = /\*\*(.*?)\*\*/g;
    const parts = text.split(boldRegex);
    
    const elements: React.ReactNode[] = [];
    
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 0) {
        // 普通文字
        if (parts[i]) {
          elements.push(<span key={i}>{parts[i]}</span>);
        }
      } else {
        // 粗體文字
        elements.push(<strong key={i}>{parts[i]}</strong>);
      }
    }
    
    return <>{elements}</>;
  };

  return (
    <div className={className} style={style}>
      {parseMarkdown(children)}
    </div>
  );
}
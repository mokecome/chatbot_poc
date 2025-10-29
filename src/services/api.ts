export interface ChatMessage {
  message: string;
  session_id?: string;
}

export interface StreamResponse {
  type: "text" | "end" | "error";
  content: string;
  session_id: string;
}

const DEFAULT_BASE_URL = (() => {
  const fromEnv = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();
  if (!fromEnv) {
    return "";
  }
  return fromEnv.replace(/\/+$/, "");
})();

export class ChatAPI {
  private baseURL: string;
  private sessionId: string | null = null;

  constructor(baseURL: string = DEFAULT_BASE_URL) {
    this.baseURL = baseURL;
  }

  private resolveURL(path: string): string {
    if (!this.baseURL) {
      return path;
    }
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;
    return `${this.baseURL}${normalizedPath}`;
  }

  async sendMessage(
    message: string,
    onChunk?: (chunk: string) => void,
    onComplete?: (fullMessage: string) => void,
    onError?: (error: string) => void
  ): Promise<string> {
    const payload: ChatMessage = {
      message,
      session_id: this.sessionId || undefined,
    };

    try {
      const response = await fetch(this.resolveURL("/api/chat"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText || ""}`.trim());
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("無法讀取伺服器回應 (Missing readable stream)");
      }

      const decoder = new TextDecoder();
      let fullMessage = "";
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line.startsWith("data:")) {
            continue;
          }

          try {
            const data = JSON.parse(line.slice(5).trim()) as StreamResponse;

            if (data.session_id && !this.sessionId) {
              this.sessionId = data.session_id;
            }

            if (data.type === "text") {
              fullMessage += data.content;
              onChunk?.(data.content);
            } else if (data.type === "end") {
              onComplete?.(fullMessage);
              return fullMessage;
            } else if (data.type === "error") {
              onError?.(data.content);
              throw new Error(data.content);
            }
          } catch (parseError) {
            console.warn("Failed to parse SSE payload:", line, parseError);
          }
        }
      }

      return fullMessage;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "未知錯誤，請稍後再試";
      onError?.(errorMessage);
      throw error instanceof Error ? error : new Error(String(error));
    }
  }

  getSessionId(): string | null {
    return this.sessionId;
  }

  clearSession(): void {
    this.sessionId = null;
  }
}

export const chatAPI = new ChatAPI();

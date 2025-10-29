# Chatbot POC

這是一個基於 Vite + React 的對話式介面原型，提供浮動聊天機器人與滾動條測試工具。  
主要功能：

- 浮動聊天機器人可與後端 API 溝通，支援視窗拖曳、最小化與最大化。
- 具備滾動條測試面板，方便檢查不同滾動行為與樣式。
- 前端頁面整合 Tailwind 風格的行銷版型。

## 開發指令

- `npm install`：安裝依賴。
- `npm run dev`：啟動開發伺服器。
- `npm run build`：產出 `dist/` 供部署。

## 部署到 Vercel

1. 在 Vercel 建立專案並連結此 Git 儲存庫。
2. Build Command 使用 `npm run build`，Output Directory 設為 `dist`。
3. 在專案 **Settings → Environment Variables** 新增：
   - `OPENAI_API_KEY`：OpenAI 服務金鑰（必填，不然後端會回傳友善的備援訊息）。
   - `POSTGRES_URL`（或 `POSTGRES_URL_NON_POOLING`）：啟用 Vercel Postgres 後會自動提供，後端會優先使用這些連線字串。
   - 其他自訂變數（如 `FRONTEND_ORIGIN`、`SYSTEM_PROMPT`）可依需求加入。
4. 後端已透過 `api/index.py` 以 Flask 形式部署為 Vercel Serverless Function，所有 `/api/*` 路徑直接在同網域下存取；請先在 Vercel 儲存區啟用 Postgres 並完成資料庫遷移（可複製原有 SQLite 資料）。
5. 部署完成後即可取得預覽與正式網址；確認功能正常後可一鍵 Promote。

> **注意**：`vercel.json` 已設定將 `/api/*` 轉送至 `api/index.py`，並指定 Python 3.11 運行環境，確保 Flask 伺服器在 Vercel 上正常處理請求。

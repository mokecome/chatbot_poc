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
3. 如需 API 金鑰，請在 Vercel 專案設定中新增環境變數。
4. 部署完成後即可取得預覽與正式網址。

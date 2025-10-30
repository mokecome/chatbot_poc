"""Flask application that manages surveys backed by SQLite."""

from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template_string,
    request,
    stream_with_context,
    send_from_directory,
)
from flask_cors import CORS
from openai import OpenAI
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_storage_base() -> str:
    """
    Determine where to place writable artifacts (SQLite, uploads).

    When running on Vercel or other serverless providers, the project directory
    is read-only and we must fall back to a tmp filesystem.
    """
    if os.getenv("VERCEL") or os.getenv("VERCEL_ENV"):
        return os.getenv("TMPDIR") or os.getenv("TEMP") or "/tmp"
    return BASE_DIR


STORAGE_BASE = _default_storage_base()


def _resolve_database_url() -> str:
    """
    Determine which database connection string to use.

    Priority:
    1. Vercel managed Postgres environment variables.
    2. Local SQLite file (development fallback only).
    """
    vercel_pg_candidates = [
        os.getenv("POSTGRES_URL"),
        os.getenv("POSTGRES_PRISMA_URL"),
        os.getenv("POSTGRES_URL_NON_POOLING"),
    ]
    for candidate in vercel_pg_candidates:
        if candidate:
            return candidate

    default_sqlite = os.getenv("SQLITE_PATH") or os.path.join(STORAGE_BASE, "app.sqlite3")
    return f"sqlite:///{default_sqlite}"


DATABASE_URL = _resolve_database_url()

_ENGINE_KWARGS: Dict[str, Any] = {"future": True, "pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _ENGINE_KWARGS["connect_args"] = {"check_same_thread": False}

engine: Engine = create_engine(DATABASE_URL, **_ENGINE_KWARGS)


if engine.url.get_backend_name() == "sqlite":

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")


ASSET_ROUTE_PREFIX = os.getenv("ASSET_ROUTE_PREFIX", "/uploads")
ASSET_LOCAL_DIR = os.getenv("ASSET_LOCAL_DIR") or os.path.join(STORAGE_BASE, "uploads")
os.makedirs(ASSET_LOCAL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path=ASSET_ROUTE_PREFIX, static_folder=ASSET_LOCAL_DIR)

CORS(
    app,
    resources={r"/api/*": {"origins": os.getenv("FRONTEND_ORIGIN", "*")}},
    supports_credentials=False,
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_openai_client() -> Optional[OpenAI]:
    if not OPENAI_API_KEY:
        return None
    return OpenAI(api_key=OPENAI_API_KEY)


SYSTEM_PROMPT = (
    os.getenv("SYSTEM_PROMPT")
    or (
        '''你是一位親切、專業且自然的客服聊天機器人，服務於「水漾月明度假文旅（Hana Mizu Tsuki Hotel）」。

你擁有一份詳細的 Markdown 資料（以下稱為「飯店說明文件」），內容包含：
1. 房型與價格
2. 交通與聯絡資訊
3. 優惠方案（如水上腳踏車住房專案）
4. 設施介紹
5. 訂房連結
6. 環保政策
7. 周邊景點介紹

---

### 🎯 你的任務：

1️⃣ **回答基本問題**  
- 若使用者詢問旅遊、住宿、交通、價格、設施、政策、優惠或周邊景點等問題，請根據「飯店說明文件」中的資訊直接回答。  
- 若問題與住宿無關，先友善回應，再自然轉回飯店話題，例如：  
  「這個問題很有趣呢～順帶一提，您這次是打算來苗栗旅遊嗎？我可以幫您介紹一下我們的房型！」  

2️⃣ **推薦訂房**  
- 根據使用者條件（人數、日期、預算、是否想看湖景、是否親子入住等）主動推薦合適房型。  
- 每次推薦需包含：
  - 房型名稱  
  - 價格 / 晚  
  - 房內設施或特色  
  - 若有對應優惠專案（如水上腳踏車方案），請一起說明  
  - 相關網頁連結  
- 若條件不足，請主動詢問，例如：「請問您預計幾位入住？需要湖景房或浴缸嗎？」

3️⃣ **行動引導**  
- 回答結尾請附上引導句，例如：  
  「👉 要我幫您看看哪一天還有空房嗎？」  
  或  
  「👉 您可直接點這裡訂房：https://res.windsurfercrs.com/ibe/index.aspx?propertyID=17658&nono=1&lang=zh-tw&adults=2」  

---

### 💡 回覆風格：
- 如果出現url,用超連接方式呈現(可讓使用者點擊)
- 溫暖、自然、有生活感，像真人客服。
- 適度加入體驗感描述，例如：「很多客人都說晚上泡澡看湖景超放鬆！」  
- 盡量用繁體中文回答。

---

### ⚙️ 回答邏輯範例：

**使用者問：**「你們有幾種房型？」  
**回答：**  
目前共有八種房型可選：  
- 豪華雙人房 — $12,000／晚，日式軟墊與浴缸 👉 [查看詳情](http://www.younglake.com.tw/Home/ProductsDetail/3)  
- 湖景雙人房 — $14,000／晚，一大床或兩小床，側湖景  
- …（可依需要列出3～4項）  
若您想體驗湖上活動，我建議考慮【水上腳踏車住房專案】，平日雙人房只要 $3,980 起。  
👉 要我幫您查一下這週末的空房嗎？

---

### 🧾 使用說明：
在初始化模型時：
- 將此提示詞放在 `system` 或 `instruction` 層級。  
- 將整份 Markdown 文檔（# 水漾月明度假文旅 開頭的內容）放在 `user` 層級。  
- 模型即可依照文件回答與推薦。'''
    )
)
USER_PROMPT = os.getenv("USER_PROMPT") or '''# 水漾月明度假文旅（Hana Mizu Tsuki Hotel）信息总览

## (一) 客房資訊

| 房型 | 價格 / 晚 | 床型與設施 | 網頁連結 |
|------|-------------|-------------|-----------|
| 豪華雙人房（床型若需指定請來電洽詢） | $12,000元 | 日式軟墊・浴缸 | [連結](http://www.younglake.com.tw/Home/ProductsDetail/3) |
| 湖景雙人房（側湖景） | $14,000元 | 一大床・兩小床 | [連結](http://www.younglake.com.tw/Home/ProductsDetail/5) |
| 豪華三人房 | $15,000元 | 一大一小床・浴缸 | [連結](http://www.younglake.com.tw/Home/ProductsDetail/6) |
| 湖景四人房（床型若需指定請來電洽詢） | $22,000元 | 兩大床・浴缸 | [連結](http://www.younglake.com.tw/Home/ProductsDetail/7) |
| 豪華四人房（床型若需指定請來電洽詢） | $18,000元 | 兩大床・浴缸 | [連結](http://www.younglake.com.tw/Home/ProductsDetail/9) |
| 家庭四人房 | $25,000元 | 兩大床・客廳・浴缸 | [連結](http://www.younglake.com.tw/Home/ProductsDetail/8) |
| 蜜月雙人房 | $13,000元 | 一大床・客廳・浴缸 | [連結](http://www.younglake.com.tw/Home/ProductsDetail/2) |
| 水漾套房（正湖景） | $20,000元 | 一大床・浴缸 | [連結](http://www.younglake.com.tw/Home/ProductsDetail/1) |

---

## (二) 交通與聯絡資訊

- **飯店名稱**：水漾月明度假文旅（Hana Mizu Tsuki Hotel）
- **地址**：362苗栗縣頭屋鄉明德路54號
- **Google 地圖**：[前往地圖](https://www.google.com/maps?ll=24.585596,120.887298&z=17&t=m&hl=zh-TW&gl=US&mapclient=embed&cid=709365327370099103)
- **電話**：037-255-358
- **Email**：mizutsukihotel@gmail.com

---

## (三) 優惠方案 — 水上腳踏車住房專案

- **合作單位**：水漾月明 × 海棠島水域遊憩中心
- **活動日期**：114/8/28 ~ 114/10/30

### ❤️ 專案內容
**湖上同樂暢快大冒險 🏄 一泊一食（含早餐）**

#### 🍀 平日價格
| 房型 | 價格 |
|------|------|
| 豪華雙人房 | 3980元 |
| 湖景雙人房 | 4980元 |
| 豪華三人房 | 5300元 |
| 豪華四人房 | 6380元 |

#### 🍀 週六價格
| 房型 | 價格 |
|------|------|
| 豪華雙人房 | 4880元 |
| 湖景雙人房 | 7280元 |
| 豪華三人房 | 6280元 |
| 豪華四人房 | 7380元 |

---

### 🎉 專案贈送
1. 早餐（依房型人數贈送）
2. 水上自行車兌換券（半小時）－價值350元 / 張
3. 7歲以下孩童不佔床不收費（早餐需另收費）
4. 小孩身高須滿120公分以上方可自行騎乘水上自行車

---

### 水上自行車兌換券注意事項
1. 贈送數量依房型人數（雙人房2張 / 三人房3張 / 四人房4張）
2. 需於入住日一個月內使用完畢，逾期或遺失不予補發
3. 需於海棠島現場兌換並遵守設施安全規範
4. 小孩身高須滿120公分以上方可自行騎乘
5. 票券使用須先致電海棠島預約（非教練陪同券，若需教練陪同需加價）
6. 加購海棠島水域遊憩中心 Span Outdoor 相關活動（SUP / 獨木舟 / 水上自行車）可享 **9折優惠**

---

### ♥️ 暑假優惠再加碼
專案可加購 **水漾環湖電動自行車**
- $250元 / 台 / 2.5小時（再贈飲料一瓶）
- 騎乘至海棠島僅需約15分鐘

---

## (四) 設施介紹

| 設施 | 網頁連結 | 備註 |
|------|-----------|------|
| 環湖電動自行車 | [連結](http://www.younglake.com.tw/Home/FacilityDetail/14) | 可租借 |
| 渡假會議 | [連結](http://www.younglake.com.tw/Home/FacilityDetail/4) | 適合商務與活動 |
| 汗蒸幕體驗 | [連結](http://www.younglake.com.tw/Home/FacilityDetail/11) | 放鬆身心 |
| 西餐廳 | [連結](http://www.younglake.com.tw/Home/FacilityDetail/7) | 中式桌菜・客家風味・歐式百匯<br>預約專線：037-255358 |
| 視聽室 | [連結](http://www.younglake.com.tw/Home/FacilityDetail/6) | 影音娛樂空間 |
| 水漾小賽車手俱樂部 | [連結](http://www.younglake.com.tw/Home/FacilityDetail/10) | 兒童遊樂設施 |
| 24SHOP 智能販賣機 | [連結](http://www.younglake.com.tw/Home/FacilityDetail/8) | 無人販售服務 |
| 清潔服務機器人 | [連結](http://www.younglake.com.tw/Home/FacilityDetail/12) | 智能清潔體驗 |

---

## (五) 訂房資訊
[立即訂房 ➜](https://res.windsurfercrs.com/ibe/index.aspx?propertyID=17658&nono=1&lang=zh-tw&adults=2)

---

## (六) 環保政策 — 一次性備品提供
自 **2025年1月1日** 起，客房將不再提供一次性備品。  
建議旅客自行攜帶個人盥洗用品，如有需求可洽櫃檯。  
造成不便之處，敬請見諒。

---

## (七) 周邊景點介紹

### 湖畔與水上活動
- **日新島**：全台唯一位於水庫中的島嶼，可步行或騎自行車前往。
- **海棠島水域遊憩中心**：距離飯店約9分鐘車程，提供 SUP、獨木舟、水上自行車等活動。
- **明德水庫環湖**：沿途可欣賞湖光山色，部分路段設有自行車道。

### 森林與花園
- **橙香森林**：擁有玻璃屋與橙香隧道，適合親子休閒。
- **雅聞玫瑰園**：以玫瑰花為主題的休閒農場。
- **葛瑞絲香草田**：距離飯店約2分鐘車程，可欣賞各式香草植物。

### 其他推薦
- **皇家高爾夫球場**：適合高爾夫愛好者。
- **魯冰花休閒農莊**：提供餐飲與湖畔休閒空間。
- **卓也小屋**：可體驗藍染、在地料理與綠色旅遊活動。'''


def utcnow() -> datetime.datetime:
    return datetime.datetime.utcnow()


def ensure_schema() -> None:
    """Create the SQLite schema if it does not exist yet."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id TEXT UNIQUE,
                    display_name TEXT,
                    avatar_url TEXT,
                    gender TEXT,
                    birthday TEXT,
                    email TEXT,
                    phone TEXT,
                    source TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    last_interaction_at TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS surveys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS survey_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    survey_id INTEGER NOT NULL REFERENCES surveys(id) ON DELETE CASCADE,
                    question_type TEXT NOT NULL,
                    question_text TEXT NOT NULL,
                    description TEXT,
                    font_size INTEGER,
                    options_json TEXT,
                    is_required INTEGER DEFAULT 0,
                    display_order INTEGER DEFAULT 0,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS survey_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    survey_id INTEGER NOT NULL REFERENCES surveys(id) ON DELETE CASCADE,
                    member_id INTEGER REFERENCES members(id) ON DELETE SET NULL,
                    external_id TEXT,
                    answers_json TEXT NOT NULL,
                    is_completed INTEGER DEFAULT 1,
                    completed_at TIMESTAMP,
                    source TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
                ON chat_messages(session_id, created_at)
                """
            )
        )


ensure_schema()


def ensure_chat_session(session_id: Optional[str] = None) -> str:
    """Return an existing chat session id or create a new one."""
    chat_session_id = session_id or uuid.uuid4().hex
    now = utcnow()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO chat_sessions (id, created_at, updated_at)
                VALUES (:id, :now, :now)
                ON CONFLICT(id) DO UPDATE SET updated_at = :now
                """
            ),
            {"id": chat_session_id, "now": now},
        )
    return chat_session_id


def save_chat_message(session_id: str, role: str, content: str) -> None:
    """Persist a chat message for a given session."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO chat_messages (session_id, role, content, created_at)
                VALUES (:sid, :role, :content, :created_at)
                """
            ),
            {
                "sid": session_id,
                "role": role,
                "content": content,
                "created_at": utcnow(),
            },
        )
        conn.execute(
            text(
                """
                UPDATE chat_sessions
                SET updated_at = :updated_at
                WHERE id = :sid
                """
            ),
            {"sid": session_id, "updated_at": utcnow()},
        )


def fetch_chat_history(session_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    """Fetch the most recent chat history for the session in chronological order."""
    if limit <= 0:
        limit = 1

    query = text(
        f"""
        SELECT role, content
        FROM chat_messages
        WHERE session_id = :sid
        ORDER BY created_at DESC
        LIMIT {limit}
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(query, {"sid": session_id}).mappings().all()

    # Reverse to chronological order
    return [dict(row) for row in reversed(rows)]


def build_assistant_messages(history: Iterable[Dict[str, str]], user_content: str) -> List[Dict[str, str]]:
    """Prepare the message payload for the OpenAI Responses API."""
    messages: List[Dict[str, str]] = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    if USER_PROMPT:
        messages.append({"role": "user", "content": USER_PROMPT})
    for item in history:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})
    return messages


def to_responses_input(messages: Iterable[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Convert chat messages to the structure expected by the OpenAI Responses API."""
    structured: List[Dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content_type = "input_text"
        if role == "assistant":
            content_type = "output_text"
        structured.append(
            {
                "role": role,
                "content": [{"type": content_type, "text": message["content"]}],
            }
        )
    return structured


def format_sse(payload: Dict[str, Any]) -> str:
    """Serialize a Python dictionary into a Server-Sent Events data frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
@app.post("/chat")
def api_chat():
    if not request.is_json:
        return jsonify({"error": "Payload must be JSON."}), 400

    payload = request.get_json(force=True) or {}
    message = (payload.get("message") or "").strip()
    requested_session = payload.get("session_id")

    if not message:
        return jsonify({"error": "message is required"}), 400

    session_id = ensure_chat_session(
        requested_session if isinstance(requested_session, str) else None
    )
    history = fetch_chat_history(session_id)

    # Persist the user's message before streaming.
    save_chat_message(session_id, "user", message)

    client = get_openai_client()

    def generate():
        logger.info("Streaming response for session %s", session_id)
        yield format_sse({"type": "session", "content": "", "session_id": session_id})

        if client is None:
            assistant_text = (
                "Unable to reach the OpenAI service right now. Please check the server logs.\n\n"
                f"Message waiting to send: {message}\n"
                "Confirm that OPENAI_API_KEY is configured before retrying."
            )
            save_chat_message(session_id, "assistant", assistant_text)
            yield format_sse(
                {"type": "text", "content": assistant_text, "session_id": session_id}
            )
            yield format_sse({"type": "end", "content": "", "session_id": session_id})
            return

        accumulated: List[str] = []
        try:
            structured_messages = build_assistant_messages(history, message)
            with client.responses.stream(
                model=OPENAI_MODEL,
                input=to_responses_input(structured_messages),
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        delta = event.delta or ""
                        if not delta:
                            continue
                        accumulated.append(delta)
                        yield format_sse(
                            {
                                "type": "text",
                                "content": delta,
                                "session_id": session_id,
                            }
                        )
                    elif event.type == "response.error":
                        error_message = getattr(event, "message", "") or "AI model error."
                        raise RuntimeError(error_message)

                final_response = stream.get_final_response()

        except Exception:
            logger.exception("Chat streaming failed for session %s", session_id)
            error_message = (
                "We hit an unexpected issue while generating the concierge reply."
                " Please try again shortly or reach our staff directly."
            )
            yield format_sse(
                {"type": "error", "content": error_message, "session_id": session_id}
            )
            return

        full_text = "".join(accumulated).strip()

        if not full_text:
            try:
                outputs = getattr(final_response, "output", None) or []
                for item in outputs:
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            full_text = content.get("text", "").strip()
                            if full_text:
                                break
                    if full_text:
                        break
            except Exception:
                full_text = ""

        if full_text:
            save_chat_message(session_id, "assistant", full_text)
        else:
            fallback_text = ("I'm sorry, I couldn't craft a reply just now. Please rephrase or contact our concierge team.")
            save_chat_message(session_id, "assistant", fallback_text)
            yield format_sse(
                {"type": "text", "content": fallback_text, "session_id": session_id}
            )

        yield format_sse({"type": "end", "content": "", "session_id": session_id})

    response = Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
    )
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


def fetchall(sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    with engine.begin() as conn:
        return [
            dict(row)
            for row in conn.execute(text(sql), params or {}).mappings().all()
        ]


def fetchone(sql: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        result = conn.execute(text(sql), params or {}).mappings().first()
        return dict(result) if result else None


def execute(sql: str, params: Optional[Dict[str, Any]] = None) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    trimmed = str(value).strip()
    return trimmed or None


def upsert_member(
    external_id: Optional[str],
    display_name: Optional[str] = None,
    avatar_url: Optional[str] = None,
    gender: Optional[str] = None,
    birthday: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    source: Optional[str] = "form",
) -> Optional[int]:
    """Create or update a member record identified by an external id."""
    external_id = _clean(external_id)
    if not external_id:
        return None

    now = utcnow()
    data = {
        "display_name": _clean(display_name),
        "avatar_url": _clean(avatar_url),
        "gender": _clean(gender),
        "birthday": _clean(birthday),
        "email": _clean(email),
        "phone": _clean(phone),
        "source": source or "form",
        "updated_at": now,
        "last_interaction_at": now,
    }

    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM members WHERE external_id = :ext"),
            {"ext": external_id},
        ).scalar()
        if existing:
            conn.execute(
                text(
                    """
                    UPDATE members
                       SET display_name=:display_name,
                           avatar_url=:avatar_url,
                           gender=:gender,
                           birthday=:birthday,
                           email=:email,
                           phone=:phone,
                           source=:source,
                           updated_at=:updated_at,
                           last_interaction_at=:last_interaction_at
                     WHERE id=:member_id
                    """
                ),
                {**data, "member_id": existing},
            )
            return int(existing)

        insert_params = {
            "external_id": external_id,
            **data,
            "created_at": now,
        }
        result = conn.execute(
            text(
                """
                INSERT INTO members (
                    external_id,
                    display_name,
                    avatar_url,
                    gender,
                    birthday,
                    email,
                    phone,
                    source,
                    created_at,
                    updated_at,
                    last_interaction_at
                ) VALUES (
                    :external_id,
                    :display_name,
                    :avatar_url,
                    :gender,
                    :birthday,
                    :email,
                    :phone,
                    :source,
                    :created_at,
                    :updated_at,
                    :last_interaction_at
                )
                """
            ),
            insert_params,
        )
        member_id = result.lastrowid
    return int(member_id) if member_id is not None else None


QUESTION_TYPE_ALIASES: Dict[str, List[str]] = {
    "TEXT": ["TEXT", "INPUT", "SHORT_TEXT"],
    "TEXTAREA": ["TEXTAREA", "LONG_TEXT", "PARAGRAPH"],
    "SINGLE_CHOICE": ["SINGLE_CHOICE", "SINGLE", "RADIO", "CHOICE_SINGLE"],
    "MULTI_CHOICE": ["MULTI_CHOICE", "MULTI", "CHECKBOX", "CHOICE_MULTI", "MULTIPLE"],
    "SELECT": ["SELECT", "DROPDOWN", "PULLDOWN"],
    "NAME": ["NAME"],
    "PHONE": ["PHONE", "TEL", "MOBILE"],
    "EMAIL": ["EMAIL"],
    "BIRTHDAY": ["BIRTHDAY", "DOB", "DATE_OF_BIRTH", "DATE"],
    "ADDRESS": ["ADDRESS"],
    "GENDER": ["GENDER", "SEX"],
    "IMAGE": ["IMAGE", "PHOTO"],
    "VIDEO": ["VIDEO"],
    "ID_NUMBER": ["ID_NUMBER", "IDENTIFICATION"],
    "LINK": ["LINK", "URL"],
}

DEFAULT_QUESTION_TYPE = "TEXT"


def normalize_question_type(raw: Any) -> str:
    token = _clean(str(raw) if raw is not None else None)
    if not token:
        return DEFAULT_QUESTION_TYPE
    token = token.replace("-", "_").upper()
    for canonical, aliases in QUESTION_TYPE_ALIASES.items():
        if token == canonical or token in aliases:
            return canonical
    for canonical, aliases in QUESTION_TYPE_ALIASES.items():
        if any(alias in token for alias in aliases):
            return canonical
    return DEFAULT_QUESTION_TYPE


def register_survey_from_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a survey described by JSON payload."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a mapping")

    name = _clean(payload.get("name")) or "Survey"
    description = _clean(payload.get("description"))
    category = _clean(payload.get("category"))
    questions = payload.get("questions") or []
    if not isinstance(questions, list):
        raise ValueError("questions must be a list")

    now = utcnow()
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO surveys (name, description, category, is_active, created_at, updated_at)
                VALUES (:name, :description, :category, 1, :now, :now)
                """
            ),
            {"name": name, "description": description, "category": category, "now": now},
        )
        survey_id = int(result.lastrowid)

        for idx, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                continue
            q_type = normalize_question_type(question.get("question_type"))
            options = question.get("options") or question.get("options_json") or []
            if not isinstance(options, list):
                options = []
            entry = {
                "survey_id": survey_id,
                "question_type": q_type,
                "question_text": _clean(question.get("question_text")) or f"Question {idx}",
                "description": _clean(question.get("description")),
                "font_size": question.get("font_size") if isinstance(question.get("font_size"), int) else None,
                "options_json": json.dumps(options, ensure_ascii=False),
                "is_required": 1 if question.get("is_required") else 0,
                "display_order": question.get("order") if isinstance(question.get("order"), int) else idx,
                "created_at": now,
                "updated_at": now,
            }
            conn.execute(
                text(
                    """
                    INSERT INTO survey_questions (
                        survey_id,
                        question_type,
                        question_text,
                        description,
                        font_size,
                        options_json,
                        is_required,
                        display_order,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        :survey_id,
                        :question_type,
                        :question_text,
                        :description,
                        :font_size,
                        :options_json,
                        :is_required,
                        :display_order,
                        :created_at,
                        :updated_at
                    )
                    """
                ),
                entry,
            )

    logger.info("Survey %s created with %s questions", survey_id, len(questions))
    return {"survey_id": survey_id, "question_count": len(questions)}


def load_survey_meta(survey_id: int) -> Dict[str, Any]:
    survey = fetchone(
        "SELECT id, name, description FROM surveys WHERE id = :sid", {"sid": survey_id}
    )
    if not survey:
        raise ValueError(f"survey {survey_id} not found")

    rows = fetchall(
        """
        SELECT id,
               question_type,
               question_text,
               description,
               font_size,
               options_json,
               is_required,
               display_order
          FROM survey_questions
         WHERE survey_id = :sid
         ORDER BY display_order ASC, id ASC
        """,
        {"sid": survey_id},
    )

    questions: List[Dict[str, Any]] = []
    for row in rows:
        options: List[Any]
        try:
            options = json.loads(row.get("options_json") or "[]")
        except json.JSONDecodeError:
            options = []
        questions.append(
            {
                "id": row["id"],
                "question_type": row["question_type"],
                "question_text": row["question_text"],
                "description": row.get("description"),
                "font_size": row.get("font_size"),
                "options": options,
                "is_required": bool(row.get("is_required")),
                "display_order": row.get("display_order"),
            }
        )

    return {
        "id": survey["id"],
        "name": survey["name"],
        "description": survey.get("description") or "",
        "questions": questions,
    }


def save_survey_submission(
    survey_id: int,
    answers: Dict[str, Any],
    participant: Optional[Dict[str, Any]] = None,
) -> None:
    """Store a survey response."""
    if not fetchone("SELECT 1 FROM surveys WHERE id=:sid", {"sid": survey_id}):
        raise ValueError("survey not found")
    if not isinstance(answers, dict):
        raise ValueError("answers must be a mapping")

    normalized: Dict[str, Any] = {}
    for key, value in answers.items():
        if not isinstance(key, str) or not key.startswith("q_"):
            continue
        suffix = key.split("_", 1)[1] if "_" in key else key
        if isinstance(value, list):
            normalized[suffix] = value
        elif value is None:
            normalized[suffix] = ""
        else:
            normalized[suffix] = str(value)

    participant = participant or {}
    external_id = (
        participant.get("external_id")
        or participant.get("id")
        or participant.get("identifier")
    )
    display_name = participant.get("display_name") or participant.get("name")
    email = participant.get("email")
    phone = participant.get("phone")

    member_id = upsert_member(
        external_id,
        display_name=display_name,
        email=email,
        phone=phone,
        source="form",
    )

    now = utcnow()
    execute(
        """
        INSERT INTO survey_responses (
            survey_id,
            member_id,
            external_id,
            answers_json,
            is_completed,
            completed_at,
            source,
            ip_address,
            user_agent,
            created_at,
            updated_at
        )
        VALUES (
            :survey_id,
            :member_id,
            :external_id,
            :answers_json,
            1,
            :completed_at,
            :source,
            :ip_address,
            :user_agent,
            :created_at,
            :updated_at
        )
        """,
        {
            "survey_id": survey_id,
            "member_id": member_id,
            "external_id": _clean(external_id),
            "answers_json": json.dumps(normalized, ensure_ascii=False),
            "completed_at": now,
            "source": "form",
            "ip_address": request.headers.get("X-Forwarded-For", request.remote_addr),
            "user_agent": request.headers.get("User-Agent"),
            "created_at": now,
            "updated_at": now,
        },
    )


SURVEY_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ survey.name or "Survey" }}</title>
  <style>
    body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0; background: #f6f7fb; color: #111827; }
    .wrap { max-width: 720px; margin: 0 auto; padding: 32px 16px; }
    .card { background: #ffffff; border-radius: 16px; box-shadow: 0 20px 40px rgba(15, 23, 42, 0.12); padding: 28px; }
    h1 { margin: 0 0 16px; font-size: 24px; }
    .desc { margin: 0 0 24px; color: #475569; font-size: 15px; }
    .participant { border: 1px dashed #cbd5f5; border-radius: 12px; padding: 16px; margin-bottom: 24px; background: #f8fafc; }
    .participant label { display: block; font-weight: 500; margin-bottom: 12px; }
    .participant input { width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid #d1d9e6; font-size: 15px; margin-top: 6px; }
    .question { margin-bottom: 22px; }
    .prompt { display: block; font-weight: 600; margin-bottom: 8px; }
    .required { color: #dc2626; margin-left: 4px; }
    .description { font-size: 14px; color: #64748b; margin-bottom: 8px; }
    input[type="text"], input[type="tel"], input[type="email"], input[type="date"], input[type="url"], textarea, select {
      width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid #d1d9e6; font-size: 15px; box-sizing: border-box;
    }
    textarea { min-height: 96px; resize: vertical; }
    .options { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip { display: flex; align-items: center; gap: 6px; padding: 8px 12px; border: 1px solid #cbd5f5; border-radius: 999px; background: #f8fafc; cursor: pointer; }
    .chip input { margin: 0; }
    button { width: 100%; padding: 14px 16px; border: none; border-radius: 12px; background: #2563eb; color: #ffffff; font-size: 16px; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: 0.7; cursor: wait; }
    .hint { margin-top: 16px; font-size: 13px; color: #64748b; }
    .status { margin-top: 18px; font-size: 15px; }
    .status.error { color: #b91c1c; }
    .status.success { color: #047857; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>{{ survey.name or "Survey" }}</h1>
      {% if survey.description %}
      <p class="desc">{{ survey.description }}</p>
      {% endif %}
      <form id="surveyForm">
        <input type="hidden" name="sid" value="{{ survey_id }}">
        <div class="participant">
          <label>
            Contact (optional)
            <input type="text" name="participant_id" placeholder="Email or phone">
          </label>
          <label>
            Name (optional)
            <input type="text" name="participant_name" placeholder="Your name">
          </label>
        </div>
        {% for q in survey.questions %}
        {% set qtype = (q.question_type or "").lower() %}
        {% set field_name = "q_" ~ q.id %}
        <div class="question" data-type="{{ qtype }}" data-id="{{ q.id }}"{% if q.is_required %} data-required="1"{% endif %}>
          <label class="prompt">{{ q.question_text or ("Question " ~ loop.index) }}{% if q.is_required %}<span class="required">*</span>{% endif %}</label>
          {% if q.description %}<div class="description">{{ q.description }}</div>{% endif %}
          {% if qtype in ["text", "name", "address", "phone", "email", "birthday", "id_number", "link"] %}
            {% set input_type = {
              "text": "text",
              "name": "text",
              "address": "text",
              "phone": "tel",
              "email": "email",
              "birthday": "date",
              "id_number": "text",
              "link": "url"
            }[qtype] if qtype in ["text","name","address","phone","email","birthday","id_number","link"] else "text" %}
            <input type="{{ input_type }}" name="{{ field_name }}"{% if q.is_required %} required{% endif %}>
          {% elif qtype == "textarea" %}
            <textarea name="{{ field_name }}"{% if q.is_required %} required{% endif %}></textarea>
          {% elif qtype in ["single_choice", "gender"] %}
            <div class="options">
              {% for opt in q.options %}
                {% set value = opt.value if opt.value is not none else (opt.label if opt.label is not none else "option_" ~ loop.index) %}
                {% set label = opt.label if opt.label is not none else (opt.value if opt.value is not none else "Option " ~ loop.index) %}
                <label class="chip">
                  <input type="radio" name="{{ field_name }}" value="{{ value }}"{% if q.is_required and loop.first %} required{% endif %}>
                  {{ label }}
                </label>
              {% endfor %}
              {% if not q.options %}
              <div>No options configured.</div>
              {% endif %}
            </div>
          {% elif qtype == "multi_choice" %}
            <div class="options">
              {% for opt in q.options %}
                {% set value = opt.value if opt.value is not none else (opt.label if opt.label is not none else "option_" ~ loop.index) %}
                {% set label = opt.label if opt.label is not none else (opt.value if opt.value is not none else "Option " ~ loop.index) %}
                <label class="chip">
                  <input type="checkbox" name="{{ field_name }}" value="{{ value }}"{% if q.is_required and loop.first %} required{% endif %}>
                  {{ label }}
                </label>
              {% endfor %}
              {% if not q.options %}
              <div>No options configured.</div>
              {% endif %}
            </div>
          {% elif qtype == "select" %}
            <select name="{{ field_name }}"{% if q.is_required %} required{% endif %}>
              <option value="">Select??/option>
              {% for opt in q.options %}
                {% set value = opt.value if opt.value is not none else (opt.label if opt.label is not none else "option_" ~ loop.index) %}
                {% set label = opt.label if opt.label is not none else (opt.value if opt.value is not none else "Option " ~ loop.index) %}
                <option value="{{ value }}">{{ label }}</option>
              {% endfor %}
            </select>
          {% else %}
            <input type="text" name="{{ field_name }}"{% if q.is_required %} required{% endif %}>
          {% endif %}
        </div>
        {% endfor %}
        <button type="submit" id="submitBtn">Submit</button>
        <p class="hint">We only use the information to support your request.</p>
      </form>
      <div id="formMessage" class="status" hidden></div>
    </div>
  </div>
  <script>
    const form = document.getElementById("surveyForm");
    const messageEl = document.getElementById("formMessage");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      messageEl.hidden = true;
      const sidField = form.querySelector("input[name='sid']");
      const sid = sidField ? Number(sidField.value) : NaN;
      if (!sid) {
        messageEl.textContent = "Invalid survey id.";
        messageEl.className = "status error";
        messageEl.hidden = false;
        return;
      }

      const sections = Array.from(form.querySelectorAll(".question"));
      const data = {};
      let missingRequired = false;

      sections.forEach((section) => {
        const type = (section.getAttribute("data-type") || "").toLowerCase();
        const id = section.getAttribute("data-id");
        const name = "q_" + id;
        const required = section.hasAttribute("data-required");

        if (type === "multi_choice") {
          const values = Array.from(
            section.querySelectorAll("input[type='checkbox'][name='" + name + "']:checked")
          ).map((el) => el.value);
          if (required && values.length === 0) {
            missingRequired = true;
          }
          data[name] = values;
        } else if (type === "single_choice" || type === "gender") {
          const chosen = section.querySelector("input[type='radio'][name='" + name + "']:checked");
          if (required && !chosen) {
            missingRequired = true;
          }
          data[name] = chosen ? chosen.value : "";
        } else if (type === "select") {
          const selectEl = section.querySelector("select[name='" + name + "']");
          const value = selectEl ? selectEl.value : "";
          if (required && !value) {
            missingRequired = true;
          }
          data[name] = value;
        } else {
          const field = section.querySelector("[name='" + name + "']");
          const value = field ? field.value : "";
          if (required && !value) {
            missingRequired = true;
          }
          data[name] = value;
        }
      });

      if (missingRequired) {
        messageEl.textContent = "Please complete the required fields.";
        messageEl.className = "status error";
        messageEl.hidden = false;
        return;
      }

      const participant = {
        external_id: (form.querySelector("input[name='participant_id']").value || "").trim(),
        display_name: (form.querySelector("input[name='participant_name']").value || "").trim()
      };
      if (!participant.external_id) {
        delete participant.external_id;
      }
      if (!participant.display_name) {
        delete participant.display_name;
      }

      const payload = { sid, data, participant };

      try {
        form.querySelector("#submitBtn").disabled = true;
        const response = await fetch("/__survey_submit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const result = await response.json().catch(() => ({ ok: false, error: "Unexpected response." }));
        if (result.ok) {
          messageEl.textContent = "Thank you! Your response has been recorded.";
          messageEl.className = "status success";
          form.reset();
        } else {
          messageEl.textContent = result.error || "Unable to submit the survey.";
          messageEl.className = "status error";
        }
      } catch (err) {
        console.error("Submit error:", err);
        messageEl.textContent = "An unexpected error occurred.";
        messageEl.className = "status error";
      } finally {
        form.querySelector("#submitBtn").disabled = false;
        messageEl.hidden = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/")
def health() -> tuple[str, int]:
    return "OK", 200


@app.get(f"{ASSET_ROUTE_PREFIX}/<path:filename>")
def serve_uploads(filename: str):
    return send_from_directory(ASSET_LOCAL_DIR, filename, conditional=True)


@app.get("/survey/form")
def survey_form():
    sid = request.args.get("sid", type=int)
    if not sid:
        abort(400, "missing sid")
    try:
        meta = load_survey_meta(sid)
    except ValueError:
        abort(404, "survey not found")
    return render_template_string(SURVEY_TEMPLATE, survey=meta, survey_id=sid)


@app.get("/__survey_load")
def survey_load():
    sid = request.args.get("sid", type=int)
    if not sid:
        return jsonify({"error": "missing sid"}), 400
    try:
        data = load_survey_meta(sid)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(data)


@app.post("/__survey_submit")
def survey_submit():
    if request.is_json:
        payload = request.get_json(force=True) or {}
        sid = payload.get("sid") or payload.get("survey_id")
        answers = payload.get("data") or payload.get("answers") or {}
        participant = payload.get("participant") or {}
    else:
        data = request.form.to_dict(flat=False)
        sid = data.get("sid", [None])[0]
        answers = {
            key: (values if len(values) > 1 else values[0])
            for key, values in data.items()
            if key.startswith("q_")
        }
        participant = {
            "external_id": (data.get("participant_id", [""])[0] or "").strip(),
            "display_name": (data.get("participant_name", [""])[0] or "").strip(),
        }

    try:
        sid_int = int(sid)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid sid"}), 400

    try:
        save_survey_submission(sid_int, answers, participant)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8300"))
    debug_mode = os.getenv("FLASK_DEBUG", "0") in {"1", "true", "True"}
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

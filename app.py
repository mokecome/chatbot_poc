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
    1. Explicit DATABASE_URL (allows overriding everything).
    2. Vercel managed Postgres environment variables.
    3. Local SQLite file (development fallback only).
    """
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit

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


SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """You are the virtual concierge for Stellar International Hotel. Provide information only about this property and follow these principles:
    1. Cover room types, rates, add-on packages, check-in/out, dining & bar venues, events, facilities, transportation, and loyalty benefits.
    2. Politely decline topics unrelated to the hotel (e.g., other attractions or politics) and suggest calling +886-2-1234-5678 or emailing concierge@stellarhotel.tw for further help.
    3. Answer in Traditional Chinese with a warm, professional tone. Use lists or step-by-step guidance when it improves clarity and highlight 24/7 concierge support.
    4. When details are uncertain, invite guests to confirm with on-duty staff to ensure accuracy.
    5. Protect privacy: request only contact details required for bookings or service follow-up.
    Maintain these guidelines so every guest receives an exceptional stay experience."""
)


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
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
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

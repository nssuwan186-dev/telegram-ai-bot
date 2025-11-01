"""Hotel OS Telegram bot with AI assistant, payment slip verification, and expense tracking (Webhook version)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Dict, Optional, Sequence

import google.generativeai as genai
import pytesseract
import uvicorn
from fastapi import FastAPI, Request, Response
from PIL import Image
from telegram import KeyboardButton, ReplyKeyboardMarkup, Update, WebAppInfo
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# --------------------------------------------------------------------------
# FastAPI App Initialization
# --------------------------------------------------------------------------
app = FastAPI()

# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Dataclasses and settings
# --------------------------------------------------------------------------
@dataclass
class Settings:
    telegram_bot_token: str
    authorized_user_id: int
    gemini_api_key: str
    gemini_model: str
    pms_blueprint_path: Path
    database_path: Path
    slip_storage_dir: Path
    webapp_base_url: str
    webapp_static_dir: Path
    webapp_host: str
    webapp_port: int
    serve_webapp: bool
    ocr_language: str
    tesseract_cmd: Optional[str]
    amount_tolerance: Decimal
    webhook_url: Optional[str]


@dataclass
class SlipExtractionResult:
    raw_text: str
    amount: Optional[Decimal]
    payment_date: Optional[date]
    payment_time: Optional[time]


# --------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------
def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


def _load_blueprint(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"PMS blueprint file not found at {path}. Set PMS_BLUEPRINT_PATH to a valid file."
        ) from exc
    if not content:
        raise RuntimeError(
            "PMS blueprint file is empty. Ensure the document contains the blueprint content.",
        )
    return content


def load_settings() -> Settings:
    telegram_bot_token = _require_env("TELEGRAM_BOT_TOKEN")
    gemini_api_key = _require_env("GEMINI_API_KEY")

    authorized_user_raw = _require_env("TELEGRAM_USER_ID")
    try:
        authorized_user_id = int(authorized_user_raw)
    except ValueError as exc:
        raise RuntimeError("TELEGRAM_USER_ID must be an integer") from exc

    gemini_model = os.getenv("GEMINI_MODEL", "gemini-pro")

    blueprint_path_str = os.getenv("PMS_BLUEPRINT_PATH")
    if blueprint_path_str:
        blueprint_path = Path(blueprint_path_str).expanduser().resolve()
    else:
        blueprint_path = Path(__file__).resolve().parent / "resources" / "pms_blueprint.txt"

    database_path = Path(os.getenv("HOTEL_DB_PATH", "./data/hotel_os_bot.db")).expanduser().resolve()
    slip_storage_dir = (
        Path(os.getenv("SLIP_STORAGE_DIR", "./data/slips")).expanduser().resolve()
    )

    webapp_static_dir = (
        Path(os.getenv("WEBAPP_STATIC_DIR", "./webapp")).expanduser().resolve()
    )
    webapp_base_url = os.getenv(
        "WEBAPP_BASE_URL",
        "http://localhost:8080/index.html",
    )
    webapp_host = os.getenv("WEBAPP_HOST", "0.0.0.0")
    webapp_port = int(os.getenv("WEBAPP_PORT", "8080"))
    serve_webapp = os.getenv("SERVE_WEBAPP", "true").lower() == "true"

    ocr_language = os.getenv("OCR_LANGUAGE", "eng,tha")
    tesseract_cmd = os.getenv("TESSERACT_CMD")

    tolerance_raw = os.getenv("AMOUNT_TOLERANCE", "1.00")
    try:
        amount_tolerance = Decimal(tolerance_raw)
    except InvalidOperation as exc:
        raise RuntimeError("AMOUNT_TOLERANCE must be a decimal number") from exc

    webhook_url = os.getenv("WEBHOOK_URL")

    return Settings(
        telegram_bot_token=telegram_bot_token,
        authorized_user_id=authorized_user_id,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        pms_blueprint_path=blueprint_path,
        database_path=database_path,
        slip_storage_dir=slip_storage_dir,
        webapp_base_url=webapp_base_url,
        webapp_static_dir=webapp_static_dir,
        webapp_host=webapp_host,
        webapp_port=webapp_port,
        serve_webapp=serve_webapp,
        ocr_language=ocr_language,
        tesseract_cmd=tesseract_cmd,
        amount_tolerance=amount_tolerance,
        webhook_url=webhook_url,
    )


def configure_ocr(settings: Settings) -> None:
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

def initialize_directories(settings: Settings) -> None:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    settings.slip_storage_dir.mkdir(parents=True, exist_ok=True)
    settings.webapp_static_dir.mkdir(parents=True, exist_ok=True)

def initialize_database(settings: Settings) -> None:
    def _init() -> None:
        with sqlite3.connect(settings.database_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bookings (
                    id TEXT PRIMARY KEY,
                    guest_name TEXT,
                    total_due REAL,
                    currency TEXT,
                    status TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS payment_slips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    booking_id TEXT,
                    reference_code TEXT,
                    slip_image_path TEXT,
                    extracted_amount REAL,
                    extracted_date TEXT,
                    extracted_time TEXT,
                    status TEXT,
                    ocr_text TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount REAL NOT NULL,
                    category TEXT NOT NULL,
                    note TEXT,
                    created_by TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    _init()

def start_static_server(settings: Settings) -> Optional[Thread]:
    if not settings.serve_webapp:
        logger.info("Skipping internal web app server start (SERVE_WEBAPP=false)")
        return None

    if not settings.webapp_static_dir.exists():
        logger.warning("Web app directory %s does not exist", settings.webapp_static_dir)

    handler = partial(SimpleHTTPRequestHandler, directory=str(settings.webapp_static_dir))
    server = ThreadingHTTPServer((settings.webapp_host, settings.webapp_port), handler)

    def _serve() -> None:
        logger.info(
            "Serving Telegram Web App from %s at http://%s:%s/",
            settings.webapp_static_dir,
            settings.webapp_host,
            settings.webapp_port,
        )
        try:
            server.serve_forever()
        except Exception:
            logger.exception("Web app server encountered an error")

    thread = Thread(target=_serve, name="webapp-server", daemon=True)
    thread.start()
    return thread


# --------------------------------------------------------------------------
# Authorization helpers
# --------------------------------------------------------------------------
def _is_authorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    if user is None:
        return False
    authorized_user_id = context.application.bot_data.get("authorized_user_id")
    return authorized_user_id is not None and user.id == authorized_user_id


async def _notify_unauthorized(update: Update) -> None:
    if update.message is not None:
        await update.message.reply_text("คุณไม่มีสิทธิ์ใช้งานบอทนี้ค่ะ")


# --------------------------------------------------------------------------
# Command handlers
# --------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update, context):
        await _notify_unauthorized(update)
        return

    settings: Settings = context.application.bot_data["settings"]
    reply_markup = ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(
                    text="🛎️ เปิดระบบบริหารโรงแรม",
                    web_app=WebAppInfo(url=settings.webapp_base_url),
                )
            ]
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )

    if update.message:
        await update.message.reply_text(
            "ยินดีต้อนรับสู่ hotel_os_bot ระบบบริหารจัดการโรงแรมแบบครบวงจรครับ",
            reply_markup=reply_markup,
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update, context):
        await _notify_unauthorized(update)
        return

    help_text = (
        "คำสั่งที่พร้อมใช้งาน:\n"
        "• /start - แสดงปุ่มเปิดเว็บแอป\n"
        "• /menu - เปิดเว็บแอประบบบริหาร\n"
        "• /pms_blueprint - ส่งพิมพ์เขียวระบบ PMS\n"
        "• ส่งข้อความอื่นเพื่อคุยกับผู้ช่วย AI"
    )
    if update.message:
        await update.message.reply_text(help_text)


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update, context):
        await _notify_unauthorized(update)
        return

    settings: Settings = context.application.bot_data["settings"]
    if update.message:
        await update.message.reply_text(
            "เปิดเมนูหลักได้เลยครับ",
            reply_markup=ReplyKeyboardMarkup(
                [[ 
                    KeyboardButton(
                        text="🛎️ เปิดระบบบริหารโรงแรม",
                        web_app=WebAppInfo(url=settings.webapp_base_url),
                    )
                ]],
                resize_keyboard=True,
            ),
        )


async def ls_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update, context):
        await _notify_unauthorized(update)
        return

    if update.message is None:
        return

    try:
        entries = sorted(path.name for path in Path.cwd().iterdir())
    except OSError as exc:
        logger.exception("Failed to list directory contents")
        await update.message.reply_text(f"เกิดข้อผิดพลาดในการอ่านโฟลเดอร์: {exc}")
        return

    listing = "\n".join(entries) if entries else "(ไม่มีไฟล์ในโฟลเดอร์)"
    await update.message.reply_text(listing)


async def pms_blueprint(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update, context):
        await _notify_unauthorized(update)
        return

    if update.message is None:
        return

    static_content: Optional[Dict[str, str]] = context.application.bot_data.get("static_documents")
    if not static_content or "pms_blueprint" not in static_content:
        await update.message.reply_text(
            "ขออภัย ไม่พบพิมพ์เขียวในระบบ กรุณาติดต่อผู้ดูแลระบบ",
        )
        return

    for chunk in _chunk_for_telegram(static_content["pms_blueprint"]):
        await update.message.reply_text(chunk)


# --------------------------------------------------------------------------
# AI conversation fallback
# --------------------------------------------------------------------------
async def handle_ai_conversation(text: str, model: genai.GenerativeModel) -> str:
    try:
        response = await model.generate_content_async(text)
        return response.text or "ผมยังไม่แน่ใจว่าจะตอบอย่างไร ลองพิมพ์ใหม่อีกครั้งนะครับ"
    except Exception as exc:  # noqa: BLE001
        logger.error("Error calling Gemini API: %s", exc)
        return (
            "ตอนนี้ระบบ AI มีปัญหาในการประมวลผลครับ กรุณาลองใหม่อีกครั้งภายหลัง"
        )


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update, context):
        await _notify_unauthorized(update)
        return

    if update.message is None or update.message.text is None:
        return

    model: Optional[genai.GenerativeModel] = context.application.bot_data.get("gemini_model")
    if model is None:
        logger.error("Gemini model is not initialized")
        await update.message.reply_text("โมเดล AI ยังไม่พร้อมใช้งานครับ กรุณาลองใหม่อีกครั้ง")
        return

    ai_response = await handle_ai_conversation(update.message.text, model)
    await update.message.reply_text(ai_response)


# --------------------------------------------------------------------------
# Web App data handling
# --------------------------------------------------------------------------
async def web_app_data_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update, context):
        await _notify_unauthorized(update)
        return

    if update.message is None or update.message.web_app_data is None:
        return

    try:
        payload = json.loads(update.message.web_app_data.data)
    except json.JSONDecodeError:
        await update.message.reply_text("ไม่สามารถอ่านข้อมูลที่ส่งมาจาก Web App ได้ครับ")
        return

    payload_type = payload.get("type")
    if payload_type == "payment_slip":
        await process_payment_slip(update, context, payload)
    elif payload_type == "expense_entry":
        await process_expense_entry(update, context, payload)
    else:
        await update.message.reply_text("ยังไม่รองรับคำสั่งจาก Web App ประเภทนี้ครับ")


async def process_payment_slip(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    payload: Dict[str, str],
) -> None:
    settings: Settings = context.application.bot_data["settings"]

    booking_id = payload.get("booking_id")
    reference = payload.get("reference") or ""
    file_id = payload.get("file_id")

    if not booking_id or not file_id:
        await update.message.reply_text("ข้อมูลไม่ครบถ้วน กรุณาลองใหม่อีกครั้ง")
        return

    destination = settings.slip_storage_dir / f"{booking_id}_{int(datetime.utcnow().timestamp())}.jpg"

    try:
        telegram_file = await context.bot.get_file(file_id)
        await telegram_file.download_to_drive(custom_path=str(destination))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to download slip image")
        await update.message.reply_text(f"ไม่สามารถดาวน์โหลดสลิปได้: {exc}")
        return

    try:
        extraction = await asyncio.to_thread(
            extract_slip_information,
            destination,
            settings,
        )
    except RuntimeError as exc:
        logger.exception("OCR processing failed")
        await update.message.reply_text(str(exc))
        await _record_payment_slip(
            settings,
            booking_id=booking_id,
            reference=reference,
            slip_path=str(destination),
            amount=None,
            status="ocr_failed",
            raw_text=None,
            extracted_date=None,
            extracted_time=None,
        )
        return

    verification = await asyncio.to_thread(
        verify_booking_amount,
        settings,
        booking_id,
        extraction.amount,
    )

    status = verification["status"]

    await _record_payment_slip(
        settings,
        booking_id=booking_id,
        reference=reference,
        slip_path=str(destination),
        amount=extraction.amount,
        status=status,
        raw_text=extraction.raw_text,
        extracted_date=(extraction.payment_date.isoformat() if extraction.payment_date else None),
        extracted_time=(extraction.payment_time.isoformat() if extraction.payment_time else None),
    )

    if status == "verified":
        await update.message.reply_text(
            "ตรวจสอบสลิปเรียบร้อย ✅ ยอดชำระตรงกับข้อมูลการจอง",
        )
    elif status == "booking_not_found":
        await update.message.reply_text(
            "ไม่พบหมายเลขการจองที่ระบุ กรุณาตรวจสอบอีกครั้ง",
        )
    elif status == "amount_mismatch":
        expected = verification.get("expected_amount")
        extracted_amount = extraction.amount
        await update.message.reply_text(
            "ยอดชำระไม่ตรงกับข้อมูลในระบบ\n"
            f"ยอดที่ระบบคาดหวัง: {expected}\n"
            f"ยอดที่ตรวจพบ: {extracted_amount}"
        )
    elif status == "amount_missing":
        await update.message.reply_text(
            "ไม่สามารถอ่านยอดชำระจากสลิปได้ กรุณาตรวจสอบความชัดของรูปภาพ",
        )
    else:
        await update.message.reply_text("ไม่สามารถตรวจสอบสลิปได้ กรุณาลองอีกครั้งภายหลัง")


async def _record_payment_slip(
    settings: Settings,
    booking_id: str,
    reference: str,
    slip_path: str,
    amount: Optional[Decimal],
    status: str,
    raw_text: Optional[str],
    extracted_date: Optional[str],
    extracted_time: Optional[str],
) -> None:
    def _insert() -> None:
        with sqlite3.connect(settings.database_path) as conn:
            conn.execute(
                """
                INSERT INTO payment_slips (
                    booking_id,
                    reference_code,
                    slip_image_path,
                    extracted_amount,
                    extracted_date,
                    extracted_time,
                    status,
                    ocr_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    booking_id,
                    reference,
                    slip_path,
                    float(amount) if amount is not None else None,
                    extracted_date,
                    extracted_time,
                    status,
                    raw_text,
                ),
            )

    await asyncio.to_thread(_insert)


async def process_expense_entry(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    payload: Dict[str, str],
) -> None:
    settings: Settings = context.application.bot_data["settings"]
    amount_raw = payload.get("amount")
    category = payload.get("category") or "อื่นๆ"
    note = payload.get("note") or ""

    if not amount_raw:
        await update.message.reply_text("กรุณาระบุจำนวนเงินก่อนบันทึกค่าใช้จ่ายครับ")
        return

    try:
        amount = Decimal(str(amount_raw))
    except InvalidOperation:
        await update.message.reply_text("รูปแบบจำนวนเงินไม่ถูกต้อง กรุณากรอกใหม่")
        return

    user_id = update.effective_user.id if update.effective_user else None

    def _insert() -> None:
        with sqlite3.connect(settings.database_path) as conn:
            conn.execute(
                """
                INSERT INTO expenses (amount, category, note, created_by)
                VALUES (?, ?, ?, ?)
                """,
                (float(amount), category, note, str(user_id) if user_id else None),
            )

    await asyncio.to_thread(_insert)

    await update.message.reply_text(
        "บันทึกค่าใช้จ่ายเรียบร้อย ✅",
    )


# --------------------------------------------------------------------------
# OCR and verification helpers
# --------------------------------------------------------------------------
def extract_slip_information(path: Path, settings: Settings) -> SlipExtractionResult:
    try:
        image = Image.open(path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"ไม่สามารถเปิดรูปภาพสำหรับ OCR ได้: {exc}") from exc

    try:
        raw_text = pytesseract.image_to_string(image, lang=settings.ocr_language)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "ไม่พบโปรแกรม Tesseract OCR กรุณาติดตั้งและตั้งค่า TESSERACT_CMD ให้ถูกต้อง",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"เกิดข้อผิดพลาดขณะประมวลผล OCR: {exc}") from exc

    amount = _extract_amount(raw_text)
    payment_date = _extract_date(raw_text)
    payment_time = _extract_time(raw_text)

    return SlipExtractionResult(
        raw_text=raw_text,
        amount=amount,
        payment_date=payment_date,
        payment_time=payment_time,
    )


def _extract_amount(text: str) -> Optional[Decimal]:
    amount_pattern = re.compile(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+[.,]\d{2})")
    candidates = amount_pattern.findall(text)
    for value in candidates[::-1]:  # iterate from last match assuming amount near end
        normalized = value.replace(",", "").replace(" ", "")
        if normalized.count(".") > 1:
            normalized = normalized.replace(".", "", normalized.count(".") - 1)
        try:
            return Decimal(normalized)
        except InvalidOperation:
            continue
    return None

def _extract_date(text: str) -> Optional[date]:
    date_pattern = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")
    match = date_pattern.search(text)
    if not match:
        return None
    candidate = match.group(1)
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m/%y"):
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    return None

def _extract_time(text: str) -> Optional[time]:
    time_pattern = re.compile(r"(\d{1,2}[:.]\d{2}(?::\d{2})?)")
    match = time_pattern.search(text)
    if not match:
        return None
    candidate = match.group(1).replace(".", ":")
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(candidate, fmt).time()
        except ValueError:
            continue
    return None

def verify_booking_amount(
    settings: Settings,
    booking_id: str,
    extracted_amount: Optional[Decimal],
) -> Dict[str, Optional[Decimal]]:
    result: Dict[str, Optional[Decimal]] = {"status": "pending"}

    with sqlite3.connect(settings.database_path) as conn:
        cursor = conn.execute(
            "SELECT total_due FROM bookings WHERE id = ?",
            (booking_id,),
        )
        row = cursor.fetchone()

    if row is None:
        result["status"] = "booking_not_found"
        return result

    expected_amount = Decimal(str(row[0]))
    result["expected_amount"] = expected_amount

    if extracted_amount is None:
        result["status"] = "amount_missing"
        return result

    difference = abs(expected_amount - extracted_amount)
    if difference <= settings.amount_tolerance:
        result["status"] = "verified"
    else:
        result["status"] = "amount_mismatch"

    return result


# --------------------------------------------------------------------------
# Telegram utility helpers
# --------------------------------------------------------------------------
def _chunk_for_telegram(text: str, limit: int = 3500) -> Sequence[str]:
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_text = paragraph.strip()
        if not paragraph_text:
            continue

        paragraph_length = len(paragraph_text) + 2
        if current and current_length + paragraph_length > limit:
            chunks.append("\n\n".join(current))
            current = []
            current_length = 0

        if paragraph_length > limit:
            raw = paragraph_text
            while len(raw) > limit:
                chunks.append(raw[:limit])
                raw = raw[limit:]
            if raw:
                current.append(raw)
                current_length = len(raw) + 2
        else:
            current.append(paragraph_text)
            current_length += paragraph_length

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# --------------------------------------------------------------------------
# Webhook setup and main entry point
# --------------------------------------------------------------------------
async def setup_bot() -> Application:
    """Initialize the bot and its handlers."""
    settings = load_settings()
    configure_ocr(settings)
    initialize_directories(settings)
    initialize_database(settings)

    # Note: The static server is less relevant for a pure webhook setup on Cloud Run,
    # but can be useful for local testing. It's disabled by default on production.
    if os.getenv("ENV", "production").lower() == "development":
        start_static_server(settings)

    genai.configure(api_key=settings.gemini_api_key)
    gemini_model = genai.GenerativeModel(settings.gemini_model)
    static_documents: Dict[str, str] = {
        "pms_blueprint": _load_blueprint(settings.pms_blueprint_path),
    }

    application = Application.builder().token(settings.telegram_bot_token).build()

    application.bot_data.update(
        {
            "authorized_user_id": settings.authorized_user_id,
            "gemini_model": gemini_model,
            "static_documents": static_documents,
            "settings": settings,
        }
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("menu", menu_command))
    application.add_handler(CommandHandler("ls", ls_command))
    application.add_handler(CommandHandler("pms_blueprint", pms_blueprint))
    application.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, web_app_data_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    return application


@app.on_event("startup")
async def startup_event():
    """On startup, initialize the bot and set the webhook."""
    application = await setup_bot()
    settings: Settings = application.bot_data["settings"]

    if not settings.webhook_url:
        logger.warning("WEBHOOK_URL is not set, skipping webhook setup.")
        return

    await application.bot.set_webhook(
        url=f"{settings.webhook_url}/webhook/{settings.telegram_bot_token}"
    )
    
    # Store the application instance in the FastAPI app state
    app.state.application = application
    logger.info("Bot application initialized and webhook is set.")


@app.post("/webhook/{token}")
async def webhook_handler(request: Request, token: str):
    """Handle incoming Telegram updates."""
    application: Application = app.state.application
    
    # Check if the token is correct
    if token != application.bot_data["settings"].telegram_bot_token:
        return Response(status_code=403)

    try:
        update_data = await request.json()
        update = Update.de_json(data=update_data, bot=application.bot)
        await application.process_update(update)
        return Response(status_code=200)
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from webhook request")
        return Response(status_code=400)
    except Exception:
        logger.exception("Error processing webhook update")
        return Response(status_code=500)


@app.get("/")
def health_check():
    """A simple health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    # This part is for local development and won't be used by Cloud Run.
    # Cloud Run uses the `uvicorn` command in the Dockerfile.
    settings = load_settings()
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
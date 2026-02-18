import os
import json
import logging
import asyncio
import io
import gc
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, cast, Callable, Generator

import dashscope
from dashscope import Generation
from http import HTTPStatus
from telegram import Message, Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, constants, WebAppInfo, KeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
from dotenv import load_dotenv
from faster_whisper import WhisperModel
# PaddleOCR Import - handled carefully for memory
try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False
    print("Warning: paddleocr not installed.")

load_dotenv()

# Placeholder for Google Sheets to avoid runtime errors if creds missing
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Configuration ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

GOOGLE_SHEETS_CREDENTIALS = "credentials.json"
SPREADSHEET_ID = "1o4yG81XMKyhTAAxQDDFYfAdEzR2ah7ILxQElflDwevo"

# Global Whisper Model (Load once)
try:
    # Run on CPU for broad compatibility in standard containers
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    logger.info("Faster-Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None
    
# Global PaddleOCR Model (Lazy load or load once depending on RAM strategy)
# For 512MB RAM, keeping both loaded is risky. 
# But loading on every request is slow. Let's try loading once and hope for the best, or create on demand if needed.
# Using use_angle_cls=True enhances accuracy but uses more RAM. We set lang='en'.
paddle_ocr = None
if HAS_PADDLE:
    try:
         # Use lightweight model if possible
         paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
         logger.info("PaddleOCR loaded successfully.")
    except Exception as e:
         logger.error(f"Failed to load PaddleOCR: {e}")

# --- Global States ---
REGISTER_NAME, REGISTER_CG = range(2)

# --- Database (Local Mock) ---
USER_DB_FILE = "user_db.json"

def load_user_db() -> Dict[str, Any]:
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_db(db: Dict[str, Any]):
    with open(USER_DB_FILE, "w") as f:
        json.dump(db, f, indent=4)

user_db = load_user_db()

# --- Load Syllabus ---
try:
    with open("syllabus_context.txt", "r") as f:
        SYLLABUS_CONTEXT = f.read()
except FileNotFoundError:
    SYLLABUS_CONTEXT = "You are a helpful Economics Tutor for Singapore JC students."

# --- Prompts & System Instructions ---
SYSTEM_PROMPT = f"""
You are an expert Economics Tutor for Singapore Junior College students (H1/H2).
Your goal is to guide students using the Socratic method (scaffolding), NOT to give direct answers immediately.
Always strictly adhere to the Singapore JC syllabus.

{SYLLABUS_CONTEXT}

RULES:
1. If the user asks a non-Economics question, politely redirect them to Economics.
2. Tone: Encouraging, professional, patient.
3. Use British English spelling (e.g., 'colour', 'maximise', 'centre') in all responses.
4. For Essays (Text provided from OCR): Summarize strengths, identify weaknesses, and scaffold improvements.
5. For Diagrams (Labels provided from OCR): Check accuracy based on the LABELS and Context. If labels imply a wrong shift or concept, correct it.
   - If ACCURATE: Compliment.
   - If INACCURATE: Explain the error and providing a SCAFFOLDING QUESTION/HINT.
   - CRITICAL: DO NOT provide the full correct answer YET, unless the user has failed 5 times in a row.
"""

# --- Google Sheets Setup ---
def log_to_sheets(user_info: Dict, interaction_type: str, content: str, response: str):
    if not HAS_GSPREAD or not os.path.exists(GOOGLE_SHEETS_CREDENTIALS):
        return

    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_SHEETS_CREDENTIALS, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SPREADSHEET_ID).sheet1
        
        row = [
            datetime.now().isoformat(),
            user_info.get("name"),
            user_info.get("cg"),
            interaction_type,
            content[:500], # Truncate likely large OCR text
            response[:500]
        ]
        sheet.append_row(row)
    except Exception as e:
        logger.error(f"Failed to log to sheets: {e}")

# --- Helpers ---
def get_main_keyboard():
    # Replace with your actual hosted Web App URL
    web_app_url = "https://behappie4eva.github.io/Chatbot/" 
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("🎤 Speech to Text", web_app=WebAppInfo(url=web_app_url))],
            ["Ask a Question", "Help"]
        ],
        resize_keyboard=True
    )

async def stream_dashscope_response(messages: List[Dict], model_name: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Helper to stream DashScope responses to Telegram."""
    full_response = ""
    message_obj = None
    last_update_time = 0.0
    
    try:
        # Use simple Generation call for text
        responses = Generation.call(model=model_name, messages=messages, result_format='message', stream=True)

        for response in responses:
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content
                full_response = content # DashScope streams full text so far usually
                
                current_time = datetime.now().timestamp()
                if (current_time - last_update_time > 1.0) or message_obj is None:
                    if full_response.strip():
                        if message_obj is None:
                            message_obj = await update.message.reply_text(full_response + "...")
                        else:
                            try:
                                await message_obj.edit_text(full_response + "...")
                            except Exception:
                                pass
                        last_update_time = current_time
            else:
                logger.error(f"DashScope Error: {response.code} - {response.message}")
                if message_obj is None:
                    await update.message.reply_text("Thinking...")
                break
        
        # Final update
        if message_obj:
            await message_obj.edit_text(full_response)
        elif full_response:
             await update.message.reply_text(full_response)
        
        return full_response

    except Exception as e:
        logger.error(f"Streaming Error: {e}")
        return ""

# --- Bot Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    chat_id = str(user.id)
    
    if chat_id in user_db:
        await update.message.reply_text(
            f"Welcome back, {user_db[chat_id]['name']}! How can I help you with Economics today?",
            reply_markup=get_main_keyboard()
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "Welcome to the Economics Chatbot! \nPlease tell me your **Official Name**."
    )
    return REGISTER_NAME

async def register_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["name"] = update.message.text
    await update.message.reply_text("Thanks! Now, please enter your **Civics Group (CG)**.")
    return REGISTER_CG

async def register_cg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    chat_id = str(user.id)
    
    user_db[chat_id] = {
        "name": context.user_data["name"],
        "cg": update.message.text,
        "consecutive_wrong_attempts": 0,
        "last_topic": None,
        "last_image_type": None # 'essay' or 'diagram'
    }
    save_user_db(user_db)
    
    await update.message.reply_text(
        "Registration Complete! You can now ask questions, send voice notes, or upload diagrams/essays.",
        reply_markup=get_main_keyboard()
    )
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = str(user.id)
    
    if chat_id not in user_db:
        await update.message.reply_text("Please run /start first.")
        return

    user_info = user_db[chat_id]
    interaction_type = "text"
    content_for_log = ""
    bot_response_text = ""
    
    # 0. Typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

    # 1. Handle Voice (Faster-Whisper)
    if update.message.voice:
        interaction_type = "voice"
        if not whisper_model:
            await update.message.reply_text("Voice processing is currently unavailable.")
            return
            
        try:
            file = await context.bot.get_file(update.message.voice.file_id)
            # Use unique temp file
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_ogg:
                await file.download_to_drive(temp_ogg.name)
                temp_ogg_path = temp_ogg.name

            # Transcribe
            await update.message.reply_text("🎤 *Transcribing...*", parse_mode=constants.ParseMode.MARKDOWN)
            
            # Offload to thread
            def run_transcription(path):
                segments, _ = whisper_model.transcribe(path, beam_size=5, language="en")
                return " ".join([s.text for s in segments])

            transcribed_text = await asyncio.to_thread(run_transcription, temp_ogg_path)
            
            # Clean up temp file
            os.remove(temp_ogg_path)
            
            # Explicit GC for low-memory environments
            gc.collect()
            
            await update.message.reply_text(f"🎤 *You said:* {transcribed_text}", parse_mode=constants.ParseMode.MARKDOWN)
            
            # Prepare for response generation
            content_for_log = transcribed_text
            
            # Generate response to the text - Uses Qwen-Turbo (LLM) not VLM
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': transcribed_text}
            ]
            bot_response_text = await stream_dashscope_response(messages, "qwen-turbo", update, context)

        except Exception as e:
            logger.error(f"Voice Error: {e}")
            await update.message.reply_text("Error processing voice message.")
            return

    # 2. Handle Images (Handwriting/Diagrams) - PaddleOCR + Qwen-Turbo
    elif update.message.photo:
        interaction_type = "image"
        
        if not paddle_ocr:
             await update.message.reply_text("OCR Engine not initialized.")
             return

        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        # Save to a local path
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
            await file.download_to_drive(temp_img.name)
            img_path = temp_img.name # Absolute path

        await update.message.reply_text("🔍 *Analyzing Image with PaddleOCR...*", parse_mode=constants.ParseMode.MARKDOWN)

        # Run PaddleOCR
        try:
            def run_ocr(path):
                # PaddleOCR result is a list of lists
                result = paddle_ocr.ocr(path, cls=True)
                # Concatenate all text found
                txt_list = [line[1][0] for line in result[0]] if result and result[0] else []
                return "\n".join(txt_list)

            extracted_text = await asyncio.to_thread(run_ocr, img_path)
            
            logger.info(f"OCR Extracted: {extracted_text[:100]}...")
            
            if not extracted_text.strip():
                await update.message.reply_text("I couldn't read any text in this image. Is it clear?")
                os.remove(img_path)
                return

            attempts = user_info.get("consecutive_wrong_attempts", 0)
            
            prompt_text = (
                f"Current consecutive wrong attempts by student: {attempts}. "
                "I have extracted the following text from an image (diagram labels or handwritten essay): \n"
                f"\"\"\"{extracted_text}\"\"\"\n\n"
                "1. Identify if this text looks like an ECONOMICS ESSAY (paragraphs) or DIAGRAM LABELS (short terms like Price, Quantity, DD, SS). "
                "2. IF ESSAY: Summarize strengths, identify weaknesses, and provide scaffolding questions. "
                "3. IF DIAGRAM LABELS: Infer the diagram context. Check accuracy of terms/implied shifts. "
                "   - If ACCURATE: Compliment. "
                "   - If INACCURATE: Explain the error and providing a SCAFFOLDING QUESTION/HINT. "
                "   - CRITICAL: DO NOT provide the full correct answer/drawing YET, unless the user has failed 5 times (attempts >= 5). "
                "   - If attempts >= 5, THEN provide the full direct correction/answer. "
                "Respond in a helpful, tutor tone."
            )

            # Use Qwen-Turbo (Text LLM)
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt_text}
            ]
            
            bot_response_text = await stream_dashscope_response(messages, "qwen-turbo", update, context)
            
            # Heuristic for state update
            lower_resp = bot_response_text.lower()
            if "correct" in lower_resp and "incorrect" not in lower_resp and "error" not in lower_resp:
                user_info["consecutive_wrong_attempts"] = 0
            elif "try again" in lower_resp or "?" in lower_resp or "incorrect" in lower_resp:
                user_info["consecutive_wrong_attempts"] += 1
            else:
                pass # Neutral or Essay feedback
            
            save_user_db(user_db)
            content_for_log = f"[OCR Text] {extracted_text}"

        except Exception as e:
            logger.error(f"OCR/Analysis Error: {e}")
            await update.message.reply_text("Error reading the image.")
        finally:
             if os.path.exists(img_path):
                os.remove(img_path)
             gc.collect()

    # 3. Handle Text (Qwen-Turbo)
    elif update.message.text:
        content_for_log = update.message.text
        if content_for_log == "Ask a Question":
             await update.message.reply_text("Go ahead, I'm listening!")
             return

        # Check for Web App Data (STT)
        if update.message.web_app_data:
             content_for_log = update.message.web_app_data.data
             interaction_type = "stt_webapp"
             await update.message.reply_text(f"📝 *Text:* {content_for_log}", parse_mode=constants.ParseMode.MARKDOWN)

        # Standard Text Chat
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': content_for_log}
        ]
        bot_response_text = await stream_dashscope_response(messages, "qwen-turbo", update, context)

    # 4. Log
    if user_info and content_for_log and bot_response_text:
        asyncio.create_task(asyncio.to_thread(cast(Callable, log_to_sheets), dict(user_info), str(interaction_type), str(content_for_log), str(bot_response_text)))

def main():
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set.")
        return

    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            REGISTER_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, register_name)],
            REGISTER_CG: [MessageHandler(filters.TEXT & ~filters.COMMAND, register_cg)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.TEXT | filters.VOICE | filters.PHOTO | filters.StatusUpdate.WEB_APP_DATA, handle_message))

    print("Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()

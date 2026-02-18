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
from dashscope import MultiModalConversation, Generation
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
# Using 'tiny.en' as requested. 'int8' quantization is default and fast.
try:
    # Run on CPU for broad compatibility in standard containers
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    logger.info("Faster-Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None

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
4. For Essays/Handwriting: Summarize strengths, identify weaknesses, and scaffold improvements.
5. For Diagrams: Validate accuracy. If wrong, provide HINTS. Only give the full answer if the user has failed 5 times in a row.
"""

# --- Google Sheets Setup ---
def log_to_sheets(user_info: Dict, interaction_type: str, content: str, response: str):
    if not HAS_GSPREAD or not os.path.exists(GOOGLE_SHEETS_CREDENTIALS):
        # logger.info(f"Mock Log: {user_info.get('name')} | {interaction_type}")
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
            content,
            response
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
        if model_name.startswith("qwen-vl"):
            # MultiModal does not always support streaming gracefully in all SDK versions, but let's try standard call first if streaming fails or use iterator
            responses = MultiModalConversation.call(model=model_name, messages=messages, stream=True)
        else:
            responses = Generation.call(model=model_name, messages=messages, result_format='message', stream=True)

        for response in responses:
            if response.status_code == HTTPStatus.OK:
                if model_name.startswith("qwen-vl"):
                     # Structure is different for VL
                     content = response.output.choices[0].message.content.strip()
                     # If stream returns full text, just update buffer.
                     # DashScope Qwen-VL streaming often returns full content-so-far.
                     full_response = content
                else:
                    # Qwen-Turbo/Max
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
            
            # Generate response to the text
            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': transcribed_text}
            ]
            bot_response_text = await stream_dashscope_response(messages, "qwen-turbo", update, context)

        except Exception as e:
            logger.error(f"Voice Error: {e}")
            await update.message.reply_text("Error processing voice message.")
            return

    # 2. Handle Images (Handwriting/Diagrams) - Qwen-VL
    elif update.message.photo:
        interaction_type = "image"
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        
        # Save to a local path
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
            await file.download_to_drive(temp_img.name)
            img_path = temp_img.name # Absolute path
            
        attempts = user_info.get("consecutive_wrong_attempts", 0)
        
        prompt_text = (
            f"Current consecutive wrong attempts by student: {attempts}. "
            "Analyze this image. "
            "1. Identify if it is an Economics DIAGRAM, HANDWRITTEN ESSAY, or OTHER. "
            "2. IF ESSAY: Summarize strengths, identify weaknesses, and provide scaffolding questions. "
            "3. IF DIAGRAM: Check accuracy against Singapore A-Level Economics standards. "
            "   - If ACCURATE: Compliment. "
            "   - If INACCURATE: Explain the error and providing a SCAFFOLDING QUESTION/HINT. "
            "   - CRITICAL: DO NOT provide the full correct answer/drawing YET, unless the user has failed 5 times (attempts >= 5). "
            "   - If attempts >= 5, THEN provide the full direct correction/answer. "
            "Respond in a helpful, tutor tone."
        )

        # Qwen-VL-Max
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": f"file://{img_path}"},
                    {"text": prompt_text}
                ]
            }
        ]

        try:
            bot_response_text = await stream_dashscope_response(messages, "qwen-vl-max", update, context)
            
            # Update state simple heuristic
            lower_resp = bot_response_text.lower()
            if "correct" in lower_resp and "incorrect" not in lower_resp and "error" not in lower_resp:
                user_info["consecutive_wrong_attempts"] = 0
            elif "try again" in lower_resp or "?" in lower_resp or "incorrect" in lower_resp:
                user_info["consecutive_wrong_attempts"] += 1
            else:
                pass # Neutral
            
            save_user_db(user_db)
            os.remove(img_path)
            gc.collect()
            content_for_log = "[Image Upload]"

        except Exception as e:
            logger.error(f"Image Error: {e}")
            await update.message.reply_text("Error processing image.")
            return

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

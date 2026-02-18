import os
import json
import logging
import asyncio
import io
import gc
import tempfile
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

import telegram
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, constants
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
from google import genai
from google.genai import types
import ffmpeg

# ... (Paddle Imports)

# --- Configuration ---
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Client placeholder
client = None

# ... (Sheets setup) ...

# ... (Model Loaders) ...

# ... (Global States) ...

# ... (DB functions) ...

# --- Load Syllabus ---
try:
    with open("syllabus_context.txt", "r") as f:
        SYLLABUS_CONTEXT = f.read()
except FileNotFoundError:
    SYLLABUS_CONTEXT = "Singapore JC H1 (8843) and H2 (9570) Economics Syllabus."

# --- Gemini Configuration ---
SYSTEM_INSTRUCTION = f"""
You are an expert Economics Tutor for Singapore Junior College students (H1 8843 / H2 9570).
Your goal is to guide students using the Socratic method (scaffolding).
ALWAYS strictly adhere to the Singapore JC syllabus:
{SYLLABUS_CONTEXT}

RULES:
1.  **Scaffolding**: NEVER give the direct answer immediately unless the user has failed repeatedly (see prompt instructions). Guide them with questions.
2.  **Evaluation**: You must evaluate the student's input.
    - If the student is WRONG, start your response with `[INCORRECT]`.
    - If the student is RIGHT, start your response with `[CORRECT]`.
    - If the student asks a question or the input is neutral/conversational, start with `[NEUTRAL]`.
3.  **Tone**: Encouraging, professional, patient.
4.  **Language**: Use strict British English (e.g., 'colour', 'maximise', 'centre').
5.  **Essays**: Summarize strengths, identify weaknesses, and scaffold improvements.
6.  **Diagrams**: Check label accuracy. If inaccurate, explain the error and provide a hint.
7.  **Fallback**: Use the current conversation context to determine if the student needs a direct answer.
"""

# Safety settings (New SDK Format)
# We will configure this in the chat creation

# ... (log_to_sheets) ...

# ... (start, register_name, register_cg, cancel) ...

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = str(user.id)
    
    if chat_id not in user_db:
        await update.message.reply_text("Please run /start first.")
        return

    user_info = user_db[chat_id]
    interaction_type = "text"
    user_text_input = ""
    
    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

    # 1. Processing Input (Voice/Image/Text)
    try:
        # ... (Voice/Image/Text logic same as before) ...
        # (I will copy the input processing block from previous file content to ensure it's preserved or reference it if I'm replacing the whole file? 
        # The tool replaces a block. I need to be careful not to delete the input processing logic.
        # I will Target the `handle_message` function specifically or the Gemini part of it.)
        
        # ... (Voice/Image logic omitted for brevity in thought, but must be in ReplacementContent)
        # Actually, I should probably use `replace_file_content` targeting specific blocks to avoid deleting the huge input processing chunk.
        
        # Let's Target the Gemini Interaction block inside handle_message.
        pass
    except Exception as e:
        logger.error(f"Input processing error: {e}")
        await update.message.reply_text("Something went wrong processing your input.")
        return

    # 2. Gemini Interaction
    try:
        current_attempts = user_info.get("consecutive_wrong_attempts", 0)
        history = user_info.get("history", [])

        if len(history) > 20: 
            history = history[-20:]
        
        prompt_suffix = ""
        if current_attempts >= 5:
            prompt_suffix = "\n\n[SYSTEM NOTICE: The student has failed 5 times consecutively. Please provide the DIRECT CORRECT ANSWER now and explain it clearly.]"
        
        # Convert history to new SDK format if needed
        # New SDK expects: role='user'|'model', parts=[types.Part.from_text(text=...)] or just strings?
        # It accepts list of dicts: [{'role': 'user', 'parts': [{'text': '...'}]}]
        # Our DB has [{'role': 'user', 'parts': ['text']}]
        # We might need to adjust. Let's try to map it.
        formatted_history = []
        for h in history:
            role = h['role']
            # potentially map 'model' to 'model'
            text_parts = h['parts']
            formatted_history.append({'role': role, 'parts': [{'text': p} for p in text_parts]})

        if not client:
             await update.message.reply_text("Bot initialization error (No Client).")
             return

        # Configure Chat
        chat = client.chats.create(
            model='gemini-1.5-flash', # Try generic alias first, or fallback handle in main?
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.7,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_MEDIUM_AND_ABOVE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_MEDIUM_AND_ABOVE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_MEDIUM_AND_ABOVE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_MEDIUM_AND_ABOVE"
                    ),
                ]
            ),
            history=formatted_history
        )
        
        final_prompt = user_text_input + prompt_suffix
        
        # Run in thread
        response = await asyncio.to_thread(chat.send_message, final_prompt)
        bot_response_raw = response.text
        
        # ... (Post-processing same as before) ...
        # ... (Save to DB, Reply, Log) ...

    except Exception as e:
        # ...
        pass

try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False
    print("Warning: paddleocr not installed.")

# --- Health Check Server for Render ---
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is likely running.")

def start_health_check_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    logger.info(f"Health check server started on port {port}")

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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Client configuration moved to main() using genai.Client


GOOGLE_SHEETS_CREDENTIALS = "credentials.json"
SPREADSHEET_ID = "1o4yG81XMKyhTAAxQDDFYfAdEzR2ah7ILxQElflDwevo"

# --- Model Management (Lazy Loading for Memory Efficiency) ---
# Render Free Tier has 512MB RAM. Streaming/Loading both models at once kills it.
# We will load on demand and UNLOAD immediately after use.

def get_whisper_model():
    """Load Whisper only when needed."""
    try:
        logger.info("Loading Whisper model (small.en)...")
        # 'small.en' matches ~512MB RAM constraints if careful, better accuracy.
        model = WhisperModel("small.en", device="cpu", compute_type="int8")
        return model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return None

def unload_whisper_model(model):
    """Explicitly unload Whisper to free RAM."""
    if model:
        del model
    gc.collect()
    logger.info("Unloaded Whisper model.")

def get_paddle_ocr():
    """Load PaddleOCR only when needed."""
    if not HAS_PADDLE: return None
    try:
        logger.info("Loading PaddleOCR...")
        # use_angle_cls=False saves RAM, lang='en'
        paddle = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        return paddle
    except Exception as e:
        logger.error(f"Failed to load PaddleOCR: {e}")
        return None

def unload_paddle_ocr(paddle):
    """Explicitly unload Paddle to free RAM."""
    if paddle:
        del paddle
    gc.collect()
    logger.info("Unloaded PaddleOCR.")

# --- Global States ---
REGISTER_NAME, REGISTER_CG = range(2)

# --- Database (Local Mock) ---
USER_DB_FILE = "user_db.json"

def load_user_db() -> Dict[str, Any]:
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
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
    SYLLABUS_CONTEXT = "Singapore JC H1 (8843) and H2 (9570) Economics Syllabus."

# --- Gemini Configuration ---
# Safety settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

SYSTEM_INSTRUCTION = f"""
You are an expert Economics Tutor for Singapore Junior College students (H1 8843 / H2 9570).
Your goal is to guide students using the Socratic method (scaffolding).
ALWAYS strictly adhere to the Singapore JC syllabus:
{SYLLABUS_CONTEXT}

RULES:
1.  **Scaffolding**: NEVER give the direct answer immediately unless the user has failed repeatedly (see prompt instructions). Guide them with questions.
2.  **Evaluation**: You must evaluate the student's input.
    - If the student is WRONG, start your response with `[INCORRECT]`.
    - If the student is RIGHT, start your response with `[CORRECT]`.
    - If the student asks a question or the input is neutral/conversational, start with `[NEUTRAL]`.
3.  **Tone**: Encouraging, professional, patient.
4.  **Language**: Use strict British English (e.g., 'colour', 'maximise', 'centre').
5.  **Essays**: Summarize strengths, identify weaknesses, and scaffold improvements.
6.  **Diagrams**: Check label accuracy. If inaccurate, explain the error and provide a hint.
7.  **Fallback**: Use the current conversation context to determine if the student needs a direct answer.
"""

# Initialize model placeholder (Configured in main)
model = None

# --- Google Sheets Setup ---
def log_to_sheets(user_info: Dict, interaction_type: str, content: str, response: str):
    if not HAS_GSPREAD:
        logger.error("Google Sheets Error: gspread library not installed.")
        return
        
    if not os.path.exists(GOOGLE_SHEETS_CREDENTIALS):
        logger.error(f"Google Sheets Error: Credentials file '{GOOGLE_SHEETS_CREDENTIALS}' not found.")
        return

    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_SHEETS_CREDENTIALS, scope)
        client = gspread.authorize(creds)
        
        # Open by Key
        try:
            sheet = client.open_by_key(SPREADSHEET_ID).sheet1
        except gspread.exceptions.APIError as e:
            if "403" in str(e):
                logger.error(f"Google Sheets Error: 403 FORBIDDEN. Please share the sheet with the email in credentials.json: {creds.service_account_email}")
            elif "404" in str(e):
                logger.error(f"Google Sheets Error: 404 NOT FOUND. Check SPREADSHEET_ID: {SPREADSHEET_ID}")
            raise e
            
        # Clean response of status tags for logging
        clean_response = response.replace("[INCORRECT]", "").replace("[CORRECT]", "").replace("[NEUTRAL]", "").strip()
        
        row = [
            datetime.now().isoformat(),
            user_info.get("name"),
            user_info.get("cg"),
            interaction_type,
            content[:1000], 
            clean_response[:1000]
        ]
        sheet.append_row(row)
        logger.info(f"Successfully logged interaction to Google Sheet for {user_info.get('name')}.")
    except Exception as e:
        logger.error(f"Failed to log to sheets: {e}")

# --- Bot Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    chat_id = str(user.id)
    
    if chat_id in user_db:
        await update.message.reply_text(
            f"Welcome back, {user_db[chat_id]['name']}! How can I help you with Economics today?",
            reply_markup=ReplyKeyboardRemove()
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
        "history": [] # Store chat history
    }
    save_user_db(user_db)
    
    await update.message.reply_text(
        "Registration Complete! You can now ask questions, send voice notes, or upload diagrams/essays.",
        reply_markup=ReplyKeyboardRemove()
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
    user_text_input = ""
    
    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

    # 1. Processing Input
    try:
        # --- VOICE ---
        if update.message.voice:
            interaction_type = "voice"
            
            # Lazy Load Whisper
            whisper = get_whisper_model()
            if not whisper:
                await update.message.reply_text("Voice processing currently unavailable (Low Memory).")
                return

            try:
                # Download OGG
                file = await context.bot.get_file(update.message.voice.file_id)
                with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_ogg:
                    await file.download_to_drive(temp_ogg.name)
                    input_path = temp_ogg.name
                
                output_wav = input_path.replace(".ogg", ".wav")
                
                # Convert OGG -> WAV
                (
                    ffmpeg
                    .input(input_path)
                    .output(output_wav, ac=1, ar='16000')
                    .run(quiet=True, overwrite_output=True)
                )

                # Transcribe
                await update.message.reply_text("🎤 *Transcribing...*", parse_mode=constants.ParseMode.MARKDOWN)
                
                def run_transcribe(model, path):
                    segments, _ = model.transcribe(path, beam_size=1, language="en")
                    return " ".join([s.text for s in segments])

                user_text_input = await asyncio.to_thread(run_transcribe, whisper, output_wav)
                
                # Cleanup
                if os.path.exists(input_path): os.remove(input_path)
                if os.path.exists(output_wav): os.remove(output_wav)
                
                await update.message.reply_text(f"🎤 *You said:* {user_text_input}", parse_mode=constants.ParseMode.MARKDOWN)
                
            finally:
                # Force Unload Whisper to save RAM for Gemini/OCR
                unload_whisper_model(whisper)

        # --- IMAGE ---
        elif update.message.photo:
            interaction_type = "image"
            
            # Lazy Load Paddle
            ocr_engine = get_paddle_ocr()
            if not ocr_engine:
                await update.message.reply_text("OCR processing unavailable (Low Memory).")
                return

            try:
                photo = update.message.photo[-1] # Largest size
                file = await context.bot.get_file(photo.file_id)
                
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
                    await file.download_to_drive(temp_img.name)
                    img_path = temp_img.name

                await update.message.reply_text("🔍 *Reading Image...*", parse_mode=constants.ParseMode.MARKDOWN)

                def run_ocr(engine, path):
                    result = engine.ocr(path, cls=True)
                    txts = [line[1][0] for line in result[0]] if result and result[0] else []
                    return "\n".join(txts)

                extracted_text = await asyncio.to_thread(run_ocr, ocr_engine, img_path)
                
                if not extracted_text.strip():
                    await update.message.reply_text("I couldn't read any text. Please try again with a clearer image.")
                    if os.path.exists(img_path): os.remove(img_path)
                    return
                    
                user_text_input = f"[IMAGE CONTENT: {extracted_text}]"
                if os.path.exists(img_path): os.remove(img_path)
            
            finally:
                # Force Unload Paddle
                unload_paddle_ocr(ocr_engine)

        # --- TEXT ---
        elif update.message.text:
            user_text_input = update.message.text
        
        else:
            return 

    except Exception as e:
        logger.error(f"Input processing error: {e}")
        await update.message.reply_text("Something went wrong processing your input.")
        return

    # 2. Gemini Interaction
    try:
        current_attempts = user_info.get("consecutive_wrong_attempts", 0)
        history = user_info.get("history", [])

        if len(history) > 20: 
            history = history[-20:]
        
        prompt_suffix = ""
        if current_attempts >= 5:
            prompt_suffix = "\n\n[SYSTEM NOTICE: The student has failed 5 times consecutively. Please provide the DIRECT CORRECT ANSWER now and explain it clearly.]"
        
        # New SDK History Format: [{'role': 'user', 'parts': [{'text': '...'}]}]
        formatted_history = []
        for h in history:
            role = h['role']
            text_parts = h['parts']
            # Ensure parts are list of dicts with 'text' key or just strings if supported.
            # SDK v2 usually supports valid Part objects or dicts.
            formatted_history.append({'role': role, 'parts': [{'text': p} for p in text_parts]})

        if not client:
             await update.message.reply_text("Bot initialization error (No Client).")
             return

        # Configure Chat
        # Using a fresh chat session with history for each request (stateless bot perspective with history injection)
        def run_chat(h, prompt):
            chat = client.chats.create(
                model='gemini-1.5-flash',
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.7,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
                    ]
                ),
                history=h
            )
            return chat.send_message(prompt)

        response = await asyncio.to_thread(run_chat, formatted_history, user_text_input + prompt_suffix)
        bot_response_raw = response.text
        
        # 3. Post-Processing
        final_response_text = bot_response_raw
        
        if bot_response_raw.startswith("[INCORRECT]"):
            user_info["consecutive_wrong_attempts"] += 1
            final_response_text = bot_response_raw.replace("[INCORRECT]", "").strip()
        elif bot_response_raw.startswith("[CORRECT]"):
            user_info["consecutive_wrong_attempts"] = 0
            final_response_text = bot_response_raw.replace("[CORRECT]", "").strip()
        elif bot_response_raw.startswith("[NEUTRAL]"):
            final_response_text = bot_response_raw.replace("[NEUTRAL]", "").strip()
        
        user_info["history"].append({"role": "user", "parts": [user_text_input]})
        user_info["history"].append({"role": "model", "parts": [bot_response_raw]})
        
        save_user_db(user_db)
        
        await update.message.reply_text(final_response_text)
        
        asyncio.create_task(asyncio.to_thread(log_to_sheets, user_info, interaction_type, user_text_input, final_response_text))

    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        await update.message.reply_text("I'm having trouble thinking right now. Please try again.")

# --- Debug Handler ---
async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Helper command to get Service Email and Model info."""
    response_lines = ["🔧 *Debug Info*"]
    
    # 1. Credentials / Email
    if os.path.exists(GOOGLE_SHEETS_CREDENTIALS):
        try:
            with open(GOOGLE_SHEETS_CREDENTIALS, 'r') as f:
                creds = json.load(f)
            email = creds.get('client_email', 'Unknown')
            response_lines.append(f"📧 *Service Email:* `{email}`")
            response_lines.append("(Share your Google Sheet with this email!)")
        except Exception as e:
            response_lines.append(f"⚠️ Creds Error: {str(e)}")
    else:
        response_lines.append("❌ Credentials file NOT found.")

    # 2. Gemini Models
    try:
        models = []
        if client:
            for m in client.models.list():
                # Filter useful models? Or just list text ones. 
                # SDK v2 models object has `name`.
                # Check formatting.
                name = m.name 
                if "gemini" in name and "flash" in name:
                     models.append(name.replace("models/", ""))
            response_lines.append(f"🤖 *Available Models (Flash):* \n`{', '.join(models)}`")
        else:
            response_lines.append("❌ Client not initialized.")
    except Exception as e:
        response_lines.append(f"❌ Model List Error: {e}")
        
    await update.message.reply_text("\n".join(response_lines), parse_mode=constants.ParseMode.MARKDOWN)

# --- Main ---
def main():
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set.")
        return

    # Debug Google API Key (Masked)
    if not GOOGLE_API_KEY:
        print("CRITICAL ERROR: GOOGLE_API_KEY is NOT set in environment variables.")
    else:
        masked_key = GOOGLE_API_KEY[:4] + "*" * (len(GOOGLE_API_KEY) - 8) + GOOGLE_API_KEY[-4:]
        print(f"GOOGLE_API_KEY found: {masked_key}")

    # Initialize Gemini Client
    global client
    if GOOGLE_API_KEY:
        try:
            client = genai.Client(api_key=GOOGLE_API_KEY)
            print("Gemini Client Initialized.")
            
            # List models to verify
            print("Accessible Models:")
            for m in client.models.list():
                if "gemini" in m.name:
                    print(f" - {m.name}")
        except Exception as e:
             print(f"Failed to initialize Gemini Client: {e}")
    else:
        print("CRITICAL: GOOGLE_API_KEY missing.")

    # Start the dummy server for Render
    start_health_check_server()

    # Check FFmpeg
    try:
        ffmpeg.input("headers_check").output("null", f="null").run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error:
        pass 
    except FileNotFoundError:
        print("CRITICAL WARNING: FFmpeg not found in path.")
        
    # Check Credentials
    if os.path.exists(GOOGLE_SHEETS_CREDENTIALS):
        print(f"Credentials file found: {GOOGLE_SHEETS_CREDENTIALS}")
        try:
            with open(GOOGLE_SHEETS_CREDENTIALS, 'r') as f:
                creds_data = json.load(f)
            print(f"Service Account Email: {creds_data.get('client_email', 'Unknown')}")
        except Exception:
            pass
    else:
        print(f"CRITICAL WARNING: Credentials file '{GOOGLE_SHEETS_CREDENTIALS}' NOT found. Logging will fail.")

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
    application.add_handler(CommandHandler("debug", debug)) # Add Debug Command
    application.add_handler(MessageHandler(filters.TEXT | filters.VOICE | filters.PHOTO, handle_message))

    print("Bot is running...")
    application.run_polling()


if __name__ == "__main__":
    main()

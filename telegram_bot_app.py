import logging
import os
import google.generativeai as genai
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Configure the Gemini API with your API key
genai.configure(api_key="AIzaSyAmVhVaV5cqDHUH0pIsiyaAsN2S4cXjeCQ")

# Replace with your actual bot token from memory
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8227507211:AAH83ww5AfhlKaNyhF3sCFkNk8pTCaxsh38")
# Replace with your actual user ID from memory, for authorized access
AUTHORIZED_USER_ID = int(os.getenv("TELEGRAM_USER_ID", "8144545476"))

async def start(update: Update, context: Application) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    if user.id == AUTHORIZED_USER_ID:
        await update.message.reply_html(
            f"Hi {user.mention_html()}! I'm your personal automation assistant. How can I help you today?",
        )
    else:
        await update.message.reply_text("You are not authorized to use this bot.")

async def handle_ai_conversation(text: str) -> str:
    """Engages in AI conversation using the Gemini API."""
    model = genai.GenerativeModel('gemini-pro')
    try:
        response = await model.generate_content_async(text)
        return response.text
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return "I'm sorry, I'm having trouble connecting to the AI at the moment. Please try again later."

async def echo(update: Update, context: Application) -> None:
    """Echo the user message or engage in AI conversation."""
    user = update.effective_user
    if user.id == AUTHORIZED_USER_ID:
        user_message = update.message.text
        ai_response = await handle_ai_conversation(user_message)
        await update.message.reply_text(ai_response)
    else:
        await update.message.reply_text("You are not authorized to use this bot.")

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("ls", ls_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


async def ls_command(update: Update, context: Application) -> None:
    """Lists the contents of the current directory."""
    user = update.effective_user
    if user.id == AUTHORIZED_USER_ID:
        try:
            import subprocess
            result = subprocess.run(['ls', '-F'], capture_output=True, text=True, check=True)
            await update.message.reply_text(f"```\n{result.stdout}\n```")
        except subprocess.CalledProcessError as e:
            await update.message.reply_text(f"Error executing command: {e.stderr}")
        except Exception as e:
            await update.message.reply_text(f"An unexpected error occurred: {e}")
    else:
        await update.message.reply_text("You are not authorized to use this bot.")

if __name__ == "__main__":
    main()


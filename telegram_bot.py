import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://127.0.0.1:8000"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

THRESHOLD = 0.5


# --- START COMMAND ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me a message, I'll check toxicity.")


# --- MESSAGE HANDLER ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    try:
        res = requests.post(f"{API_URL}/predict", json={"text": text})
        data = res.json()
    except Exception:
        await update.message.reply_text("⚠️ API error")
        return

    scores = data.get("scores", {})
    suppressed = data.get("suppressed_labels", [])

    print("DEBUG SUPPRESSED:", suppressed)
    print("DEBUG SCORES:", scores)

    # --- SUPPRESSION FIX (IMPORTANT BUG FIX) ---
    filtered_scores = {
        k: v for k, v in scores.items()
        if k not in suppressed
    }

    # If everything suppressed → treat as safe
    if not filtered_scores:
        max_score = 0
    else:
        max_score = max(filtered_scores.values())

    # --- DECISION ---
    if max_score > THRESHOLD:
        label = "⚠️ Toxic"
    else:
        label = "✅ Safe"

    reply = (
        f"{label}\n\n"
        f"Toxic: {scores.get('toxic', 0):.2f}\n"
        f"Insult: {scores.get('insult', 0):.2f}\n"
        f"Threat: {scores.get('threat', 0):.2f}"
    )

    keyboard = [
        [InlineKeyboardButton("Mark as Safe", callback_data=text)]
    ]
    markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(reply, reply_markup=markup)


# --- FEEDBACK BUTTON ---
async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    text = query.data

    try:
        requests.post(f"{API_URL}/feedback/safe", json={"text": text})
    except Exception:
        await query.answer("API error")
        return

    await query.answer()
    await query.edit_message_text("✅ Marked as safe. Model will adapt.")


# --- MAIN ---
def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_button))

    app.run_polling()


if __name__ == "__main__":
    main()

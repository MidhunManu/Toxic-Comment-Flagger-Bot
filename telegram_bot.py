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

API_URL = "http://127.0.0.1:8000"

# --- START COMMAND ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me a message, I'll check toxicity.")

# --- MESSAGE HANDLER ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    # Call your API
    res = requests.post(f"{API_URL}/predict", json={"text": text})
    data = res.json()

    scores = data["scores"]
    toxic_score = scores["toxic"]

    # Decide UI
    if toxic_score > 0.5:
        label = "⚠️ Toxic"
    else:
        label = "✅ Safe"

    reply = (
        f"{label}\n\n"
        f"Toxic: {scores['toxic']:.2f}\n"
        f"Insult: {scores['insult']:.2f}\n"
        f"Threat: {scores['threat']:.2f}"
    )

    # Inline button → mark safe
    keyboard = [
        [InlineKeyboardButton("Mark as Safe", callback_data=text)]
    ]
    markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(reply, reply_markup=markup)

# --- BUTTON HANDLER ---
async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    text = query.data

    # Call feedback API
    requests.post(f"{API_URL}/feedback/safe", json={"text": text})

    await query.answer()
    await query.edit_message_text("✅ Marked as safe. Model will adapt.")

# --- MAIN ---
def main():
    app = ApplicationBuilder().token("YOUR_BOT_TOKEN").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_button))

    app.run_polling()

if __name__ == "__main__":
    main()

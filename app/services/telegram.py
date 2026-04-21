import requests
from app.config import settings

class TelegramNotifier:
    """Service to send alerts to Telegram"""
    def __init__(self):
        self.token = getattr(settings, "TELEGRAM_BOT_TOKEN", None)
        self.chat_id = getattr(settings, "TELEGRAM_CHAT_ID", None)
        self.enabled = bool(self.token and self.chat_id)

    def send_message(self, message: str):
        if not self.enabled:
            return
        
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": self.chat_id, "text": message})
        except Exception as e:
            print(f"Telegram Error: {e}")

# Create an alias in case other files use 'TelegramService'
TelegramService = TelegramNotifier

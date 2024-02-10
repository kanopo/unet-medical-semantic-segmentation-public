from PIL import Image
import requests
import telegram

TOKEN = ""
# url_updates = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
# print(requests.get(url_updates).json())
CHAT_ID = ""

bot = telegram.Bot(TOKEN)


def send_log(message):
    try:
        bot.sendMessage(CHAT_ID, message)

    except:
        print("Telegram API Error")


def send_pred(image_path: str):
    try:
        bot.sendPhoto(CHAT_ID, photo=open(image_path, "rb"))
    except Exception as e:
        raise e

def send_file(file_path: str):
    try:
        bot.sendDocument(CHAT_ID, document=open(file_path))
    except Exception as e:
        raise e


if __name__ == "__main__":
    print("ciao")
    send_log("ciao")

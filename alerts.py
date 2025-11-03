import time

import requests

from config import THINGSPEAK_URL, THINGSPEAK_WRITE_API, ZAPIER_WEBHOOK_URL


def send_thingspeak_alert(field, value):
    try:
        res = requests.get(THINGSPEAK_URL, params={"api_key": THINGSPEAK_WRITE_API, f"field{field}": value}, timeout=5)
        if res.status_code == 200 and res.text != "0":
            print(f"⚠️ Đã gửi field{field}={value} lên ThingSpeak.")
            return True
    except Exception as e:
        print(f"❌ Lỗi gửi ThingSpeak: {e}")
    return False


def send_zapier_webhook(image_link, alert_type):
    payload = {
        "message": f"⚠️ Cảnh báo {alert_type}!",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_link": image_link,
    }
    try:
        res = requests.post(ZAPIER_WEBHOOK_URL, json=payload, timeout=10)
        if res.status_code == 200:
            print("📤 Gửi webhook tới Zapier thành công.")
            return True
        else:
            print(f"❌ Lỗi Zapier {res.status_code}: {res.text}")
    except Exception as e:
        print(f"❌ Zapier error: {e}")
    return False

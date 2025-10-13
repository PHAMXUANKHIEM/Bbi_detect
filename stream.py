import time

import cv2


def connect_to_stream(url, max_retries=5):
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print("✅ Đã kết nối ESP32-CAM.")
            return cap
        print(f"❌ Lần thử {attempt + 1} thất bại.")
        time.sleep(2)
    print("❌ Không kết nối được ESP32-CAM.")
    exit()

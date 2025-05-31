import face_recognition
import cv2
import os
import requests
import time
import numpy
import logging
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# === THINGSPEAK & ZAPIER ===
THINGSPEAK_WRITE_API = "SIB4RQW76PBZJFTG"
THINGSPEAK_URL = "https://api.thingspeak.com/update"
ZAPIER_WEBHOOK_URL = "https://hooks.zapier.com/hooks/catch/22976197/27fmde4/"  # Thay bằng webhook của bạn

# === GOOGLE DRIVE SETUP ===
SCOPES = ['https://www.googleapis.com/auth/drive.file']
GOOGLE_DRIVE_FOLDER_ID = "1kTGjnNWWKZm2gGfG1QpDfzrWX8qrZAJA"  # Thay bằng ID thư mục Drive của bạn

# === KẾT NỐI STREAM ESP32-CAM ===
esp32_stream_url = "http://192.168.43.174/stream"  # Thay bằng IP ESP32-CAM

def get_drive_service():
    creds = None
    if os.path.exists('../drive/token.json'):
        creds = Credentials.from_authorized_user_file('../drive/token.json', SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file('../drive/credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('../drive/token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def upload_to_drive(file_path, folder_id=None):
    service = get_drive_service()
    file_metadata = {'name': os.path.basename(file_path)}
    if folder_id:
        file_metadata['parents'] = [folder_id]
    media = MediaFileUpload(file_path, mimetype='image/jpeg')
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink'
    ).execute()

    link = f"https://drive.usercontent.google.com/download?id={file.get('id')}&export=view&authuser=0"
    return link

# === LOGGING ===
logging.basicConfig(filename='../log/face_recognition.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# === LOAD FACES ===
known_faces_dir = "../known_faces"
known_encodings, known_names = [], []

if not os.path.exists(known_faces_dir):
    print(f"❌ Thư mục {known_faces_dir} không tồn tại."); exit()

for filename in os.listdir(known_faces_dir):
    path = os.path.join(known_faces_dir, filename)
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])
        print(f"✅ Đã load: {filename}")
    else:
        print(f"⚠️ Không có khuôn mặt trong: {filename}")




def connect_to_stream(url, max_retries=5):
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print("✅ Đã kết nối ESP32-CAM.")
            return cap
        print(f"❌ Lần thử {attempt+1} thất bại.")
        time.sleep(2)
    print("❌ Không kết nối được ESP32-CAM."); exit()

# === GỬI CẢNH BÁO ===
def send_thingspeak_alert():
    try:
        res = requests.get(THINGSPEAK_URL, params={
            "api_key": THINGSPEAK_WRITE_API,
            "field1": 1
        }, timeout=5)
        if res.status_code == 200 and res.text != '0':
            print("⚠️ Đã gửi cảnh báo lên ThingSpeak.")
            return True
    except Exception as e:
        print(f"❌ Lỗi gửi ThingSpeak: {e}")
    return False

def send_zapier_webhook(image_link):
    payload = {
        "message": "⚠️ Phát hiện người lạ từ ESP32-CAM!",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_link": image_link  # Gửi link ảnh tới Zapier
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

# === VÒNG LẶP CHÍNH ===
cap = connect_to_stream(esp32_stream_url)
last_alert_time = 0
alert_interval = 30  # Giới hạn mỗi 30 giây mới cảnh báo lại
frame_count = 0
recognition_interval = 5  # Mỗi 5 frame mới nhận diện

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Không lấy được frame, thử lại...")
        cap.release()
        cap = connect_to_stream(esp32_stream_url)
        continue

    frame_count += 1
    if frame_count % recognition_interval == 0:
        rgb = numpy.ascontiguousarray(frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Người lạ"
            if True in matches:
                name = known_names[matches.index(True)]
                print(f"✅ Nhận diện: {name}")
            else:
                print("⚠️ Phát hiện người lạ!")
                current_time = time.time()
                if current_time - last_alert_time > alert_interval:
                    filename = f"intruder_{int(current_time)}.jpg"
                    cv2.imwrite(filename, frame)

                    if send_thingspeak_alert():
                        drive_link = upload_to_drive(filename, GOOGLE_DRIVE_FOLDER_ID)
                        print(f"🔗 Link Google Drive: {drive_link}")
                        send_zapier_webhook(drive_link)
                        last_alert_time = current_time

            # Vẽ khung và tên
            top, right, bottom, left = location
            color = (0, 255, 0) if name != "Người lạ" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.imshow("ESP32-CAM Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

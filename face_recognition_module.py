import os

import face_recognition


def load_known_faces(folder="known_faces"):
    known_encodings = []
    known_names = []
    if not os.path.exists(folder):
        print(f"❌ Thư mục {folder} không tồn tại.")
        exit()

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])
            print(f"✅ Đã load: {filename}")
        else:
            print(f"⚠️ Không có khuôn mặt trong: {filename}")

    return known_encodings, known_names

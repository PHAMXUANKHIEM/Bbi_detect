import cv2
import torch
import time
import requests
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis
from utils.general import non_max_suppression, scale_coords
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
import os
import glob
import warnings
import mediapipe as mp
import math

# Tắt cảnh báo FutureWarning từ InsightFace
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

# 🔑 Cấu hình mô hình & camera
YOLO_WEIGHTS = 'baby_detection_custom7/weights/best.pt'
STREAM_URL = 'http://192.168.184.101/stream'
DEVICE = select_device('0' if torch.cuda.is_available() else 'cpu')

# 🌐 Cấu hình ThingSpeak
THINGSPEAK_WRITE_API = 'XU6S8WCO3TA9PQ61'
THINGSPEAK_URL = 'https://api.thingspeak.com/update'

# ⏱️ Cấu hình thời gian
SEND_INTERVAL = 60  # gửi ThingSpeak mỗi 60 giây
LOG_INTERVAL = 30  # in log mỗi 30 giây
FACE_CHECK_INTERVAL = 5  # kiểm tra khuôn mặt mỗi 5 giây
POSE_CHECK_INTERVAL = 2  # kiểm tra tư thế mỗi 2 giây

# 📂 Thư mục chứa ảnh khuôn mặt đã biết (người thân)
KNOWN_FACES_DIR = "./known_faces/"
SIMILARITY_THRESHOLD = 0.28

# ⚠️ Ngưỡng cảnh báo tư thế
DANGER_ANGLE_THRESHOLD = 45  # Góc nghiêng nguy hiểm (độ)
LOW_CONFIDENCE_THRESHOLD = 0.5  # Ngưỡng confidence thấp cho pose detection


class PoseAnalyzer:
    """Phân tích tư thế từ MediaPipe landmarks"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_angle(self, point1, point2, point3):
        """Tính góc giữa 3 điểm"""
        # Vector từ point2 đến point1
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        # Vector từ point2 đến point3
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

        # Tính góc
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def analyze_pose(self, frame):
        """Phân tích tư thế từ frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        pose_info = {
            'pose_detected': False,
            'is_lying_face_down': False,
            'is_on_side': False,
            'is_sitting': False,
            'is_crawling': False,
            'head_angle': 0,
            'body_angle': 0,
            'danger_level': 'safe',
            'landmarks': None,
            'confidence': 0.0
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            pose_info['pose_detected'] = True
            pose_info['landmarks'] = results.pose_landmarks

            # Lấy các điểm quan trọng
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]

            # Tính confidence trung bình
            key_points = [nose, left_shoulder, right_shoulder, left_hip, right_hip]
            pose_info['confidence'] = np.mean([p.visibility for p in key_points])

            if pose_info['confidence'] > LOW_CONFIDENCE_THRESHOLD:
                # Tính góc cơ thể (từ vai đến hông)
                shoulder_center = [(left_shoulder.x + right_shoulder.x) / 2,
                                   (left_shoulder.y + right_shoulder.y) / 2]
                hip_center = [(left_hip.x + right_hip.x) / 2,
                              (left_hip.y + right_hip.y) / 2]

                # Góc nghiêng của cơ thể so với phương thẳng đứng
                body_vector = [hip_center[0] - shoulder_center[0],
                               hip_center[1] - shoulder_center[1]]
                vertical_vector = [0, 1]

                if abs(body_vector[0]) > 1e-6 or abs(body_vector[1]) > 1e-6:
                    cos_angle = np.dot(body_vector, vertical_vector) / (
                            np.linalg.norm(body_vector) * np.linalg.norm(vertical_vector) + 1e-6)
                    pose_info['body_angle'] = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

                # Phân tích tư thế cụ thể
                pose_info = self._classify_pose(pose_info, landmarks)

        return pose_info

    def _classify_pose(self, pose_info, landmarks):
        """Phân loại tư thế cụ thể"""
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]

        # Tính toán vị trí trung bình
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        knee_y = (left_knee.y + right_knee.y) / 2

        # Kiểm tra nằm sấp (mặt xuống)
        # Nếu nose.z > shoulder.z có thể là dấu hiệu nằm sấp
        nose_z = getattr(nose, 'z', 0)
        shoulder_z_avg = (getattr(left_shoulder, 'z', 0) + getattr(right_shoulder, 'z', 0)) / 2

        if nose_z > shoulder_z_avg + 0.05 and abs(pose_info['body_angle'] - 90) < 30:
            pose_info['is_lying_face_down'] = True
            pose_info['danger_level'] = 'high'

        # Kiểm tra nằm nghiêng
        shoulder_x_diff = abs(left_shoulder.x - right_shoulder.x)
        if shoulder_x_diff < 0.1 and 30 < pose_info['body_angle'] < 60:
            pose_info['is_on_side'] = True
            pose_info['danger_level'] = 'medium'

        # Kiểm tra ngồi
        if shoulder_y < hip_y < knee_y and pose_info['body_angle'] < 30:
            pose_info['is_sitting'] = True
            pose_info['danger_level'] = 'safe'

        # Kiểm tra bò
        if abs(shoulder_y - knee_y) < 0.2 and hip_y > shoulder_y:
            pose_info['is_crawling'] = True
            pose_info['danger_level'] = 'safe'

        # Cảnh báo góc nghiêng nguy hiểm
        if pose_info['body_angle'] > DANGER_ANGLE_THRESHOLD:
            if pose_info['danger_level'] == 'safe':
                pose_info['danger_level'] = 'medium'

        return pose_info

    def draw_pose(self, frame, pose_info):
        """Vẽ pose landmarks lên frame"""
        if pose_info['landmarks']:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_info['landmarks'],
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        return frame


class BabyMonitoringSystem:
    def __init__(self):
        # Khởi tạo các biến thời gian
        self.last_sent_time = 0
        self.last_log_time = 0
        self.last_face_check_time = 0
        self.last_pose_check_time = 0

        # Load mô hình YOLO
        print("🔄 Đang tải mô hình YOLO...")
        self.yolo_model = DetectMultiBackend(YOLO_WEIGHTS, device=DEVICE)
        self.yolo_model.eval()
        self.stride, self.names = self.yolo_model.stride, self.yolo_model.names
        print("✅ Đã tải xong mô hình YOLO")

        # Khởi tạo InsightFace
        print("🔄 Đang tải mô hình InsightFace...")
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
        print("✅ Đã tải xong mô hình InsightFace")

        # Khởi tạo MediaPipe Pose
        print("🔄 Đang tải MediaPipe Pose...")
        self.pose_analyzer = PoseAnalyzer()
        print("✅ Đã tải xong MediaPipe Pose")

        # Load khuôn mặt đã biết
        self.known_embeddings = []
        self.known_names = []
        self.load_known_faces()

        # Kết nối camera
        self.cap = cv2.VideoCapture(STREAM_URL)
        if not self.cap.isOpened():
            print("❌ Không kết nối được camera.")
            exit()

        print("🚀 Hệ thống sẵn sàng hoạt động!")

    def load_known_faces(self):
        """Load và encode các khuôn mặt đã biết từ thư mục"""
        if not os.path.exists(KNOWN_FACES_DIR):
            print(f"⚠️ Không tìm thấy thư mục {KNOWN_FACES_DIR}")
            print("📝 Tạo thư mục và thêm ảnh khuôn mặt người thân vào đó")
            os.makedirs(KNOWN_FACES_DIR)
            return

        # Tìm tất cả file ảnh trong thư mục
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(KNOWN_FACES_DIR, ext)))

        if not image_files:
            print(f"⚠️ Không tìm thấy ảnh nào trong {KNOWN_FACES_DIR}")
            return

        print(f"🔄 Đang load {len(image_files)} ảnh khuôn mặt đã biết...")

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = self.face_app.get(img)
            if faces:
                embedding = faces[0].normed_embedding
                self.known_embeddings.append(embedding)
                name = os.path.splitext(os.path.basename(img_path))[0]
                self.known_names.append(name)
                print(f"✅ Đã load: {name}")

        print(f"📋 Tổng cộng đã load {len(self.known_embeddings)} khuôn mặt đã biết")

    def cosine_similarity(self, embedding1, embedding2, eps=1e-5):
        """Tính Cosine Similarity giữa hai embedding"""
        embedding1 = embedding1.ravel()
        embedding2 = embedding2.ravel()
        denominator = norm(embedding1) * norm(embedding2) + eps
        return np.dot(embedding1, embedding2) / denominator

    def is_known_person(self, face_embedding):
        """Kiểm tra xem khuôn mặt có phải là người đã biết không"""
        if not self.known_embeddings:
            return False, "Unknown", 0.0

        max_similarity = 0.0
        best_match_name = "Unknown"

        for i, known_embedding in enumerate(self.known_embeddings):
            similarity = self.cosine_similarity(face_embedding, known_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_name = self.known_names[i]

        is_known = max_similarity >= SIMILARITY_THRESHOLD
        return is_known, best_match_name, max_similarity

    def send_to_thingspeak(self, label, conf, extra_info=""):
        """Gửi dữ liệu lên ThingSpeak"""
        try:
            payload = {
                'api_key': THINGSPEAK_WRITE_API,
                'field1': label,
                'field2': round(conf * 100, 2),
                'field3': extra_info
            }
            r = requests.get(THINGSPEAK_URL, params=payload, timeout=10)
            if r.status_code == 200:
                print("✅ Đã gửi dữ liệu lên ThingSpeak.")
            else:
                print(f"⚠️ Gửi thất bại: {r.status_code}")
        except Exception as e:
            print("❌ Lỗi gửi ThingSpeak:", e)

    def detect_objects(self, frame):
        """Phát hiện đối tượng bằng YOLO"""
        img = letterbox(frame, 415, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1].copy()
        img = torch.from_numpy(img).to(DEVICE)
        img = img.float() / 255.0
        img = img.unsqueeze(0)

        pred = self.yolo_model(img)
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45)

        detections = []
        if pred and any(len(det) for det in pred):
            for det in pred:
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in det:
                        label = self.names[int(cls)]
                        detections.append({
                            'label': label,
                            'confidence': float(conf),
                            'bbox': [int(x) for x in xyxy]
                        })

        return detections

    def detect_faces(self, frame):
        """Phát hiện và nhận diện khuôn mặt"""
        faces = self.face_app.get(frame)
        face_results = []

        for face in faces:
            embedding = face.normed_embedding
            is_known, name, similarity = self.is_known_person(embedding)
            bbox = face.bbox.astype(int)

            face_results.append({
                'bbox': bbox,
                'is_known': is_known,
                'name': name,
                'similarity': similarity
            })

        return face_results

    def draw_detections(self, frame, yolo_detections, face_detections, pose_info):
        """Vẽ kết quả phát hiện lên frame"""
        # Vẽ YOLO detections
        for det in yolo_detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['confidence']

            if label.lower() in ['baby_face_down', 'danger']:
                color = (0, 0, 255)  # Đỏ cho nguy hiểm
            else:
                color = (0, 255, 0)  # Xanh lá cho bình thường

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Vẽ face detections
        for face in face_detections:
            x1, y1, x2, y2 = face['bbox']
            is_known = face['is_known']
            name = face['name']
            similarity = face['similarity']

            color = (0, 255, 0) if is_known else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            status = "Known" if is_known else "STRANGER"
            text = f"{status}: {name} ({similarity:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Vẽ pose landmarks
        if pose_info['pose_detected']:
            frame = self.pose_analyzer.draw_pose(frame, pose_info)

        return frame

    def get_pose_status_text(self, pose_info):
        """Lấy text mô tả tư thế"""
        if not pose_info['pose_detected']:
            return "No pose detected"

        status_parts = []

        if pose_info['is_lying_face_down']:
            status_parts.append("🔴 LYING FACE DOWN!")
        elif pose_info['is_on_side']:
            status_parts.append("🟡 On side")
        elif pose_info['is_sitting']:
            status_parts.append("🟢 Sitting")
        elif pose_info['is_crawling']:
            status_parts.append("🟢 Crawling")
        else:
            status_parts.append("Standing/Other")

        status_parts.append(f"Angle: {pose_info['body_angle']:.1f}°")
        status_parts.append(f"Danger: {pose_info['danger_level'].upper()}")

        return " | ".join(status_parts)

    def run(self):
        """Chạy hệ thống giám sát"""
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            current_time = time.time()
            found_baby = False
            found_stranger = False
            pose_info = {'pose_detected': False, 'danger_level': 'safe'}

            # Xử lý mỗi 3 frame để tối ưu hiệu suất
            if frame_count % 3 == 0:
                # Phát hiện đối tượng bằng YOLO
                yolo_detections = self.detect_objects(frame)

                # Kiểm tra phát hiện em bé
                for det in yolo_detections:
                    label = det['label']
                    conf = det['confidence']

                    print(f"📍 YOLO phát hiện: {label} ({conf:.2f})")

                    if label.lower() not in ['baby_face_down', 'danger']:
                        found_baby = True

                    # Gửi cảnh báo nếu phát hiện tình huống nguy hiểm
                    if label.lower() in ['baby_face_down', 'danger']:
                        found_baby = True
                        if current_time - self.last_sent_time >= SEND_INTERVAL:
                            self.send_to_thingspeak(label, conf, "YOLO_DANGER_DETECTED")
                            self.last_sent_time = current_time
                            print(f"🚨 YOLO CẢNH BÁO: {label} - Đã gửi thông báo!")

                # Phát hiện tư thế bằng MediaPipe
                if current_time - self.last_pose_check_time >= POSE_CHECK_INTERVAL:
                    pose_info = self.pose_analyzer.analyze_pose(frame)
                    self.last_pose_check_time = current_time

                    if pose_info['pose_detected']:
                        print(f"🧍 Pose: {self.get_pose_status_text(pose_info)}")

                        # Gửi cảnh báo tư thế nguy hiểm
                        if pose_info['danger_level'] == 'high':
                            if current_time - self.last_sent_time >= SEND_INTERVAL:
                                danger_type = "FACE_DOWN_POSE" if pose_info['is_lying_face_down'] else "DANGEROUS_POSE"
                                self.send_to_thingspeak("dangerous_pose",
                                                        pose_info['confidence'],
                                                        danger_type)
                                self.last_sent_time = current_time
                                print(f"🚨 POSE CẢNH BÁO: Tư thế nguy hiểm - Đã gửi thông báo!")

                # Phát hiện khuôn mặt
                face_detections = []
                if current_time - self.last_face_check_time >= FACE_CHECK_INTERVAL:
                    face_detections = self.detect_faces(frame)
                    self.last_face_check_time = current_time

                    for face in face_detections:
                        if not face['is_known']:
                            found_stranger = True
                            if current_time - self.last_sent_time >= SEND_INTERVAL:
                                self.send_to_thingspeak("stranger_detected",
                                                        face['similarity'],
                                                        f"Unknown_person_{face['similarity']:.2f}")
                                self.last_sent_time = current_time
                                print(f"👤 CẢNH BÁO: Phát hiện người lạ!")
                        else:
                            print(f"👋 Nhận diện: {face['name']} ({face['similarity']:.2f})")

                # Vẽ kết quả lên frame
                frame = self.draw_detections(frame, yolo_detections, face_detections, pose_info)

                # Kiểm tra không thấy em bé
                if not found_baby and not pose_info['pose_detected']:
                    if current_time - self.last_log_time >= LOG_INTERVAL:
                        print("🚨 Không phát hiện em bé trong khung hình.")
                        self.last_log_time = current_time

                    if current_time - self.last_sent_time >= SEND_INTERVAL:
                        self.send_to_thingspeak("no_baby_detected", 1.0, "BABY_MISSING")
                        self.last_sent_time = current_time

            # Hiển thị thông tin trạng thái
            status_text = []
            if found_baby or pose_info['pose_detected']:
                status_text.append("👶 Baby: DETECTED")
            else:
                status_text.append("❌ Baby: NOT FOUND")

            if pose_info['pose_detected']:
                pose_status = self.get_pose_status_text(pose_info)
                status_text.append(f"🧍 {pose_status}")

            if found_stranger:
                status_text.append("⚠️ Stranger detected!")

            # Hiển thị trạng thái lên frame
            for i, text in enumerate(status_text):
                # Chọn màu dựa trên mức độ nguy hiểm
                if "FACE DOWN" in text or "HIGH" in text:
                    color = (0, 0, 255)  # Đỏ
                elif "MEDIUM" in text or "Stranger" in text:
                    color = (0, 165, 255)  # Cam
                else:
                    color = (255, 255, 255)  # Trắng

                cv2.putText(frame, text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Hiển thị frame
            cv2.imshow("Baby Monitoring System with Pose Detection", frame)
            frame_count += 1

            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) == ord('q'):
                break

        # Dọn dẹp
        self.cap.release()
        cv2.destroyAllWindows()
        print("🔚 Đã dừng hệ thống giám sát.")


if __name__ == "__main__":
    print("🚀 Khởi động hệ thống giám sát trẻ em với phân tích tư thế...")

    # Tạo và chạy hệ thống
    monitor = BabyMonitoringSystem()

    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\n⏹️ Dừng hệ thống bằng tay...")
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")
    finally:
        cv2.destroyAllWindows()
        print("✅ Đã thoát an toàn.")
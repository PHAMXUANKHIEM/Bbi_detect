import cv2
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

weights = r"C:\best.pt"
stream_url = "http://192.168.133.174/stream"
device = select_device("0" if torch.cuda.is_available() else "cpu")


model = DetectMultiBackend(weights, device=device)
model.eval()
stride, names = model.stride, model.names

cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("❌ Không kết nối được camera.")
    exit()
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    if frame_count % 3 == 0:
        # Resize và chuẩn bị ảnh
        img = letterbox(frame, 415, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1].copy()
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.unsqueeze(0)
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.07, iou_thres=0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = names[int(cls)]
                print(f"📍 Phát hiện: {label} ({conf:.2f})")

    cv2.imshow("Phat hien", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

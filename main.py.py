import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Only include video 1 and 4
video_paths = [
    r"C:\Users\User\OneDrive\Desktop\Ambulance detecting system\video2.mp4",
    r"C:\Users\User\OneDrive\Desktop\Ambulance detecting system\video4.mp4"
]

# Open both video feeds
caps = [cv2.VideoCapture(path) for path in video_paths]

# Get FPS of video4 to calculate 3 sec frame limit
fps_video4 = caps[1].get(cv2.CAP_PROP_FPS)
max_frames_video4 = int(fps_video4 * 3)  # 3 seconds limit

# Flags for ambulance detection
ambulance_detected = [False, False]

def is_red_ambulance(crop):
    """Check if the detected vehicle is mostly red (red ambulance)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    red_ratio = np.sum(red_mask) / (crop.shape[0] * crop.shape[1])  
    return red_ratio > 0.4


def draw_traffic_light(frame, ambulance_green, index):
    x, y, w, h = 20, 20, 40, 120
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)

    centers = [(x + w // 2, y + 20), (x + w // 2, y + 60), (x + w // 2, y + 100)]

    if ambulance_green:
        colors = [(0, 0, 100), (0, 100, 100), (0, 255, 0)]
    else:
        colors = [(0, 0, 255), (0, 255, 255), (0, 100, 0)]

    for c, color in zip(centers, colors):
        cv2.circle(frame, c, 15, color, -1)

    cv2.putText(frame, f"Camera {1 if index == 0 else 4}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


while True:
    frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()

        # 🔁 LOOP video normally if it ends
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        # 🎯 SPECIAL: video4 (index 1) → loop after 3 sec
        if i == 1:
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current_frame >= max_frames_video4:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            results = model(frame)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id]

                    if label.lower() == "truck":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[y1:y2, x1:x2]

                        if crop.size == 0:
                            continue

                        if is_red_ambulance(crop):
                            ambulance_detected[i] = True
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(frame, "Emergency Vehicle", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            draw_traffic_light(frame, ambulance_detected[i], i)

        frame = cv2.resize(frame, (640, 480))
        frames.append(frame)

    combined = np.hstack((frames[0], frames[1]))
    cv2.imshow("Camera 1 & 4 - Ambulance Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()

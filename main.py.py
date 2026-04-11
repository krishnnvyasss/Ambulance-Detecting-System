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

# Flags for ambulance detection
ambulance_detected = [False, False]

def is_red_ambulance(crop):
    """Check if the detected vehicle is mostly red (red ambulance)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Define red color ranges in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    red_ratio = np.sum(red_mask) / (crop.shape[0] * crop.shape[1])  
    return red_ratio > 0.4   # at least 40% red area


def draw_traffic_light(frame, ambulance_green, index):
    """Draw traffic light at top-left corner."""
    x, y, w, h = 20, 20, 40, 120
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)

    centers = [(x + w // 2, y + 20), (x + w // 2, y + 60), (x + w // 2, y + 100)]

    # If ambulance detected → green light ON, else red ON
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
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Run YOLO detection
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
                            cv2.putText(frame, "AMBULANCE", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            draw_traffic_light(frame, ambulance_detected[i], i)

        frame = cv2.resize(frame, (640, 480))
        frames.append(frame)

    # Display both feeds side by side
    combined = np.hstack((frames[0], frames[1]))
    cv2.imshow("Camera 1 & 4 - Ambulance Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()

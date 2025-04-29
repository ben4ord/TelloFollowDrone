import cv2
import time
from djitellopy import Tello
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Initialize Tello ---
tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

# Start video stream (no takeoff)
tello.streamon()
frame_read = tello.get_frame_read()


# Load YOLOv8 model (choose nano for speed)
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' or others if needed


locked_id = None
lock_lost_count = 0
max_lost_frames = 150
tracker = DeepSort(max_age=15, n_init=3)
last_locked_center = None


# Frame dimensions
maxW, maxH = 640, 480
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.namedWindow("Tello Person Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tello Person Detection", maxW, maxH)

print("[INFO] Starting YOLOv8 person detection. Press 'q' to quit.")

while True:
    frame = frame_read.frame
    frame = cv2.resize(frame, (640, 480))

    # --- YOLO person detection ---
    results = model(frame, verbose=False)[0]
    person_boxes = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        if class_id == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            person_boxes.append(([x1, y1, w, h], conf, 'person'))

    # --- Filter detections ---
    filtered_detections = []

    if locked_id is None or last_locked_center is None:
        # Not locked yet — allow all detections
        filtered_detections = person_boxes
    else:
        # Locked: only keep detection closest to last known center
        min_dist = float('inf')
        best_det = None

        for det in person_boxes:
            x, y, w, h = det[0]
            cx, cy = x + w // 2, y + h // 2
            dist = ((cx - last_locked_center[0]) ** 2 + (cy - last_locked_center[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best_det = det

        if best_det:
            filtered_detections = [best_det]

    # --- Update tracker ---
    tracks = tracker.update_tracks(filtered_detections, frame=frame)

    # --- Process tracks ---
    person_found = False
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        if locked_id is None:
            locked_id = track_id
            print(f"[INFO] Locked onto person ID: {locked_id}")

        if track_id == locked_id:
            person_found = True
            lock_lost_count = 0
            last_locked_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Locked Person {track_id}", (x1, y1 - 10), font, 0.6, (255, 255, 255), 2)
            break  # Only display the locked person

    # --- Handle lock loss ---
    if not person_found and locked_id is not None:
        lock_lost_count += 1
        if lock_lost_count > max_lost_frames:
            print("[INFO] Lost lock. Resetting.")
            locked_id = None
            last_locked_center = None
            lock_lost_count = 0

    # --- Show frame ---
    cv2.imshow("Tello Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
tello.streamoff()

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


# Frame dimensions
maxW, maxH = 640, 480
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.namedWindow("Tello Person Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tello Person Detection", maxW, maxH)

print("[INFO] Starting YOLOv8 person detection. Press 'q' to quit.")

while True:
    # --- Get video frame from Tello ---
    frame = frame_read.frame
    frame = cv2.resize(frame, (640, 480))

    # --- Step 1: Detect only if not locked ---
    if locked_id is None:
        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_id == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))
    else:
        detections = []  # Skip detection — use tracking only

    # --- Step 2: Update tracker ---
    tracks = tracker.update_tracks(detections, frame=frame)

    # --- Step 3: Draw only the locked person ---
    person_found = False
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        if locked_id is None:
            locked_id = track_id
            print(f"[INFO] Locked onto person ID: {locked_id}")

        if track_id == locked_id:
            person_found = True
            lock_lost_count = 0
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Locked Person {track_id}", (x1, y1 - 10), font, 0.6, (255, 255, 255), 2)
            break  # Only show locked person

    # --- Step 4: Handle lost lock ---
    if not person_found and locked_id is not None:
        lock_lost_count += 1
        if lock_lost_count > max_lost_frames:
            print("[INFO] Lost lock. Resetting.")
            locked_id = None
            lock_lost_count = 0

    # --- Step 5: Show frame ---
    cv2.imshow("Tello Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
tello.streamoff()
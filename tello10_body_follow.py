import cv2
import time
from djitellopy import Tello
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Initialize and connect to the Tello drone ---
tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

# --- Start the Tello's video stream (but no takeoff) ---
tello.streamon()

frame_read = tello.get_frame_read()

# --- Load YOLOv8 model ('yolov8s.pt') - Built in model
model = YOLO("yolov8n.pt")

# --- Initialize person tracking state ---
locked_id = None                # The ID of the currently locked person
lock_lost_count = 0             # Counts how many frames we've lost the locked person
max_lost_frames = 150           # Number of frames allowed to lose the target before reset

# max_age: Controls how many consecutive frames a track can disappear for (i.e., not be matched with a detection) before it's deleted
# n_init: Specifies how many consecutive detections are required before a track is considered 'confirmed'
tracker = DeepSort(max_age=30, n_init=3)  # Deep SORT tracker with aging tolerance
last_locked_center = None                 # (x, y) center of the last locked position for proximity filtering, set to none since we haven't assigned it yet

# --- Set up display window ---
maxW, maxH = 640, 480             # Frame size
font = cv2.FONT_HERSHEY_SIMPLEX   # Font style for tracking
cv2.namedWindow("Tello Person Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tello Person Detection", maxW, maxH)

print("[INFO] Starting YOLOv8 person detection. Press 'q' to quit.")


fly_flag = True
tello.takeoff()
tello.move_up(75)  # Boost Drone Upwards



# --- Main loop ---
while True:
    # --- Read and resize the current frame from the drone ---
    frame = frame_read.frame
    frame = cv2.resize(frame, (maxW, maxH))

    # --- Run YOLOv8 detection on the frame ---
    results = model(frame, verbose=False)[0]
    person_boxes = []  # Will hold only person detections

    # --- Filter only 'person' class (class ID 0) ---
    # We don't want to detect cars and other things here
    for box in results.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])  # Confidence score for person detection
        if class_id == 0:          # Class 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])                 # Convert box to pixel coords
            w, h = x2 - x1, y2 - y1                                # Get width and height of the bounding box
            person_boxes.append(([x1, y1, w, h], conf, 'person'))  # Format for Deep SORT

    # --- Filter detections to only include the locked person if already tracking ---
    filtered_detections = []

    if locked_id is None or last_locked_center is None:
        # If no one is locked yet, pass all person detections to Deep SORT to find a lock
        filtered_detections = person_boxes
    else:
        # If locked, keep only the detection closest to the last known position of the previous locked person
        min_dist = float('inf')   # Initialize the minimum distance (infinity for now)
        best_det = None           # Initialize the best detection to be used later

        # Loop through all detected person bounding boxes
        for det in person_boxes:
            x, y, w, h = det[0]              # Extract bounding box: top-left corner (x, y), width and height
            cx, cy = x + w // 2, y + h // 2  # Calculate the center of the detection box

            # Compute Euclidean distance from this center to the last locked person's center
            dist = ((cx - last_locked_center[0]) ** 2 + (cy - last_locked_center[1]) ** 2) ** 0.5

            # Keep the detection that is closest to the last known location of the locked person
            if dist < min_dist:
                min_dist = dist
                best_det = det

        # If a closest detection was found, update the filtered list with only that one detection
        if best_det:
            filtered_detections = [best_det]  # Send only the closest person to Deep SORT for tracking

    # --- Pass filtered detections to Deep SORT tracker ---
    tracks = tracker.update_tracks(filtered_detections, frame=frame)


    # --- Look through tracked objects ---
    person_found = False  # Flag to track whether the locked person is currently visible (in frame)

    hV = dV = vV = rV = 0 # Initializing Variables for drone movement control

    # Looping through each currently tracked object
    for track in tracks:
        if not track.is_confirmed():
            continue  # Skip tracks that aren't confirmed yet (may be noise or false positives)

        track_id = track.track_id  # Get the unique ID assigned by Deep SORT to this track
        x1, y1, x2, y2 = map(int, track.to_ltrb())  # Get bounding box coordinates: Left, Top, Right, Bottom

        if locked_id is None:
            # If we haven't locked onto anyone yet, lock onto the first confirmed track
            locked_id = track_id
            print(f"[INFO] Locked onto person ID: {locked_id}")

        if track_id == locked_id:
            # If this is the person we're locked onto:
            person_found = True  # Mark that the locked person is still being tracked
            lock_lost_count = 0  # Reset the lost frame counter since we found them

            # Update the last known center of the locked person to the current center
            last_locked_center = ((x1 + x2) // 2, (y1 + y2) // 2)


            # --- Drone control logic ---
            # Frame center
            frame_cx = maxW // 2
            frame_cy = maxH // 2

            # Bounding box center
            bbox_cx = (x1 + x2) // 2
            bbox_cy = (y1 + y2) // 2
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Horizontal offset (positive = person is to left)
            lrdelta = frame_cx - bbox_cx
            if lrdelta > 0.2 * maxW:
                rV = -60  # Rotate counter-clockwise
            elif lrdelta < -0.2 * maxW:
                rV = 60   # Rotate clockwise

            # Vertical offset (positive = person is too low)
            uddelta = frame_cy - bbox_cy
            if uddelta > 0.2 * maxH:
                vV = 30   # Move up
            elif uddelta < -0.2 * maxH:
                vV = -30  # Move down

            # Distance (bounding box width used as proxy)
            if bbox_w < 100:
                dV = 30   # Move forward
            elif bbox_w > 140:
                dV = -30  # Move backward

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Locked Person {track_id}", (x1, y1 - 10), font, 0.6, (255, 255, 255), 2)
            break  # Only track one person

    # --- Handle case where locked person is not found ---
    if not person_found and locked_id is not None:
        lock_lost_count += 1 # Increment the total number of frames the person is lost for

        if lock_lost_count > max_lost_frames:
            # If we've lost them for too long (frame threshold), reset the locked person to allow a new person to be tracked
            print("[INFO] Lost lock. Resetting.")
            locked_id = None
            last_locked_center = None
            lock_lost_count = 0

    if fly_flag:
        tello.send_rc_control(hV, dV, vV, rV) # Sending control information to the drone so it knows what to do

    # --- Display the processed frame ---
    cv2.imshow("Tello Person Detection", frame)

    # --- Quit on 'q' key ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if fly_flag:
            tello.land() # Not sure if we need this, put it here just in case
        break

# --- Cleanup ---
cv2.destroyAllWindows()
#tello.streamoff()
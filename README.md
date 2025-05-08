Tello Drone Face Recognition & Body Following System
====================================================

Overview:
---------
This project utilizes the Tello drone for autonomous user recognition and tracking using face recognition and YOLOv8 body detection. The system involves three main scripts which must be executed in order to ensure successful operation.

Setup Instructions:
-------------------

1. **Step 1: Collect Face Data**
   Script: `tello_collect_face_data.py`

   - Run this script first to collect face data.
   - You will be prompted to enter a user ID number. This ID will correspond to an index in an array of predefined names.
   - The script will capture **400 images** of your face.
   - Be sure to move your head around — take pictures at **different angles and distances** to improve recognition accuracy.
   - This data will be used for model training.

2. **Step 2: Train the Face Recognition Model**
   Script: `tello_train_face_alg.py`

   - After collecting face data, run this script.
   - It will process and train the face recognition model based on the data you just collected.
   - Ensure training completes successfully before proceeding.

3. **Step 3: Start the Tello Drone**
   Script: `tello10_body_follow2.py`

   - This script powers up the Tello drone and starts the autonomous tracking system.
   - You can face the drone in any direction — it will **spin** to search for a person using **YOLOv8 body detection**.
   - Once a body is detected, it will **approach** and use the face recognition model to verify the identity.
   - If the user is **verified**:
     - The drone will do a **backflip**
     - Then maintain a distance of approximately **10 feet**
     - And begin **following** the user.

Control Commands:
-----------------
- Press **'q'** on the keyboard to **terminate** the program.
- Press **'r'** to **reset** the drone’s target and look for a new person.
- Press **'f'** to make the drone perform a **manual flip**.

Requirements:
-------------
- Tello Drone
- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- deep_sort_realtime DeepSort
- djitellopy (for Tello Drone)


Note:
-----
Make sure the drone is fully charged and in a safe, open area before starting any of the scripts.

Enjoy your autonomous drone experience!

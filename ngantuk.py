import cv2
import mediapipe as mp
import math
import os
from datetime import datetime
import time
import pygame

# Initialize Pygame sound
pygame.init()
pygame.mixer.init()

# Load alert sounds (load once)
yawn_sound = pygame.mixer.Sound("yawn_alert.mp3")
drowsiness_sound = pygame.mixer.Sound("drowsiness_alert.mp3")
distraction_sound = pygame.mixer.Sound("distraction_alert.mp3")
sleep_sound = pygame.mixer.Sound("sleep_alert.mp3")

# Initialize sound channels
yawn_channel = pygame.mixer.Channel(0)
drowsiness_channel = pygame.mixer.Channel(1)
distraction_channel = pygame.mixer.Channel(2)
sleep_channel = pygame.mixer.Channel(3)

# Initialize Mediapipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Set the directory for saving screenshots
output_dir = 'driver_behavior_screenshots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create separate folders for each detector
detector_folders = ['yawn', 'sleep', 'distraction']
for folder in detector_folders:
    folder_path = os.path.join(output_dir, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to detect if eyes are closed and return a confidence score
def eyes_closed_confidence(landmarks):
    left_eye_upper = landmarks[386]  # Upper part of the left eye
    left_eye_lower = landmarks[374]  # Lower part of the left eye
    right_eye_upper = landmarks[159]  # Upper part of the right eye
    right_eye_lower = landmarks[145]  # Lower part of the right eye

    left_eye_distance = calculate_distance(left_eye_upper, left_eye_lower)
    right_eye_distance = calculate_distance(right_eye_upper, right_eye_lower)

    avg_eye_distance = (left_eye_distance + right_eye_distance) / 2
    confidence = max(0, min(100, (0.02 - avg_eye_distance) / 0.02 * 100))

    return confidence

# Function to detect yawning and return a confidence score
def yawn_confidence(landmarks):
    mouth_upper = landmarks[13]  # Upper lip
    mouth_lower = landmarks[14]  # Lower lip
    mouth_distance = calculate_distance(mouth_upper, mouth_lower)

    confidence = max(0, min(100, (mouth_distance - 0.03) / 0.03 * 100))

    return confidence

# Function to detect head pose (looking away) - Simplified approach
def head_pose_confidence(landmarks):
    nose_tip = landmarks[1]
    left_ear = landmarks[234]
    right_ear = landmarks[454]

    ear_distance = calculate_distance(left_ear, right_ear)
    nose_left_ear_distance = calculate_distance(nose_tip, left_ear)
    nose_right_ear_distance = calculate_distance(nose_tip, right_ear)

    # Assuming looking straight ahead is a neutral position
    if nose_left_ear_distance > ear_distance * 1.2:
        return 100  # Looking left
    elif nose_right_ear_distance > ear_distance * 1.2:
        return 100  # Looking right
    else:
        return 0

# Function to save a screenshot with timestamp and label
def save_screenshot(frame, label):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f'{label}_{timestamp}.png'
    filepath = os.path.join(output_dir, label, filename)
    cv2.imwrite(filepath, frame)
    print(f'Screenshot saved: {filepath}')

# Start video capture
cap = cv2.VideoCapture(0)

start_time = time.time()
fps = 0
frame_count = 0

# Drowsiness and Distraction Tracking Variables
drowsiness_start_time = 0
drowsiness_duration = 0
distraction_start_time = 0
distraction_duration = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_result = face_detection.process(rgb_frame)

    if face_result.detections:
        for detection in face_result.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            x2, y2 = int((bboxC.xmin + bboxC.width) * iw), int((bboxC.ymin + bboxC.height) * ih)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            result = face_mesh.process(rgb_frame)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                    yawn_conf = yawn_confidence(landmarks)
                    eyes_conf = eyes_closed_confidence(landmarks)
                    head_pose_conf = head_pose_confidence(landmarks)

                    if yawn_conf > 50:
                        cv2.putText(frame, f"Yawn {yawn_conf:.2f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
                        save_screenshot(frame, "yawn")
                        yawn_channel.play(yawn_sound)  # Play yawn alert sound

                    if eyes_conf > 50:
                        cv2.putText(frame, f"Sleep {eyes_conf:.2f}%", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2)
                        save_screenshot(frame, "sleep")
                        if drowsiness_start_time == 0:
                            drowsiness_start_time = time.time()
                        else:
                            drowsiness_duration = time.time() - drowsiness_start_time
                            if drowsiness_duration > 5:  # 5 seconds of drowsiness
                                drowsiness_channel.play(drowsiness_sound)  # Play drowsiness alert sound
                                drowsiness_start_time = 0  # Reset drowsiness start time
                    else:
                        drowsiness_start_time = 0  # Reset drowsiness start time when eyes are open

                    if head_pose_conf > 50:
                        cv2.putText(frame, f"Distraction {head_pose_conf:.2f}%", (x1, y1 - 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 255), 2)
                        save_screenshot(frame, "distraction")
                        if distraction_start_time == 0:
                            distraction_start_time = time.time()
                        else:
                            distraction_duration = time.time() - distraction_start_time
                            if distraction_duration > 5:  # 5 seconds of distraction
                                distraction_channel.play(distraction_sound)  # Play distraction alert sound

                    awake_conf = 100 - max(yawn_conf, eyes_conf)
                    if awake_conf > 50:
                        cv2.putText(frame, f"Awake {awake_conf:.2f}%", (x1, y1 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 0), 2)
                        # No screenshot for awake

    frame_count += 1
    end_time = time.time()
    if end_time - start_time > 1:
        fps = frame_count / (end_time - start_time)
        frame_count = 0
        start_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('Driver Behavioral System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()

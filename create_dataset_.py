import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

# Create directory to store dataset
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of images per sign
num_images = 200  # Adjust as needed

# Labels
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'BACKSPACE', 'SPACE'
]
label_map = {label: i for i, label in enumerate(labels)}

# Data storage
data = []
labels_list = []

cap = cv2.VideoCapture(0)

for label in labels:
    print(f"Collecting data for {label}. Press 'c' to start...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Ready to collect: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    print(f"Collecting {num_images} images for {label}...")

    collected = 0
    while collected < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # Extract landmarks from first detected hand only
            hand_landmarks = results.multi_hand_landmarks[0]
            data_aux = []
            x_, y_ = [], []

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            if len(data_aux) == 42:  # Ensure 42 features
                data.append(data_aux)
                labels_list.append(label_map[label])
                collected += 1
                print(f"Collected {collected}/{num_images} for {label}", end='\r')

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("\nData collection complete.")

# Save dataset
dataset = {'data': data, 'labels': labels_list}
with open('dataset.p', 'wb') as f:
    pickle.dump(dataset, f)

cap.release()
cv2.destroyAllWindows()

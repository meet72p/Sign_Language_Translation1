import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize camera
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
mp_draw = mp.solutions.drawing_utils  # For drawing hand landmarks

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
    19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: 'BACKSPACE', 25: ' ',
}

print("Recognized Alphabets & Numbers: ", end="", flush=True)

last_prediction_time = time.time()  # Timer to track predictions
predicted_text = ""  # Store full recognized text

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()

    if results.multi_hand_landmarks and (current_time - last_prediction_time >= 2):
        data_aux, x_, y_ = [], [], []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        # Ensure correct input shape
        data_aux = np.array(data_aux).reshape(1, -1)

        # Make a prediction every 2 seconds
        predicted_index = int(model.predict(data_aux)[0])
        predicted_character = labels_dict.get(predicted_index, '?')  # Handle unknown predictions

        if predicted_character == 'BACKSPACE':
            if predicted_text:
                predicted_text = predicted_text[:-1]  # Remove last character
                print("\b \b", end="", flush=True)  # Remove from terminal
        elif predicted_character == ' ':
            predicted_text += ' '  # Add space
            print(" ", end="", flush=True)
        else:
            predicted_text += predicted_character
            print(predicted_character, end="", flush=True)

        last_prediction_time = current_time

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))



    # Show camera feed
    cv2.imshow("Camera Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

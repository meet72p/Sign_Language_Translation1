from flask import Flask, render_template, Response, jsonify
import pickle, cv2, mediapipe as mp, numpy as np, time

app = Flask(__name__)

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = None  # Camera capture object
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Utility to draw landmarks
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 
               11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 
               21: 'W', 22: 'X', 23: 'Y', 24: 'BACKSPACE', 25: ' '}  # 24 represents backspace, 25 is space

recognized_text = ""  # Store recognized text

def generate_frames():
    global cap
    cap = cv2.VideoCapture(0)
    last_prediction_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            current_time = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            global recognized_text
            if results.multi_hand_landmarks and (current_time - last_prediction_time >= 2):
                last_prediction_time = current_time
                data_aux, x_, y_ = [], [], []

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    for i in range(len(hand_landmarks.landmark)):
                        x_.append(hand_landmarks.landmark[i].x)
                        y_.append(hand_landmarks.landmark[i].y)
                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                        data_aux.append(hand_landmarks.landmark[i].y - min(y_))
                
                data_aux = np.array(data_aux).reshape(1, -1)
                predicted_index = int(model.predict(data_aux)[0])
                predicted_character = labels_dict.get(predicted_index, '?')
                
                if predicted_character == "BACKSPACE" and recognized_text:
                    recognized_text = recognized_text[:-1]
                elif predicted_character == " ":
                    recognized_text += " "
                else:
                    recognized_text += predicted_character

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognized_text')
def get_recognized_text():
    return jsonify({"recognized_text": recognized_text})

@app.route('/clear')
def clear_text():
    global recognized_text
    recognized_text = ""
    return jsonify({"status": "Text cleared"})

if __name__ == "__main__":
    app.run(debug=True)

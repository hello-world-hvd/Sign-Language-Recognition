from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import collections
import os

app = Flask(__name__)

# Load model và các thông số
model = tf.keras.models.load_model('../model/model_tf3.keras')
mean = np.load('../scripts/mean.npy').reshape(-1)
std = np.load('../scripts/std.npy').reshape(-1)
gesture_labels = sorted(os.listdir('../dataset_try'))
sequence_length = 60

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Buffer để lưu sequence
sequence_buffer = collections.deque(maxlen=sequence_length)

current_mode = 'test'  # or 'text'
text_buffer = ""  # Đổi từ list sang string

camera_active = False
cap = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/switch_mode/<mode>')
def switch_mode(mode):
    global current_mode
    if mode in ['test', 'text']:
        current_mode = mode
    return jsonify({'status': 'success'})

@app.route('/get_text')
def get_text():
    return jsonify({'text': text_buffer})

@app.route('/start_camera')
def start_camera():
    global camera_active, cap
    if not camera_active:
        cap = cv2.VideoCapture(0)
        camera_active = True
    return jsonify({'status': 'success'})

@app.route('/stop_camera')
def stop_camera():
    global camera_active, cap
    if camera_active:
        cap.release()
        camera_active = False
    return jsonify({'status': 'success'})

def generate_frames():
    global camera_active, cap, text_buffer
    while True:
        if not camera_active:
            # Trả về frame trống hoặc hình ảnh mặc định
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', empty_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue
            
        ret, frame = cap.read()
        if not ret:
            break
            
        # Xử lý frame
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Vẽ landmarks
            mp_drawing.draw_landmarks(
                frame, 
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS
            )
            
            # Trích xuất keypoints
            landmark_row = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmark_row.extend([lm.x, lm.y])
            landmark_row = (np.array(landmark_row) - mean) / std
            sequence_buffer.append(landmark_row)
        else:
            # Không có tay trong frame
            zero_norm = (np.zeros(42) - mean) / std
            sequence_buffer.append(zero_norm)
            
        # Dự đoán nếu đủ frames
        prediction = ""
        if len(sequence_buffer) == sequence_length:
            input_data = np.expand_dims(np.array(sequence_buffer), axis=0)
            preds = model.predict(input_data, verbose=0)
            class_id = np.argmax(preds[0])
            confidence = preds[0][class_id]
            
            if confidence > 0.7:  # Ngưỡng tin cậy
                prediction_label = gesture_labels[class_id]
                prediction = f"{prediction_label} ({confidence:.2f})"
                
                if current_mode == 'text':
                    # Xử lý nối chuỗi văn bản
                    if prediction_label == 'space':
                        text_buffer += ' '
                    elif prediction_label == 'delete':
                        text_buffer = text_buffer[:-1]
                    else:
                        text_buffer += prediction_label
                
                # Vẽ text lên frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(pil_img)
                font = ImageFont.truetype("arial.ttf", 36)
                draw.text((10, 40), prediction, font=font, fill=(0, 255, 0))
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Chuyển frame thành jpg để stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
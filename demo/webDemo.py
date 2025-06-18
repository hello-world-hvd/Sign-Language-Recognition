import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image

# Load model và mean/std
model = tf.keras.models.load_model('../model/model_tf3.keras')
mean = np.load('mean.npy').reshape(-1)
std = np.load('std.npy').reshape(-1)
gesture_labels = sorted(os.listdir('../dataset_try'))
sequence_length = 30

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

st.title("Demo nhận diện thủ ngữ bằng webcam")

run = st.checkbox('Bật camera')
FRAME_WINDOW = st.image([])

sequence_buffer = []

if run:
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Không lấy được frame từ camera.")
            break
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            landmark_row = []
            for lm in hand_landmarks.landmark:
                landmark_row.append(lm.x)
                landmark_row.append(lm.y)
            landmark_row = (np.array(landmark_row) - mean) / std
            sequence_buffer.append(landmark_row)
            if len(sequence_buffer) > sequence_length:
                sequence_buffer.pop(0)
        else:
            # Nếu không phát hiện tay, thêm vector zero
            zero_norm = (np.zeros(42) - mean) / std
            sequence_buffer.append(zero_norm)
            if len(sequence_buffer) > sequence_length:
                sequence_buffer.pop(0)

        # Dự đoán nếu đủ sequence
        if len(sequence_buffer) == sequence_length:
            input_data = np.expand_dims(np.array(sequence_buffer), axis=0)
            preds = model.predict(input_data, verbose=0)
            class_id = np.argmax(preds[0])
            class_name = gesture_labels[class_id]
            confidence = preds[0][class_id]
            text = f'{class_name} ({confidence*100:.1f}%)'

            # Vẽ text tiếng Việt bằng PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.truetype("arial.ttf", 36)
            draw.text((10, 40), text, font=font, fill=(0, 255, 0))
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        FRAME_WINDOW.image(frame, channels="BGR")
    cap.release()
else:
    st.write('Tắt camera')
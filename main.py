import cv2
import mediapipe as mp
from hand_model_trainer import HandModelNN
import torch
import json
from asl_hand_model.enums import NUM_CLASSES
from hand_model_trainer import flatten_landmarks_2
from flask import Flask, render_template, Response, jsonify
import threading
import gc
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Hands and drawing utilities.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HandModelNN(input_dim=63, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("model.pth", map_location=torch.device(device)))
model.eval()

with open('labels.json', 'r') as f:
    labels = json.load(f)

latest_prediction = None
prediction_lock = threading.Lock()
crop_size = 600

def generate_frames():
    global latest_prediction
    cap = cv2.VideoCapture(0)
    # Set up the MediaPipe Hands solution.
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h_frame, w_frame, _ = frame.shape
            # Ensure the frame is large enough to extract a 600x600 crop.
            if w_frame >= crop_size and h_frame >= crop_size:
                # Calculate the coordinates to crop the center.
                x_start = (w_frame - crop_size) // 2
                y_start = (h_frame - crop_size) // 2
                # Crop the center 600x600 region.
                frame = frame[y_start:y_start + crop_size, x_start:x_start + crop_size]
            else:
                # Optionally handle cases where the frame is too small.
                # For now, we'll just resize (or you could continue without cropping).
                frame = cv2.resize(frame, (crop_size, crop_size))

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Process the image to detect hands.
            results = hands.process(image_rgb)

            # Convert back to BGR for OpenCV.
            output_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # If hands are detected, draw bounding boxes.
            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                for hand_landmarks in results.multi_hand_landmarks:
                    # Convert landmark positions to pixel coordinates.
                    landmark_points = [
                        (int(lm.x * w), int(lm.y * h))
                        for lm in hand_landmarks.landmark
                    ]

                    flat_landmarks = flatten_landmarks_2(hand_landmarks.landmark)
                    input_tensor = torch.Tensor(flat_landmarks).unsqueeze(0)
                    with torch.no_grad():
                        logits = model(input_tensor)
                        prediction = torch.argmax(logits, dim=1).item()

                    with prediction_lock:
                        latest_prediction = labels[prediction]

                    # Calculate bounding box coordinates based on landmarks.
                    xs = [pt[0] for pt in landmark_points]
                    ys = [pt[1] for pt in landmark_points]
                    x_min, y_min = min(xs), min(ys)
                    x_max, y_max = max(xs), max(ys)

                    # Draw a rectangle (bounding box) around the hand.
                    cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    cv2.putText(
                        output_image,  # The image to draw on.
                        labels[prediction],  # The prediction text.
                        (x_min, y_min - 10),  # Position the text above the bounding box.
                        cv2.FONT_HERSHEY_SIMPLEX,  # Font type.
                        0.9,  # Font scale (size).
                        (0, 255, 0),  # Text color (green, same as the rectangle).
                        2,  # Thickness of the text.
                        cv2.LINE_AA  # Line type for better aesthetics.
                    )

                    mp_drawing.draw_landmarks(
                        output_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            ret, buffer = cv2.imencode('.jpg', output_image)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            gc.collect()
    cap.release()

@app.route('/')
def index():
    # Render the HTML page.
    return render_template('main.html')

@app.route('/video_feed')
def video_feed():
    # Return the streaming response.
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/get_prediction')
def get_prediction_api():
    """Endpoint to return the latest prediction."""
    with prediction_lock:
        prediction = latest_prediction
    return jsonify({'prediction': prediction})
if __name__ == '__main__':
    app.run(debug=True, port=5001)



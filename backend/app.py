# Flask API for Emotion Recognition

from flask import Flask, request, jsonify, send_from_directory  # Import Flask core classes
from flask_cors import CORS                                      # Import CORS to allow frontend requests
import cv2                                                       # OpenCV for image processing
import numpy as np                                               # NumPy for array operations
import pandas as pd                                              # Pandas for CSV logging
from datetime import datetime                                     # For timestamps
import base64                                                    # For decoding base64 images
import pickle                                                    # To load class labels
import tensorflow as tf                                          # TensorFlow for model handling
from tensorflow.keras.models import Sequential                    # Sequential model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization  # Layers
from tensorflow.keras import regularizers                        # L2 regularization
from tensorflow.keras.applications import EfficientNetB4          # Pretrained EfficientNetB4

app = Flask(__name__, static_folder='../frontend')               # Create Flask app with static folder
CORS(app)                                                        # Enable CORS for all routes

# Load Model

print(" Loading model...")                                      # Notify model loading

base_model = EfficientNetB4(                                     # Load EfficientNetB4 base
    include_top=False,                                           # Remove top layer
    weights='imagenet',                                          # Load pretrained weights
    input_shape=(224, 224, 3)                                    # Input size for images
)

base_model.trainable = True                                      # Enable training on some layers

fine_tune_at = len(base_model.layers) - 150                      # Unfreeze last 150 layers

for layer in base_model.layers[:fine_tune_at]:                   # Freeze layers before cut point
    layer.trainable = False                                      # Set layer as non-trainable

model = Sequential([                                             # Build the full classification model
    base_model,                                                  # Include EfficientNetB4
    GlobalAveragePooling2D(),                                    # Reduce spatial dimensions
    BatchNormalization(),                                        # Normalize activations
    Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # Dense layer
    Dropout(0.5),                                                # Prevent overfitting
    BatchNormalization(),                                        # Normalize
    Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),   # Dense layer
    Dropout(0.5),                                                # Dropout
    BatchNormalization(),                                        # Normalize
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),   # Dense
    Dropout(0.4),                                                # Dropout
    BatchNormalization(),                                        # Normalize
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),   # Dense
    Dropout(0.3),                                                # Dropout
    Dense(6, activation='softmax')                               # Output layer for 6 emotions
])

model.load_weights('final_emotion_model_weights.h5')             # Load trained weights

with open('class_labels.pkl', 'rb') as f:                        # Open class label file
    class_labels_dict = pickle.load(f)                           # Load dictionary
    class_labels = class_labels_dict['classes']                  # Extract classes list

print(f"✓ Model loaded! Classes: {class_labels}")                # Print loaded classes

face_cascade = cv2.CascadeClassifier(                            # Load OpenCV face detector
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

csv_filename = 'emotion_predictions_log.csv'                     # CSV log file name

# Helper Functions

def save_to_csv(emotion, confidence, source):                    # Save prediction to CSV
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')     # Get current timestamp
    data = {                                                     # Create row dictionary
        'Timestamp': [timestamp],
        'Emotion': [emotion],
        'Confidence': [confidence],
        'Source': [source]
    }
    df = pd.DataFrame(data)                                      # Convert to DataFrame

    try:
        existing_df = pd.read_csv(csv_filename)                  # Try reading existing file
        df = pd.concat([existing_df, df], ignore_index=True)     # Append old + new data
    except FileNotFoundError:
        pass                                                     # If file missing → create new

    df.to_csv(csv_filename, index=False)                         # Save updated CSV


def predict_emotion(face_img):                                   # Predict emotion from face image
    face_resized = cv2.resize(face_img, (224, 224))              # Resize to model size
    face_resized = face_resized.astype('float32') / 255.0        # Normalize pixels
    face_resized = np.expand_dims(face_resized, axis=0)          # Add batch dimension

    predictions = model.predict(face_resized, verbose=0)         # Get prediction
    emotion_idx = np.argmax(predictions[0])                      # Highest probability class
    confidence = float(predictions[0][emotion_idx])              # Probability
    emotion = class_labels[emotion_idx]                          # Class label text

    return emotion, confidence                                   # Return both values


def process_image(img_array):                                    # Detect faces + predict emotions
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)           # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)          # Detect faces

    results = []                                                 # List for API output

    for (x, y, w, h) in faces:                                   # Loop through detected faces
        face = img_array[y:y+h, x:x+w]                           # Crop face from frame
        emotion, confidence = predict_emotion(face)              # Predict emotion

        save_to_csv(emotion, confidence, 'Web App')              # Log to CSV

        cv2.rectangle(img_array, (x, y), (x+w, y+h),             # Draw box around face
                      (0, 255, 0), 2)

        label = f"{emotion} ({confidence*100:.1f}%)"             # Prepare label text

        cv2.putText(img_array, label, (x, y-10),                 # Draw text above face
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        results.append({                                         # Append API result
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'emotion': emotion,
            'confidence': round(confidence * 100, 2)
        })

    return img_array, results                                    # Return updated image + data

# API Endpoints

@app.route('/')                                                  # Root route
def index():                                                     # Serve frontend main page
    return send_from_directory('../frontend', 'index.html')


@app.route('/<path:path>')                                      # Serve static files
def serve_static(path):
    return send_from_directory('../frontend', path)


@app.route('/api/predict', methods=['POST'])                    # Image prediction endpoint
def predict():
    try:
        data = request.json                                      # Read JSON request
        img_data = data['image'].split(',')[1]                   # Remove base64 header
        img_bytes = base64.b64decode(img_data)                   # Decode base64 → bytes
        nparr = np.frombuffer(img_bytes, np.uint8)               # Convert bytes → NumPy array
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)              # Decode image using OpenCV

        processed_img, results = process_image(img)              # Detect + predict

        _, buffer = cv2.imencode('.jpg', processed_img)          # Encode processed image
        img_base64 = base64.b64encode(buffer).decode('utf-8')    # Convert back to base64

        return jsonify({                                         # Send JSON response
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'faces': results,
            'count': len(results)
        })

    except Exception as e:                                       # Handle errors
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/stats', methods=['GET'])                        # Statistics endpoint
def get_stats():
    try:
        df = pd.read_csv(csv_filename)                           # Read CSV file

        emotion_counts = df['Emotion'].value_counts().to_dict()  # Count each emotion
        total = len(df)                                          # Total predictions
        avg_confidence = df['Confidence'].astype(float).mean()    # Average confidence

        return jsonify({                                          # Return stats
            'success': True,
            'total': total,
            'emotions': emotion_counts,
            'avg_confidence': round(avg_confidence, 2)
        })

    except FileNotFoundError:                                     # If no file yet
        return jsonify({
            'success': True,
            'total': 0,
            'emotions': {},
            'avg_confidence': 0
        })


@app.route('/api/data', methods=['GET'])                         # Return full CSV log
def get_data():
    try:
        df = pd.read_csv(csv_filename)                           # Read file
        return jsonify({
            'success': True,
            'data': df.to_dict('records')
        })
    except FileNotFoundError:                                     # No file yet
        return jsonify({'success': True, 'data': []})


@app.route('/api/clear', methods=['POST'])                       # Clear log file
def clear_data():
    try:
        import os                                                 # Import OS
        os.remove(csv_filename)                                   # Delete CSV file
        return jsonify({'success': True, 'message': 'Data cleared'})
    except FileNotFoundError:                                     # File not found
        return jsonify({'success': True, 'message': 'No data to clear'})

# Run Server

if __name__ == '__main__':                                       # Run app only if executed directly
    print("\ Starting Flask server...")                        # Start notification
    print(" Open: http://localhost:5000")                      # Localhost URL
    app.run(debug=True, host='0.0.0.0', port=5000)               # Start Flask server

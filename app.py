from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import os
from pydub import AudioSegment
import traceback
import ffmpeg

from keras.models import load_model
from voice_analysis import analyze_voice

# Point PyDub to ffmpeg
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

app = Flask(__name__)
CORS(app)

# ============================================
# LOAD DEEPFAKE DETECTION MODEL
# ============================================
MODEL_PATH = r"saved_models\SceneFake_CNN_SMOTE.h5"
model = load_model(MODEL_PATH)

label_map = {"real": 0, "fake": 1}
inv_label_map = {0: "real", 1: "fake"}

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================
# MFCC Extraction (40-dim)
# =============================
def extract_mfcc(filepath, n_mfcc=40):
    y, sr = librosa.load(filepath, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)  # (40,)
    return mfcc_mean

user_data = {}

@app.route('/health-check', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200


@app.route('/submit-form', methods=['POST'])
def handle_form():
    data = request.json
    name = data.get('name')
    age = data.get('age')
    user_data['form'] = {'name': name, 'age': age}
    return jsonify({'status': 'success'})


# ===========================
# /upload-voice
# - saves input_audio.webm & input_audio.wav
# - runs voice_analysis for dashboard
# ===========================
@app.route('/upload-voice', methods=['POST'])
def handle_voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio received'}), 400

    try:
        audio_file = request.files['audio']

        webm_path = os.path.join(UPLOAD_FOLDER, 'input_audio.webm')
        wav_path = os.path.join(UPLOAD_FOLDER, 'input_audio.wav')

        # Save original upload
        audio_file.save(webm_path)

        # Convert to WAV
        ffmpeg.input(webm_path).output(wav_path).run(overwrite_output=True)

        # Analyze for dashboard
        features = analyze_voice(wav_path)
        user_data['voice'] = features

        return jsonify({'status': 'success'})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Audio processing failed: {str(e)}'}), 500


@app.route('/analyze-latest-voice', methods=['GET'])
def get_latest_voice_analysis():
    if 'voice' in user_data:
        return jsonify({'features': user_data['voice']})
    return jsonify({'error': 'No voice data found'}), 404


# ============================================================
# REAL / FAKE AUDIO PREDICTION USING CNN MODEL
# ============================================================
@app.route('/predict-deepfake', methods=['POST'])
def predict_deepfake():
    try:
        use_saved = request.form.get("use_saved")
        print("use_saved =", use_saved)

        # 1) CASE A: use_saved == "true" (called from Processing.jsx)
        if use_saved == "true":
            wav_path = os.path.join(UPLOAD_FOLDER, "input_audio.wav")

            if not os.path.exists(wav_path):
                return jsonify({
                    "error": "No saved audio found. Please record/upload again."
                }), 400

        # 2) CASE B: direct upload to /predict-deepfake (not used now, but supported)
        else:
            if 'audio' not in request.files:
                return jsonify({"error": "No audio file uploaded"}), 400

            audio_file = request.files['audio']
            webm_path = os.path.join(UPLOAD_FOLDER, "predict_audio.webm")
            wav_path = os.path.join(UPLOAD_FOLDER, "predict_audio.wav")

            audio_file.save(webm_path)
            ffmpeg.input(webm_path).output(wav_path).run(overwrite_output=True)

        # ---- Extract MFCC & Predict ----
        mfcc = extract_mfcc(wav_path)
        mfcc = np.expand_dims(mfcc, axis=0)   # (1, 40)
        mfcc = np.expand_dims(mfcc, axis=2)   # (1, 40, 1)

        proba = model.predict(mfcc)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(np.max(proba))
        label = inv_label_map[pred_class]

        response = {
            "prediction": label,
            "confidence": confidence,
            "probabilities": {
                "real": float(proba[0]),
                "fake": float(proba[1])
            }
        }

        print("PREDICTION RESPONSE:", response)
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)

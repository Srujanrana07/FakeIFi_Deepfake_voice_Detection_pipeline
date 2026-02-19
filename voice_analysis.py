# voice_analysis.py
import librosa
import numpy as np
import json

def analyze_voice(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # --- Waveform (downsample for frontend)
        waveform = y[::len(y)//1000].tolist() if len(y) > 1000 else y.tolist()

        # --- Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        spectrogram = S_db.tolist()  # <-- KEEP FULL MATRIX


        # --- Pitch Curve
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_curve = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch_curve.append(pitches[index, i])
        pitch_curve = [float(p) for p in pitch_curve if p > 0][:100]  # limit to 100 points

        # --- Speech Rate Estimation (approximate)
        intervals = librosa.effects.split(y, top_db=30)
        duration = librosa.get_duration(y=y, sr=sr)
        spoken_duration = sum((end - start) / sr for start, end in intervals)
        words_per_minute = (spoken_duration / duration) * 160  # Assuming 160 words per minute normal speech
        syllables_per_second = words_per_minute / 60 * 1.5  # Rough approx

        # --- Silence Percentage
        silence_percentage = 100 - (spoken_duration / duration) * 100

        # --- MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_means = mfcc.mean(axis=1).tolist()

        # --- Dummy Emotion Analysis (placeholder for ML model)
        emotion = {
            "type": "Confident",  # Could be improved with real model
            "confidence": 75
        }

        return {
            "waveform": waveform,
            "spectrogram": spectrogram,
            "pitchCurve": pitch_curve,
            "speechRate": {
                "wordsPerMinute": round(words_per_minute, 2),
                "syllablesPerSecond": round(syllables_per_second, 2)
            },
            "silencePercentage": round(silence_percentage, 2),
            "emotion": emotion,
            "mfccCoefficients": [round(coeff, 2) for coeff in mfcc_means]
        }

    except Exception as e:
        print(f"Voice analysis failed: {e}")
        return {"error": str(e)}

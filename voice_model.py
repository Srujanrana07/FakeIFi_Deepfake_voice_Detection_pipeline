import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import sounddevice as sd
import soundfile as sf

class VoiceSentimentPredictor:
    def __init__(self, model_path, sample_rate=22050, duration=4):
        """
        Initialize predictor with pre-trained model.
        
        Args:
            model_path: Path to your trained .h5 model file
            sample_rate: Target sample rate (must match training)
            duration: Expected audio duration in seconds (must match training)
        """
        self.model = load_model(model_path)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = 40  # Typical value, adjust if your model used different
        
        # These should match your training data labels
        self.emotion_labels = [
            'neutral', 'calm', 'happy', 'sad', 
            'angry', 'fearful', 'disgust', 'surprised'
        ]

    def _preprocess_audio(self, audio_data):
        """Internal preprocessing of audio data"""
        # Ensure correct length
        if len(audio_data) > self.sample_rate * self.duration:
            audio_data = audio_data[:self.sample_rate * self.duration]
        else:
            padding = self.sample_rate * self.duration - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), mode='constant')
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc
        )
        
        # Normalize and reshape for model input
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch and channel dims
        
        return mfcc

    def predict_from_file(self, wav_path):
        """
        Predict emotion from WAV file.
        
        Args:
            wav_path: Path to WAV audio file
            
        Returns:
            Dictionary with emotion probabilities
        """
        # Load audio file
        y, sr = librosa.load(wav_path, sr=self.sample_rate)
        
        # Preprocess and predict
        features = self._preprocess_audio(y)
        predictions = self.model.predict(features)[0]
        
        # Return probabilities for all emotions
        return {
            emotion: float(prob) 
            for emotion, prob in zip(self.emotion_labels, predictions)
        }

    def predict_from_microphone(self, record_seconds=4, save_path=None):
        """
        Record audio and predict emotion.
        
        Args:
            record_seconds: Duration to record
            save_path: Optional path to save recording
            
        Returns:
            Dictionary with emotion probabilities
        """
        print(f"Recording for {record_seconds} seconds...")
        recording = sd.rec(
            int(record_seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()  # Wait until recording is finished
        
        if save_path:
            sf.write(save_path, recording, self.sample_rate)
            print(f"Recording saved to {save_path}")
        
        return self.predict_from_array(recording.flatten())

    def predict_from_array(self, audio_array):
        """
        Predict emotion from numpy array of audio data.
        
        Args:
            audio_array: 1D numpy array of audio samples
            
        Returns:
            Dictionary with emotion probabilities
        """
        features = self._preprocess_audio(audio_array)
        predictions = self.model.predict(features)[0]
        
        return {
            emotion: float(prob)
            for emotion, prob in zip(self.emotion_labels, predictions)
        }


# Example Usage
if __name__ == "__main__":
    # Initialize with your trained model
    predictor = VoiceSentimentPredictor(model_path="your_trained_model.h5")
    
    # Option 1: Predict from WAV file
    results = predictor.predict_from_file("user_audio.wav")
    print("Emotion probabilities from file:")
    for emotion, prob in results.items():
        print(f"{emotion}: {prob:.2%}")
    
    # Option 2: Record from microphone and predict
    # results = predictor.predict_from_microphone(record_seconds=4, save_path="recording.wav")
    # print("\nEmotion probabilities from recording:")
    # for emotion, prob in results.items():
    #     print(f"{emotion}: {prob:.2%}")
    
    # Get the emotion with highest probability
    predicted_emotion = max(results.items(), key=lambda x: x[1])[0]
    print(f"\nPredicted emotion: {predicted_emotion}")
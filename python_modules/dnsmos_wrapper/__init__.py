#!/usr/bin/env python3
"""
DNSMOS wrapper for Julia integration
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
from pathlib import Path

# Add the DNSMOS directory to the path
current_dir = Path(__file__).parent
# Use the DNSMOS submodule
dnsmos_submodule_dir = current_dir.parent / "DNSMOS" / "DNSMOS"
if dnsmos_submodule_dir.exists():
    sys.path.insert(0, str(dnsmos_submodule_dir))

# Import the DNSMOS local module
try:
    from dnsmos_local import ComputeScore
except ImportError as e:
    raise ImportError(f"Could not import DNSMOS. Make sure git submodules are initialized: {e}")

class DNSMOSWrapper:
    def __init__(self):
        """Initialize DNSMOS wrapper"""
        # Set up model paths using the submodule
        current_dir = Path(__file__).parent
        dnsmos_models_dir = current_dir.parent / "DNSMOS" / "DNSMOS" / "DNSMOS"
        
        primary_model_path = str(dnsmos_models_dir / "sig_bak_ovr.onnx")
        p808_model_path = str(dnsmos_models_dir / "model_v8.onnx")
        
        self.compute_score = ComputeScore(primary_model_path, p808_model_path)
        
    def calculate_dnsmos(self, audio_path):
        """
        Calculate DNSMOS scores for an audio file
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary containing SIG, BAK, and OVRL scores
        """
        try:
            # Load audio file
            audio, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Calculate DNSMOS scores
            # The ComputeScore class expects a file path, so we need to save the audio temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio, 16000)
                clip_dict = self.compute_score(tmp_file.name, 16000, False)
                os.unlink(tmp_file.name)  # Delete temporary file
            
            # Return all three P.835 scores
            scores = {
                'SIG': float(clip_dict['SIG']),
                'BAK': float(clip_dict['BAK']),
                'OVRL': float(clip_dict['OVRL'])
            }
            
            return scores
            
        except Exception as e:
            print(f"Error calculating DNSMOS: {e}")
            return None
    
    def calculate_dnsmos_from_array(self, audio_array, sample_rate=16000):
        """
        Calculate DNSMOS scores from audio array
        
        Args:
            audio_array (numpy.ndarray): Audio samples
            sample_rate (int): Sample rate of the audio
            
        Returns:
            dict: Dictionary containing SIG, BAK, and OVRL scores
        """
        try:
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
            # Calculate DNSMOS scores
            # The ComputeScore class expects a file path, so we need to save the audio temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_array, 16000)
                clip_dict = self.compute_score(tmp_file.name, 16000, False)
                os.unlink(tmp_file.name)  # Delete temporary file
            
            # Return all three P.835 scores
            scores = {
                'SIG': float(clip_dict['SIG']),
                'BAK': float(clip_dict['BAK']),
                'OVRL': float(clip_dict['OVRL'])
            }
            
            return scores
            
        except Exception as e:
            print(f"Error calculating DNSMOS: {e}")
            return None

# Create a global instance
dnsmos_wrapper = DNSMOSWrapper()

def dnsmos(audio_path_or_array, sample_rate=16000):
    """
    Main function to calculate DNSMOS scores
    
    Args:
        audio_path_or_array: Either a file path (str) or audio array (numpy.ndarray)
        sample_rate (int): Sample rate (only used if audio_array is provided)
        
    Returns:
        dict: Dictionary containing SIG, BAK, and OVRL scores
    """
    if isinstance(audio_path_or_array, str):
        return dnsmos_wrapper.calculate_dnsmos(audio_path_or_array)
    else:
        return dnsmos_wrapper.calculate_dnsmos_from_array(audio_path_or_array, sample_rate)

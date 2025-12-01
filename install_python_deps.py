#!/usr/bin/env python3
"""
HADatasets Python Dependencies Installation Script

This script installs all required Python packages for the HADatasets metrics module,
including PESQ and DNSMOS implementations.

Requirements:
- Python 3.7+
- pip or conda
- Internet connection for downloading packages

Usage:
    python install_python_deps.py [--conda] [--force] [--verbose]
"""

import os
import sys
import subprocess
import shutil
import urllib.request
import zipfile
import tempfile
from pathlib import Path

def print_status(message, status="INFO"):
    """Print a formatted status message."""
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "RESET": "\033[0m"     # Reset
    }
    print(f"{colors.get(status, '')}[{status}]{colors['RESET']} {message}")

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print_status("Python 3.7+ is required", "ERROR")
        sys.exit(1)
    print_status(f"Python {sys.version_info.major}.{sys.version_info.minor} detected", "SUCCESS")

def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_conda():
    """Check if conda is available."""
    try:
        subprocess.run(["conda", "--version"], 
                      check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_package(package, use_conda=False, force=False, verbose=False):
    """Install a Python package using pip or conda."""
    if use_conda:
        cmd = ["conda", "install", "-y", "-c", "conda-forge", package]
    else:
        cmd = [sys.executable, "-m", "pip", "install"]
        if force:
            cmd.append("--force-reinstall")
        cmd.append(package)
    
    if verbose:
        print_status(f"Running: {' '.join(cmd)}", "INFO")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=not verbose)
        print_status(f"Successfully installed {package}", "SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to install {package}: {e}", "ERROR")
        if not verbose and e.stderr:
            print(f"Error details: {e.stderr.decode()}")
        return False

def check_dnsmos_submodule():
    """Check if DNSMOS submodule is properly set up."""
    print_status("Checking DNSMOS submodule...", "INFO")
    
    # Check if DNSMOS submodule exists
    dnsmos_dir = Path("python_modules/DNSMOS")
    if not dnsmos_dir.exists():
        print_status("DNSMOS submodule not found. Please run: ./setup_submodules.sh", "ERROR")
        return False
    
    # Check if it's a proper git submodule
    git_dir = dnsmos_dir / ".git"
    if not git_dir.exists():
        print_status("DNSMOS directory exists but is not a git submodule", "WARNING")
        print_status("Please run: git submodule update --init --recursive", "INFO")
        return False
    
    print_status("DNSMOS submodule is properly set up", "SUCCESS")
    return True

def create_dnsmos_wrapper():
    """Create a Python wrapper for DNSMOS."""
    wrapper_content = '''"""
DNSMOS Python Wrapper

This module provides a Python interface to Microsoft's DNSMOS implementation
for speech quality assessment.

Reference:
Reddy, C. K. A., Gopal, V., & Cutler, R. (2022). DNSMOS P.835: A non-intrusive 
perceptual objective speech quality metric to evaluate noise suppressors. 
ICASSP 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE.
"""

import os
import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Add DNSMOS to the path
current_dir = Path(__file__).parent
dnsmos_dir = current_dir / "DNSMOS"
if dnsmos_dir.exists():
    sys.path.insert(0, str(dnsmos_dir))

try:
    from DNSMOS.dnsmos_local import DNSMOS
except ImportError:
    print("Warning: DNSMOS not found. Please ensure DNSMOS is properly installed.")
    DNSMOS = None

class DNSMOSWrapper:
    """Wrapper for DNSMOS speech quality assessment."""
    
    def __init__(self):
        """Initialize DNSMOS wrapper."""
        if DNSMOS is None:
            raise ImportError("DNSMOS not available. Please install DNSMOS first.")
        
        self.dnsmos = DNSMOS()
        self.initialized = False
        
    def _initialize(self):
        """Initialize DNSMOS models."""
        if not self.initialized:
            try:
                # Initialize the DNSMOS models
                self.dnsmos.initialize()
                self.initialized = True
            except Exception as e:
                raise RuntimeError(f"Failed to initialize DNSMOS: {e}")
    
    def dnsmos(self, audio, sample_rate):
        """
        Calculate DNSMOS score for audio.
        
        Args:
            audio (np.ndarray): Audio signal (mono, float32)
            sample_rate (int): Sample rate in Hz
            
        Returns:
            float: DNSMOS score (1-5 scale, higher is better)
        """
        self._initialize()
        
        # Ensure audio is the right format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Calculate DNSMOS score
        try:
            score = self.dnsmos.score(audio, sample_rate)
            return float(score)
        except Exception as e:
            raise RuntimeError(f"DNSMOS calculation failed: {e}")

# Global DNSMOS instance
_dnsmos_instance = None

def dnsmos(audio, sample_rate):
    """
    Calculate DNSMOS score for audio.
    
    Args:
        audio (np.ndarray): Audio signal (mono, float32)
        sample_rate (int): Sample rate in Hz
        
    Returns:
        float: DNSMOS score (1-5 scale, higher is better)
    """
    global _dnsmos_instance
    
    if _dnsmos_instance is None:
        try:
            _dnsmos_instance = DNSMOSWrapper()
        except Exception as e:
            raise ImportError(f"Failed to initialize DNSMOS: {e}")
    
    return _dnsmos_instance.dnsmos(audio, sample_rate)

if __name__ == "__main__":
    # Test DNSMOS functionality
    print("Testing DNSMOS...")
    
    # Create a test signal
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note
    
    try:
        score = dnsmos(test_audio, sample_rate)
        print(f"DNSMOS test successful! Score: {score:.3f}")
    except Exception as e:
        print(f"DNSMOS test failed: {e}")
'''
    
    wrapper_path = Path("python_modules") / "dnsmos_wrapper.py"
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    print_status("DNSMOS wrapper created", "SUCCESS")

def main():
    """Main installation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Install Python dependencies for HADatasets")
    parser.add_argument("--conda", action="store_true", help="Use conda instead of pip")
    parser.add_argument("--force", action="store_true", help="Force reinstall packages")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print_status("HADatasets Python Dependencies Installation", "INFO")
    print_status("=" * 50, "INFO")
    
    # Check Python version
    check_python_version()
    
    # Check package manager
    if args.conda:
        if not check_conda():
            print_status("Conda not found. Please install conda or use pip.", "ERROR")
            sys.exit(1)
        print_status("Using conda for package installation", "INFO")
    else:
        if not check_pip():
            print_status("Pip not found. Please install pip or use conda.", "ERROR")
            sys.exit(1)
        print_status("Using pip for package installation", "INFO")
    
    # Core dependencies
    core_packages = [
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "pandas",
        "tqdm"
    ]
    
    # Metrics-specific packages
    metrics_packages = [
        "pypesq",
        "onnxruntime"
    ]
    
    print_status("Installing core dependencies...", "INFO")
    for package in core_packages:
        install_package(package, args.conda, args.force, args.verbose)
    
    print_status("Installing metrics dependencies...", "INFO")
    for package in metrics_packages:
        install_package(package, args.conda, args.force, args.verbose)
    
    # Check DNSMOS submodule
    if check_dnsmos_submodule():
        create_dnsmos_wrapper()
    
    print_status("Installation completed!", "SUCCESS")
    print_status("You can now use the HADatasets metrics module.", "INFO")
    
    # Test installation
    print_status("Testing installation...", "INFO")
    try:
        import numpy as np
        import librosa
        import soundfile as sf
        import pypesq
        import onnxruntime as ort
        print_status("All core packages imported successfully", "SUCCESS")
    except ImportError as e:
        print_status(f"Import test failed: {e}", "WARNING")
        print_status("Some packages may not be properly installed", "WARNING")

if __name__ == "__main__":
    main()

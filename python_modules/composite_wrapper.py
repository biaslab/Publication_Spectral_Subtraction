"""
Composite Speech Quality Metrics Wrapper

Computes the CSIG, CBAK, COVL composite measures from Hu & Loizou (2008),
"Evaluation of Objective Quality Measures for Speech Enhancement,"
IEEE Trans. Audio Speech Lang. Process., 16(1), pp. 229-238.

These are linear regressions of PESQ, LLR, WSS, and segSNR that predict
subjective MOS ratings from the ITU-T P.835 methodology.

Requires: pysepm (pip install https://github.com/schmiph2/pysepm/archive/master.zip)
"""

import numpy as np

# NumPy 2.0 removed np.NaN; restore it so pysepm works unpatched.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

_pysepm = None


def _ensure_pysepm():
    global _pysepm
    if _pysepm is None:
        try:
            import pysepm
            _pysepm = pysepm
        except ImportError:
            raise ImportError(
                "pysepm not installed. Install with:\n"
                "  pip install https://github.com/schmiph2/pysepm/archive/master.zip"
            )


def composite(clean, enhanced, fs):
    """
    Compute CSIG, CBAK, COVL composite metrics.

    Parameters
    ----------
    clean : np.ndarray
        Clean reference signal (1-D float).
    enhanced : np.ndarray
        Enhanced/processed signal (1-D float, same length as clean).
    fs : int
        Sampling rate in Hz (must be 16000).

    Returns
    -------
    dict with keys 'CSIG', 'CBAK', 'COVL' (float, range [1, 5]).
    """
    _ensure_pysepm()

    clean = np.asarray(clean, dtype=np.float64)
    enhanced = np.asarray(enhanced, dtype=np.float64)

    csig, cbak, covl = _pysepm.composite(clean, enhanced, fs)

    return {"CSIG": float(csig), "CBAK": float(cbak), "COVL": float(covl)}


if __name__ == "__main__":
    # Quick smoke test
    fs = 16000
    t = np.linspace(0, 1.0, fs)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    noisy = clean + 0.1 * np.random.randn(len(t))
    scores = composite(clean, noisy, fs)
    print(f"CSIG={scores['CSIG']:.2f}  CBAK={scores['CBAK']:.2f}  COVL={scores['COVL']:.2f}")

from setuptools import setup, find_packages

setup(
    name="dnsmos_wrapper",
    version="0.1.0",
    description="DNSMOS wrapper for Julia integration",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy", 
        "librosa",
        "soundfile",
        "onnxruntime"
    ],
    python_requires=">=3.7",
)

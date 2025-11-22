import subprocess

def detect_gpu():
    #detect if a GPU is aviable on this system
    try:
        subprocess.run(["nvidia-smi"],stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("GPU detected — using GPU mode for XGBoost")
        return True
    except Exception:
        print("No GPU detected — falling back to CPU mode")
        return False

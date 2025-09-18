import torch

def set_device(): 

    if torch.backends.mps.is_available(): 
        print("Device: Using MPS on Apple Silicon")
        return "mps"

    elif torch.cuda.is_available():
        print("CUDA Device available")
        return "cud"
    
    else: 
        raise("DeviceError: No GPU found. Training on CPU is not feasible")

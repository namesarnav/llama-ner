import torch
from warnings import showwarning

def set_device(): 

    if torch.backends.mps.is_available(): 
        print("Device: Using MPS on Apple Silicon")
        torch.device("mps")

    elif torch.cuda.is_available():
        print("CUDA Device available, using cuda")  
        torch.device("cuda")
    
    else: 
        showwarning("No GPU found. Training on CPU is not recommended")
        torch.device("cpu")

import torch
import wfdb
import numpy as np
import sklearn

print("Torch:", torch.__version__)
print("WFDB:", wfdb.__version__)
print("NumPy:", np.__version__)
print("CUDA:", torch.cuda.is_available())

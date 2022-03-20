import torch.cuda
print("cuda" if torch.cuda.is_available() else "cpu")


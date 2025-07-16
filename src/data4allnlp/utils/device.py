import torch

def select_device(device: str = "auto") -> torch.device:
        """
        Selects the appropriate device based on user input and system availability.
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
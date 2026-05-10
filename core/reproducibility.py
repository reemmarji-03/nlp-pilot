import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Best-effort deterministic seeding for local numerical components."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

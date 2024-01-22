from .deps import MASE_HW_DEPS
import os


def get_modules():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        d
        for d in os.listdir(current_dir)
        if os.path.isdir(os.path.join(current_dir, d))
    ]

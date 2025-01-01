import os
import sys
import dill
import numpy as np
import pandas as pd
from src.exception import CustomException


def save_object(obj, file_path):
    """
    Save an object to a file using dill
    
    Args:
        obj: Object to be saved
        file_path: Path to save the object
    """
    try:
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(f"Save object failed: {str(e)}", sys)
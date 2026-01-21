import numpy as np
import os
from glob import glob

def load_raw_data(key=None):
    if key:
        return glob(os.path.join(os.getcwd(), 
                      "dataset", f"*{key}*.raw"))
    else:
        return glob(os.path.join(os.getcwd(), 
                      "dataset", "*.raw"))
    
def load_in_numpy(file_path, 
                  shape:tuple[int, ...] = (1000,1000,1000), 
                  dtype=np.uint8):
    """Load raw binary data into a numpy array with given shape and dtype."""
    return np.fromfile(file_path, dtype=dtype).reshape(shape)

def load_in_numpy_list(file_paths:list[str], 
                       shape:tuple[int, ...] = (1000,1000,1000), 
                       dtype=np.uint8):
    """Load multiple raw binary data files into a list of numpy arrays."""
    file_list = [load_in_numpy(fp, shape, dtype) for fp in file_paths]
    if len(file_list) == 1:
        return file_list[0]
    else:
        return file_list
    
def load_with_keyword_among_list(file_paths:list[str], 
                                    keyword:str, 
                                    shape:tuple[int, ...] = (1000,1000,1000), 
                                    dtype=np.uint8):
    """Load raw binary data files that contain a specific keyword in their filename."""
    filtered_files = [fp for fp in file_paths if keyword in os.path.basename(fp)]
    return load_in_numpy_list(filtered_files, shape, dtype)
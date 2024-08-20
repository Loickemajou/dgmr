import datetime as dt
from pathlib import Path
from typing import List
import tensorflow as tf

import h5py
import numpy as np
from scipy.ndimage import zoom

from dgmr.settings import DATA_PATH, INPUT_STEPS, TIMESTEP


def get_list_files(date: dt.datetime) -> List[Path]:
    delta = dt.timedelta(minutes=TIMESTEP)
    dates = [date + i * delta for i in range(-INPUT_STEPS + 1, 1)]
    # filenames = [d.strftime("%Y_%m_%d_%H_%M.rad.becomp00.image.rate.becomp00_comp_sri.hdf") for d in dates]
    filenames = [f"{d.year}{d:%m}{d:%d}{d:%H}{d:%M}{0}{0}.rad.becomp00.image.rate.becomp00_comp_sri.hdf" for d in dates]
    return [DATA_PATH / f for f in filenames]
   


def open_radar_file(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as ds:
        array = np.array(ds["dataset1"]["data1"]["data"])
        array = np.expand_dims(array, -1)  # Add channel dims
        print(array.shape)
        tensor = tf.convert_to_tensor(array)
        print(tensor.shape)
        x = pad_along_axis(tensor, axis=0, pad_size=3)
        x = pad_along_axis(x, axis=1, pad_size=68)
        x=tf.image.resize(x,(1536,1280))
        array = x.numpy()
    return array

def pad_along_axis(x, pad_size = 3, axis = 2):
    '''
        Pad input to be divisible by 2. 
        height of 765 to 768
    '''
    if pad_size <= 0:
        return x
    npad = [(0, 0)] * x.ndim
    npad[axis] = (0, pad_size)

    return tf.pad(x, paddings=npad, constant_values=0)
      
def get_input_array(paths: List[Path]) -> np.ndarray:
    arrays = [open_radar_file(path) for path in paths]
 
    # Put values outside radar field to 0
    mask = np.where(arrays[0] == 65535, 1, 0)
    arrays = [np.where(array == 65535, 0, array) for array in arrays]
    # Rescale to 1km resolution
    arrays = [zoom(array, (0.5, 0.5)) for array in arrays]
    mask = zoom(mask, (0.5, 0.5))

    array = np.stack(arrays)
   
    print(array)
    array = array / 100 * 12  # Conversion from mm cumulated in 5min to mm/h
   
    return array, mask

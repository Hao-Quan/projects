from os import listdir
from os.path import isfile, join

import pandas as pd
import os

landmarks_frame = pd.read_hdf("data.h5", key="X")
labels = pd.read_hdf("data.h5", key="Y")

print("")

import json
import os

# Open and read the JSON file
with open('pyna/tables/source/tables_jet_source.json', 'r') as file:
    data = json.load(file)


import numpy as np
import pandas as pd
import sys

a = np.load('/Users/laurensvoet/Documents/Code/pyNA_old/pyNA/data/sources/jet/spectral_function_extended_T.npy')
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

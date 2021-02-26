# Libraries import

import numpy as np
import pandas as pd

# Input data files are available in the "../input/" directory.

import os
for dirname, _, filenames in os.walk('./input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


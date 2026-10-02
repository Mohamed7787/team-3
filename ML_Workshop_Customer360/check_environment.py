import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib

print("Python:", sys.version.split()[0])
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)

if sys.version_info[:2] != (3, 7):
    print("\nWARNING: This workshop was prepared for Python 3.7.16.")
else:
    print("\nPython major/minor version is compatible.")

print("Environment check complete.")

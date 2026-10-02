import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
df=pd.read_csv("C:\\Users\\PC\\Downloads\\ML_Workshop_Customer360\\data\\customer_360_ml_workshop.csv")

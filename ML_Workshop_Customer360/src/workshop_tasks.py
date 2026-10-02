"""
Customer 360 ML Workshop — Student Starter File
Python 3.7.16 compatible.

Complete every TODO. Refer to README.md for requirements.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_STATE = 42
DATA_PATH = "data/customer_360_ml_workshop.csv"


def part_a_eda(df):
    """TODO: Inspect data, missing values, target distribution, and business patterns."""
    # TODO
    pass


def part_b_preprocessing(df):
    """TODO: Build numeric/categorical preprocessing pipelines and return required objects."""
    # TODO
    pass


def part_c_classification(df):
    """TODO: Train, compare, evaluate, and tune churn classifiers."""
    # TODO
    pass


def part_d_regression(df):
    """TODO: Train and compare revenue regression models."""
    # TODO
    pass


def part_e_clustering(df):
    """TODO: Apply K-Means, Agglomerative Clustering, and DBSCAN."""
    # TODO
    pass


def part_f_pca_anomalies(df):
    """TODO: Apply PCA and Isolation Forest."""
    # TODO
    pass


def part_g_customer360(df):
    """TODO: Integrate model outputs into the final Customer 360 table."""
    # TODO
    pass


def main():
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded:", df.shape)

    # Uncomment sections as you complete them.
    # part_a_eda(df)
    # part_b_preprocessing(df)
    # part_c_classification(df)
    # part_d_regression(df)
    # part_e_clustering(df)
    # part_f_pca_anomalies(df)
    # part_g_customer360(df)


if __name__ == "__main__":
    main()

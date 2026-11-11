import pandas as pd
import sys
import numpy as np
from pandas.compat import pickle_compat

sys.modules["pandas.indexes"] = pd.core.indexes

with open("dataset/LSWMD.pkl", "rb") as file:
    df = pickle_compat.Unpickler(
        file, encoding="latin1"
    ).load()

# Select an Edge-Ring wafer
sample = df[
    df["failureType"].apply(
        lambda x: str(x[0][0]) == "Edge-Ring"
        if len(x) and len(x[0]) else False
    )
].iloc[0]

np.save("results/sample_wafer.npy", sample["waferMap"])

print("Sample wafer saved successfully!")
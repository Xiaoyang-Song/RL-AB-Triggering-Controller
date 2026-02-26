import pandas as pd
import numpy as np


results_df = pd.read_csv("evaluation_results.csv")
print(results_df.head())

trigger_collision_cases = (
    results_df.loc[
        (results_df["triggered"]) & (results_df["collision"]) & (results_df["trigger_frame"] < results_df["collision_frame"])
    ]
    .reset_index(drop=True)
)

print(trigger_collision_cases)
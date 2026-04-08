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

# trigger_collision_cases = (
#     results_df.loc[
#         (results_df["triggered"]) & (results_df["collision"])
#     ]
#     .reset_index(drop=True)
# )

# count_trigger: 4098
# count_case_1 (no trigger but collision): 354
# count_case_2 (trigger and collision): 1823
# count_case_3 (trigger but no collision): 300

print(trigger_collision_cases)
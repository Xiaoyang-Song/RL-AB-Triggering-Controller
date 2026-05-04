import pandas as pd
import numpy as np


results_df = pd.read_csv("evaluation_results.csv")
print(results_df[['trajectory_id', 'trigger_frame', 'collision', 'triggered',
                               'collision_frame', 'ttc_at_trigger', 
                               'pjoint']].head(5))

# trigger_collision_cases = (
#     results_df.loc[s
#         (results_df["triggered"]) & (results_df["collision"]) & (results_df["trigger_frame"] < results_df["collision_frame"])
#     ]
#     .reset_index(drop=True)
# )

trigger_collision_cases = (
    results_df.loc[
        (results_df["triggered"]==False) & (results_df["collision"]==False)
    ]
    .reset_index(drop=True)
)

# count_trigger: 4098
# count_case_1 (no trigger but collision): 354
# count_case_2 (trigger and collision): 1823
# count_case_3 (trigger but no collision): 300

print(trigger_collision_cases[['trajectory_id', 'collision', 'triggered', 
                            #    'trigger_frame','collision_frame', 
                               'ttc_at_trigger', 
                               'pjoint']].head(12))

# =========================
# Success Trigger Rate
# =========================
# Filter for high-risk cases: collision=True AND pjoint > 0.2
high_risk_cases = results_df.loc[(results_df['collision'] == True) & (results_df['pjoint'] > 0.4)]

# Count how many of these high-risk cases were triggered
high_risk_triggered = high_risk_cases.loc[high_risk_cases['triggered'] == True]

total_high_risk = len(high_risk_cases)
triggered_count = len(high_risk_triggered)

success_trigger_rate = (triggered_count / total_high_risk * 100) if total_high_risk > 0 else 0

print("\n" + "="*60)
print("SUCCESS TRIGGER RATE")
print("="*60)
print(f"High-Risk Cases (collision=True AND pjoint > 0.2): {total_high_risk}")
print(f"Successfully Triggered: {triggered_count}")
print(f"Success Trigger Rate: {success_trigger_rate:.2f}%")
print("="*60)

# =========================
# Low Injury Risk with Collision
# =========================
# Cases with collision but low injury risk: collision=True AND pjoint <= 0.2
low_risk_collision_cases = results_df.loc[(results_df['collision'] == True) & (results_df['pjoint'] <= 0.4)]

low_risk_triggered = len(low_risk_collision_cases.loc[low_risk_collision_cases['triggered'] == True])
low_risk_not_triggered = len(low_risk_collision_cases.loc[low_risk_collision_cases['triggered'] == False])
total_low_risk_collision = len(low_risk_collision_cases)

low_risk_trigger_rate = (low_risk_triggered / total_low_risk_collision * 100) if total_low_risk_collision > 0 else 0
low_risk_no_trigger_rate = (low_risk_not_triggered / total_low_risk_collision * 100) if total_low_risk_collision > 0 else 0

print("\n" + "="*60)
print("LOW INJURY RISK WITH COLLISION")
print("="*60)
print(f"Cases (collision=True AND pjoint <= 0.2): {total_low_risk_collision}")
print(f"Triggered: {low_risk_triggered} ({low_risk_trigger_rate:.2f}%)")
print(f"Not Triggered: {low_risk_not_triggered} ({low_risk_no_trigger_rate:.2f}%)")
print("="*60)

# =========================
# No Collision Cases
# =========================
# Cases with no collision: collision=False
no_collision_cases = results_df.loc[results_df['collision'] == False]

no_collision_triggered = len(no_collision_cases.loc[no_collision_cases['triggered'] == True])
no_collision_not_triggered = len(no_collision_cases.loc[no_collision_cases['triggered'] == False])
total_no_collision = len(no_collision_cases)

no_collision_trigger_rate = (no_collision_triggered / total_no_collision * 100) if total_no_collision > 0 else 0
no_collision_no_trigger_rate = (no_collision_not_triggered / total_no_collision * 100) if total_no_collision > 0 else 0

print("\n" + "="*60)
print("NO COLLISION CASES")
print("="*60)
print(f"Cases (collision=False): {total_no_collision}")
print(f"Triggered: {no_collision_triggered} ({no_collision_trigger_rate:.2f}%)")
print(f"Not Triggered: {no_collision_not_triggered} ({no_collision_no_trigger_rate:.2f}%)")
print("="*60)

# =========================
# Failure Cases
# =========================
# Failure cases: high-risk but did NOT trigger
# high_risk_failed = high_risk_cases.loc[high_risk_cases['triggered'] == False]

# print("\n" + "="*60)
# print("FAILURE CASES (High-Risk but Did NOT Trigger)")
# print("="*60)
# print(f"Total Failure Cases: {len(high_risk_failed)}\n")

# if len(high_risk_failed) > 0:
#     print(high_risk_failed[['trajectory_id', 'collision', 'triggered', 'pjoint', 
#                             'ttc_at_trigger', 'collision_frame']].to_string())
# else:
#     print("No failure cases found!")
# print("="*60)
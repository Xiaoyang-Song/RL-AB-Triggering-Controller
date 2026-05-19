import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse evaluation results for airbag triggering.")

    # Paths
    parser.add_argument("--results_dir", type=str, default="results")

    # Reward / eta parameters (used only for resolving filename)
    parser.add_argument("--b1", type=float, default=2.0)
    parser.add_argument("--c1", type=float, default=1.5)
    parser.add_argument("--b2", type=float, default=1.0)
    parser.add_argument("--c2", type=float, default=3.0)
    parser.add_argument("--c3", type=float, default=2.0)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--train_noise_std", type=float, default=0.0,
                        help="Noise std used during training (0 = no noise). Used to locate the correct results file.")
    parser.add_argument("--label", type=str, default="", help="Optional label tag appended to CSV filename (e.g., 'rep1')")

    return parser.parse_args()


def build_param_suffix(args):
    suffix = f"b1{args.b1}_c1{args.c1}_b2{args.b2}_c2{args.c2}_c3{args.c3}_eta{args.eta}"
    if args.train_noise_std > 0.0:
        suffix += f"_noise{args.train_noise_std}"
    return suffix

def main():
    args = parse_args()
    suffix = build_param_suffix(args)
    label_part = f"_{args.label}" if args.label else ""
    results_path = f"{args.results_dir}/evaluation_results_{suffix}{label_part}.csv"

    print(f"Loading results from: {results_path}")
    results_df = pd.read_csv(results_path)

    print(results_df[[
        "trajectory_id", "trigger_frame", "collision", "triggered",
        "collision_frame", "ttc_at_trigger", "pjoint"
    ]].head(5))

    # =========================
    # Success Trigger Rate
    # High-risk: collision=True AND pjoint > eta
    # =========================
    high_risk_cases = results_df.loc[
        (results_df["collision"] == True) & (results_df["pjoint"] > args.eta)
    ]
    high_risk_triggered = high_risk_cases.loc[high_risk_cases["triggered"] == True]

    total_high_risk = len(high_risk_cases)
    triggered_count = len(high_risk_triggered)
    success_trigger_rate = (triggered_count / total_high_risk * 100) if total_high_risk > 0 else 0

    print("\n" + "=" * 60)
    print("SUCCESS TRIGGER RATE  (Type-II complement)")
    print("=" * 60)
    print(f"High-risk cases (collision=True AND pjoint > {args.eta}): {total_high_risk}")
    print(f"Successfully triggered:                                    {triggered_count}")
    print(f"Success trigger rate:                                      {success_trigger_rate:.2f}%")
    print(f"Type-II error rate (missed high-risk):                     {100 - success_trigger_rate:.2f}%")
    print("=" * 60)

    # =========================
    # Low Injury Risk with Collision  (contributes to Type-I)
    # collision=True AND pjoint <= eta
    # =========================
    low_risk_collision_cases = results_df.loc[
        (results_df["collision"] == True) & (results_df["pjoint"] <= args.eta)
    ]
    total_low_risk_collision = len(low_risk_collision_cases)
    low_risk_triggered = len(low_risk_collision_cases.loc[low_risk_collision_cases["triggered"] == True])
    low_risk_not_triggered = len(low_risk_collision_cases.loc[low_risk_collision_cases["triggered"] == False])
    low_risk_trigger_rate = (low_risk_triggered / total_low_risk_collision * 100) if total_low_risk_collision > 0 else 0
    low_risk_no_trigger_rate = (low_risk_not_triggered / total_low_risk_collision * 100) if total_low_risk_collision > 0 else 0

    print("\n" + "=" * 60)
    print("LOW INJURY RISK WITH COLLISION  (Type-I source)")
    print("=" * 60)
    print(f"Cases (collision=True AND pjoint <= {args.eta}):  {total_low_risk_collision}")
    print(f"Triggered (unnecessary):   {low_risk_triggered} ({low_risk_trigger_rate:.2f}%)")
    print(f"Not triggered (correct):   {low_risk_not_triggered} ({low_risk_no_trigger_rate:.2f}%)")
    print("=" * 60)

    # =========================
    # No Collision Cases  (contributes to Type-I)
    # =========================
    no_collision_cases = results_df.loc[results_df["collision"] == False]
    total_no_collision = len(no_collision_cases)
    no_collision_triggered = len(no_collision_cases.loc[no_collision_cases["triggered"] == True])
    no_collision_not_triggered = len(no_collision_cases.loc[no_collision_cases["triggered"] == False])
    no_collision_trigger_rate = (no_collision_triggered / total_no_collision * 100) if total_no_collision > 0 else 0
    no_collision_no_trigger_rate = (no_collision_not_triggered / total_no_collision * 100) if total_no_collision > 0 else 0

    print("\n" + "=" * 60)
    print("NO COLLISION CASES  (Type-I source)")
    print("=" * 60)
    print(f"Cases (collision=False):           {total_no_collision}")
    print(f"Triggered (unnecessary):   {no_collision_triggered} ({no_collision_trigger_rate:.2f}%)")
    print(f"Not triggered (correct):   {no_collision_not_triggered} ({no_collision_no_trigger_rate:.2f}%)")
    print("=" * 60)

    # =========================
    # Overall Type-I / Type-II summary
    # Mirrors the logic in evaluate.py and train.py
    # =========================
    type1_denom = total_low_risk_collision + total_no_collision
    type1_count = low_risk_triggered + no_collision_triggered
    type1_rate = (type1_count / type1_denom * 100) if type1_denom > 0 else 0

    type2_denom = total_high_risk
    type2_count = total_high_risk - triggered_count
    type2_rate = (type2_count / type2_denom * 100) if type2_denom > 0 else 0

    print("\n" + "=" * 60)
    print("OVERALL ERROR SUMMARY")
    print("=" * 60)
    print(f"Type-I  error (false trigger):   {type1_count}/{type1_denom} = {type1_rate:.2f}%")
    print(f"Type-II error (missed trigger):  {type2_count}/{type2_denom} = {type2_rate:.2f}%")
    print("=" * 60)

    # =========================
    # Triggered + collision detail
    # =========================
    trigger_collision_cases = results_df.loc[
        results_df["triggered"] & results_df["collision"]
    ].reset_index(drop=True)

    # print("\nTriggered + collision cases:")
    # print(trigger_collision_cases[[
    #     "trajectory_id", "collision", "triggered",
    #     "ttc_at_trigger", "pjoint"
    # ]].head(12))


if __name__ == "__main__":
    main()
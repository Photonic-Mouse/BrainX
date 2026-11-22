import json
import pandas as pd
import numpy as np

# Load summaries and predictions
agg_df = pd.read_csv('results/evaluation/final_test_aggregates.csv')
sum_df = pd.read_csv('results/evaluation/final_test_metrics_summary.csv')
ci_data = json.load(open('results/evaluation/confidence_intervals.json'))
mcnemar = json.load(open('results/evaluation/mcnemar_paired_comparisons.json'))
preds = pd.read_csv('results/evaluation/test_predictions_frozen_winner.csv')

print("=" * 82)
print("          BRAINSCOPE-XAI: FINAL FROZEN MODEL TEST EVALUATION RESULTS          ")
print("=" * 82)
print("  Model Architecture : ResNet-18 (ImageNet Pretrained)")
print("  Selected Pipeline  : raw_rot5 (Raw Grayscale + Training-Only Random Rotation +/-5 deg)")
print("  Imbalance Strategy : Inverse-Frequency Class Weights [1.786, 24.889, 0.500, 0.714]")
print("  Inference Pipeline : Deterministic Resize(224, 224) + ImageNet Normalize")
print("  Test Cohort Size   : 960 Images (480 Non, 336 Very Mild, 135 Mild, 9 Moderate)")
print("=" * 82)

print("\n[1] OVERALL ACCURACY & CONFIDENCE INTERVALS (FROZEN WINNER: raw_rot5)")
print("-" * 82)
print(f"{'Seed':<8} | {'Correct/Total':<14} | {'Test Accuracy':<14} | {'95% Wilson CI':<20} | {'Macro-F1':<10}")
print("-" * 82)
for s in [17, 42, 2026]:
    sub = sum_df[(sum_df['model_code'] == 'raw_rot5') & (sum_df['seed'] == s)].iloc[0]
    ci = ci_data[f'seed_{s}']['accuracy_wilson_95_ci']
    ci_str = f"[{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]"
    corr = int(sub['correct_predictions'])
    tot = int(sub['total_test_images'])
    pct = (corr / tot) * 100
    acc = sub['test_accuracy'] * 100
    f1 = sub['test_macro_f1'] * 100
    print(f"Seed {s:<3} | {corr:>3}/{tot} ({pct:.2f}%) | {acc:>6.2f}%       | {ci_str:<20} | {f1:>6.2f}%")

print("-" * 82)
w_agg = agg_df[agg_df['model_code'] == 'raw_rot5'].iloc[0]
acc_m = w_agg['test_accuracy_mean'] * 100
acc_s = w_agg['test_accuracy_std'] * 100
f1_m = w_agg['test_macro_f1_mean'] * 100
f1_s = w_agg['test_macro_f1_std'] * 100
print(f"MEAN +/- STD : {acc_m:.2f}% +/- {acc_s:.2f}% Accuracy  |  {f1_m:.2f}% +/- {f1_s:.2f}% Macro-F1")
print("=" * 82)

print("\n[2] BENCHMARK COMPARISON ACROSS ALL TESTED MODELS (N=960)")
print("-" * 82)
print(f"{'Model Code':<15} | {'Description':<32} | {'Mean Accuracy':<16} | {'Mean Macro-F1':<14}")
print("-" * 82)
for _, r in agg_df.iterrows():
    m_name = r['model_code']
    desc = r['model_description'][:32]
    a_str = f"{r['test_accuracy_mean']*100:.2f}% +/- {r['test_accuracy_std']*100:.2f}%"
    f_str = f"{r['test_macro_f1_mean']*100:.2f}% +/- {r['test_macro_f1_std']*100:.2f}%"
    star = " *WINNER*" if m_name == "raw_rot5" else ""
    print(f"{m_name:<15} | {desc:<32} | {a_str:<16} | {f_str:<14}{star}")
print("=" * 82)

print("\n[3] CLASS-WISE METRICS BREAKDOWN (FROZEN WINNER: raw_rot5)")
print("-" * 82)
print(f"{'Class Name':<20} | {'Test Scans':<10} | {'Avg Recall':<12} | {'Avg Precision':<14} | {'Avg F1-Score':<12}")
print("-" * 82)
winner_runs = sum_df[sum_df['model_code'] == 'raw_rot5']
classes = [
    ('Moderate_Demented', 9, 'moderate_recall', 'moderate_precision', 'moderate_f1'),
    ('Mild_Demented', 135, 'mild_recall', 'mild_precision', 'mild_f1'),
    ('Very_Mild_Demented', 336, 'verymild_recall', 'verymild_precision', 'verymild_f1'),
    ('Non_Demented', 480, 'non_recall', 'non_precision', 'non_f1')
]
for c_name, count, rec_col, prec_col, f1_col in classes:
    rec = winner_runs[rec_col].mean() * 100
    prec = winner_runs[prec_col].mean() * 100
    f1 = winner_runs[f1_col].mean() * 100
    print(f"{c_name:<20} | {count:<10} | {rec:>6.2f}%      | {prec:>6.2f}%        | {f1:>6.2f}%")
print("=" * 82)

print("\n[4] MODEL PREDICTION CONFIDENCE ANALYSIS")
print("-" * 82)
sub_p = preds[preds['seed'] == 17]
corr = sub_p[sub_p['is_correct'] == 1]
inc = sub_p[sub_p['is_correct'] == 0]
probs_corr = corr[['prob_mild', 'prob_moderate', 'prob_non', 'prob_verymild']].max(axis=1)
probs_inc = inc[['prob_mild', 'prob_moderate', 'prob_non', 'prob_verymild']].max(axis=1)
print(f"When prediction is CORRECT (936 scans) : Mean Conf = {probs_corr.mean()*100:.2f}%  | Median Conf = {probs_corr.median()*100:.2f}%")
print(f"When prediction is WRONG   (24 scans)  : Mean Conf = {probs_inc.mean()*100:.2f}%  | Median Conf = {probs_inc.median()*100:.2f}%")
print("=" * 82)

print("\n[5] CONFUSION MATRIX (PEAK RUN: SEED 17, 97.50% ACCURACY)")
print("-" * 82)
cm = np.array(ci_data['seed_17']['confusion_matrix'])
print("                 Predicted: Mild    Mod    Non   VeryMild")
row_labels = ['Actual Mild (135)', 'Actual Mod (9)  ', 'Actual Non (480) ', 'Actual VeryMild(336)']
for idx, r_lbl in enumerate(row_labels):
    print(f"{r_lbl:20s}  :  {cm[idx, 0]:>4d}   {cm[idx, 1]:>4d}   {cm[idx, 2]:>4d}   {cm[idx, 3]:>4d}")
print("=" * 82)

print("\n[6] PAIRED McNEMAR TEST (STATISTICAL SIGNIFICANCE ON 960 SCANS)")
print("-" * 82)
mc17 = mcnemar['seed_17']['winner_vs_raw_no_aug']
p_val = mc17['p_value']
odds_r = mc17['odds_ratio']
gain = mc17['risk_difference_accuracy_gain'] * 100
chi2 = mc17['mcnemar_chi2_statistic']
or_ci = mc17['odds_ratio_95_ci']
rd_ci = mc17['risk_difference_95_ci']
print("raw_rot5 vs Baseline Control (raw_no_aug) on Seed 17:")
print(f"  Chi-Square Statistic = {chi2:.3f}")
print(f"  p-value              = {p_val:.6f}  (Statistically Significant, p < 0.01)")
print(f"  Odds Ratio           = {odds_r:.2f} (95% CI: [{or_ci[0]:.2f}, {or_ci[1]:.2f}])")
print(f"  Net Accuracy Gain    = +{gain:.2f}% (95% CI: [{rd_ci[0]*100:.2f}%, {rd_ci[1]*100:.2f}%])")
print("=" * 82)

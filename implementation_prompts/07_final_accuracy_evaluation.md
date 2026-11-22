# Stage 7 prompt — evaluate and report the frozen accuracy result

Copy this prompt only after reviewing and freezing the Stage 6 pipeline:

---

Do only Stage 7. Do not change labels, the split, model weights, preprocessing, augmentation, threshold, or class order after opening test results. Do not automatically begin Grad-CAM or website work.

## Test-set status

The current test set has 960 images and its results have already been examined; CLAHE was previously chosen after comparing those test metrics. Preserve the exact test membership, but be honest that a new report on the same images is a previously consulted internal holdout, not a fresh untouched confirmation. If a separately sourced external dataset is already available, report it separately as external testing. Do not replace the current split or call validation an external test.

## Final evaluation loop

1. Freeze and record code version, split-manifest checksum, selected checkpoint per seed, preprocessing, augmentation, class order, and inference transform.
2. Verify the evaluation loader is deterministic and applies no random augmentation.
3. Evaluate the frozen selected pipeline and predeclared baselines on the same fixed test rows. Do not repeat evaluation to select a new candidate.
4. Save one row per test image with an opaque image ID, true label, predicted label, per-class probabilities, model ID, and seed. Do not expose personal identifiers.
5. Calculate correct/total and accuracy; macro-F1, weighted-F1, balanced accuracy; per-class precision, recall, and F1; confusion matrix; and 95% confidence intervals. Explain that image-level intervals do not establish person-level generalization when subject IDs are unknown.
6. For paired models, compare correctness on the same images with a paired method such as McNemar’s test. Report the effect size and uncertainty, not only a p-value.
7. Recompute all numbers from saved predictions and confirm reports, tables, and plots match.
8. Compare results across the predeclared seeds. Report mean and standard deviation. Do not report only the best seed.

## Accuracy interpretation

With 960 test images, one changed prediction changes accuracy by about 0.104 percentage points. A claimed 99.99% cannot be meaningfully resolved by this test size. Do not promise 99% or 99.99%; report what the frozen run actually achieved. If it is below the aspiration, identify the likely bottleneck and recommend a new experiment only as a future manually invoked stage.

If all 960 predictions were correct, report 960/960 on this image set and its uncertainty; do not equate that with proven 99.99% accuracy on future patients. Augmentation does not increase the independent test sample size.

## Deliverables

Save a final metrics CSV/JSON, per-image prediction CSV, confusion matrix, confidence intervals, paired comparison, seed summary, and a plain-language report clearly labeled “previously consulted internal holdout” unless truly separate external data were used. Do not calibrate or present softmax values as clinical certainty in this accuracy stage. End with the achieved result and remaining limits, then stop. Grad-CAM and website work are explicitly out of scope.

## References

- CLAIM 2024 reporting guidance: https://pubs.rsna.org/doi/10.1148/ryai.240300
- MRI split-leakage evidence: https://www.nature.com/articles/s41598-021-01681-w
- Scikit-learn classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
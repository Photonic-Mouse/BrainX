# Stage 6 prompt — optimize the combined pipeline using validation only

Copy this prompt after Stages 3–5 are reviewed:

---

Do only Stage 6. The goal is to select the strongest stable preprocessing/augmentation/training recipe without using test data. Keep the existing 70/15/15 membership unchanged. Do not proceed to final evaluation automatically.

## Freeze the candidate list first

Before running anything, write down the candidates from prior stages:
- raw/no augmentation;
- raw/best validation-selected augmentation;
- best validation-selected preprocessing/no augmentation;
- best preprocessing/best augmentation;
- one intentionally stronger but still plausible stress condition only if Stage 4 already defined it.

Use the same imbalance strategy and training recipe selected in Stage 5. Do not add new filters, model architectures, or augmentation families after seeing scores.

## Repeated optimization loop

For each candidate, train across the same three seeds, using the same initial pretrained weights, training split, optimizer schedule, and epoch limit. For every epoch, save validation accuracy, macro-F1, balanced accuracy, per-class recall, and loss. Use early stopping patience 5 on validation macro-F1.

After each full candidate:
1. Append results to an immutable experiment ledger; never overwrite an earlier run.
2. Compare the candidate’s mean and standard deviation across seeds with the best prior candidate.
3. Promote it only when the gain is repeatable across seeds and does not come from ignoring a minority class. Report accuracy even when macro-F1 selects a different recipe.
4. If two successive candidates fail to improve the best validation result, stop the search and keep the prior best stable candidate.
5. Stop when the predeclared list is complete, even if the aspiration of 99% was not met. Do not add experiments solely to force 99% or 99.99%.

If a candidate is highly variable, repeat that candidate with the same predeclared seeds; do not select its best single seed. The result should reflect typical performance across runs, not a lucky initialization.

## Model selection rule

Because the user cares about accuracy but the dataset is severely imbalanced, select by validation macro-F1 as the primary guardrail and accuracy as a co-primary outcome. If the accuracy leader has materially worse Moderate recall or macro-F1, show both candidates and explain the tradeoff instead of hiding it. Prefer the simplest recipe when differences are smaller than seed variation.

## Deliverables

Write a final validation-only leaderboard with mean ± standard deviation, class-wise metrics, candidate configuration, seed, best epoch, and chosen frozen pipeline. Include the complete experiment ledger. State clearly that test data were not used for this stage. Stop and wait for Stage 7.

## References

- Official metric documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
- MRI data-splitting risks: https://www.nature.com/articles/s41598-021-01681-w
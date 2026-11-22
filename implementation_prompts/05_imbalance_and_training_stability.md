# Stage 5 prompt — compare imbalance handling and training stability

Copy this prompt after Stage 4 is reviewed:

---

Do only Stage 5. Preserve the fixed manifest. Use the best validation-selected preprocessing/augmentation so far. Do not tune on the test set and do not create synthetic copies of validation/test images.

## Why this stage matters

The dataset has 64 Moderate images total and only 45 in train. Accuracy can look high while the model misses a minority class. Every method must report overall accuracy and class-aware results.

## Predeclared imbalance candidates

Compare with the same seeds, model, input pipeline, training budget, and checkpoint rule:

1. Ordinary cross-entropy.
2. Inverse-frequency class-weighted cross-entropy computed from training counts only.
3. A balanced training sampler without class weights.

Do not combine the weighted sampler and class weights in the main comparison. That can correct the same imbalance twice. Keep synthetic oversampling, SMOTE on pixels, and GAN-generated MRI out of the first pass. If focal loss is proposed, it must be one separately named candidate with a literature-based rationale and a user-approved follow-up prompt; do not quietly add it to the loop.

## Training-stability loop

For each candidate and each of the three seeds:

1. Train using the frozen common recipe.
2. Record train/validation loss, accuracy, macro-F1, balanced accuracy, per-class precision/recall/F1, and confusion matrix each epoch.
3. Use validation macro-F1 to select the checkpoint.
4. Record seed-to-seed variation and whether class gains trade off against majority-class accuracy.
5. If all candidates are unstable, run at most three predeclared learning-rate settings from Stage 2, identically for each imbalance candidate. Stop after two successive settings fail to improve mean validation macro-F1.

Do not change augmentation and imbalance method in the same comparison. Do not change the split. Do not present 9 Moderate test cases as a precise estimate of Moderate performance.

## Deliverables

Save one table for every strategy/seed and a concise recommendation naming the selected method and its accuracy/macro-F1/per-class tradeoffs. Preserve all runs and stop. Wait for the next prompt.

## References

- Review the dataset’s verified class counts and methods in the project’s literature PDFs.
- Official PyTorch losses and sampling: https://docs.pytorch.org/docs/stable/nn.html and https://docs.pytorch.org/docs/stable/data.html
- CLAIM 2024: https://pubs.rsna.org/doi/10.1148/ryai.240300
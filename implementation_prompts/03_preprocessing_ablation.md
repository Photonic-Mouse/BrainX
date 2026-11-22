# Stage 3 prompt — run the preprocessing experiment loop

Copy this prompt after Stage 2 is reviewed:

---

Do only Stage 3. Use the common trainer and fixed split from the prior stages. Do not start augmentation experiments. Do not view test predictions or compute new test scores.

## Research question

Does a particular preprocessing step improve validation performance when model, training settings, split, and augmentation are held constant?

## Predeclared candidates

Run these conditions with augmentation disabled in all arms:

1. Original grayscale with common resize and normalization.
2. Global histogram equalization as a comparison to the existing low-scoring HE result.
3. CLAHE clipLimit 1.5, tileGridSize 8×8.
4. CLAHE clipLimit 2.0, tileGridSize 8×8.
5. CLAHE clipLimit 2.5, tileGridSize 8×8.

Use the same per-image processing in train and validation. Keep image dimensions, interpolation, grayscale conversion, and normalization identical. Save comparison panels from fixed train/validation examples for quality review. Do not choose a method just because its images look more contrasted.

Do not add denoising, cropping, registration, skull stripping, bias correction, or other filters to this grid. The current data are 2D compressed images; 3D MRI operations are inappropriate without volumetric source data and a justified workflow. If Stage 1 identified a specific acquisition artifact, propose one additional preprocessing candidate in the report, but wait for a new user prompt before expanding this stage.

## Fair experiment loop

For every listed candidate:
1. Use the same three seeds, common trainer, model initialization policy, train-only imbalance strategy, epoch budget, and validation checkpoint rule.
2. Train only on train images. Select checkpoints using validation macro-F1.
3. Save validation probabilities, class-wise metrics, confusion matrix, and run configuration.
4. Repeat the exact candidate for all seeds.
5. Compare mean and spread. Report accuracy as a requested outcome, and also macro-F1, balanced accuracy, and every class’s recall. Do not call an accuracy gain an improvement if minority-class performance collapses.
6. Keep the best validation candidate and all lower-performing candidates. Do not tune on test results.

Stop after all five candidates are complete. If the latest candidates do not improve the best validation result, do not generate new CLAHE values. When scores are within seed variation, prefer the least aggressive and simplest preprocessing.

## Deliverables

Write a candidate/seed summary table, validation curves, confusion matrices, fixed-image preprocessing panels, and a brief conclusion identifying the best validation choice and tradeoffs. Stop before augmentation experiments.

## References

- OpenCV CLAHE API: https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html
- Consult the project’s original preprocessing papers, verifying their actual data, split unit, and test setup.
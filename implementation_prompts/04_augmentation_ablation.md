# Stage 4 prompt — run the training-only augmentation experiment loop

Copy this prompt after Stage 3 is reviewed:

---

Do only Stage 4. Keep the exact split and use the common trainer. Use the Stage 3 validation-selected preprocessing, plus the raw condition as a control where required to test interaction. Do not start class-imbalance or final-test work automatically.

## Objective

Measure whether plausible training-only transforms improve generalization on the fixed validation set. Increased augmentation is not automatically better; a transform can distort useful anatomy.

## Predeclared candidate recipes

For both raw input and the Stage 3 selected preprocessing, compare:

1. No augmentation.
2. Random rotation up to ±5 degrees.
3. Random rotation up to ±10 degrees, matching the legacy training behavior.
4. Mild affine augmentation: rotation up to ±5 degrees, translation no more than 3% of image width/height, scale 0.95–1.05.

If Stage 1 provides evidence of intensity variation that resembles acquisition variation, add at most one grayscale-only mild brightness/contrast or low-level noise condition, defined before training. Do not use color jitter on grayscale MRI.

Do not use horizontal flips unless dataset orientation is known and left/right reversal is scientifically acceptable. Do not begin with strong shear, large elastic warping, aggressive blur, or synthetic images. Do not make offline augmented copies that could be mixed across the fixed splits.

## Loop

1. Run a one-seed pilot for each candidate to catch coding and GPU-memory issues.
2. If the implementation passes checks, run the full predeclared candidates across the three fixed training seeds.
3. Apply transformations online to train images only. Leave validation images unchanged.
4. Record each seed’s accuracy, macro-F1, balanced accuracy, per-class recall, best epoch, and configuration.
5. Compare mean and variation across seeds. Select by validation evidence, with accuracy reported alongside class-aware metrics.
6. If two successive predeclared recipes fail to improve the best validation result, stop adding transformations. Keep the earlier best stable candidate.
7. Do not inspect test scores, select the best seed, or continue searching until the validation score reaches a desired number.

If GPU time is limited, finish the pilot for every candidate and then repeat the no-augmentation control plus the top two validation candidates across all three seeds. Label this reduced confirmation design clearly.

## Interpretation

Online augmentation creates correlated training variations, not new independent people. A model scoring above 99% on validation does not prove 99% performance on new people. Report whether the same transformation helped each class, especially Moderate, which has only 45 training images.

## Deliverables

Save before/after training examples, augmentation configuration, per-seed histories, a validation comparison table, and a recommendation of none/mild/no stable benefit. Stop after this stage and wait.

## References

- Official Torchvision transforms: https://docs.pytorch.org/vision/stable/transforms.html
- MRI split-leakage study: https://www.nature.com/articles/s41598-021-01681-w
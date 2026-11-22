# Stage 1 prompt — audit data and repair paths without changing the split

Copy this prompt to the implementation agent by itself:

---

You are working in the existing BrainScope-XAI repository. Do only Stage 1. Do not train a model, modify labels, change the split ratio, change any image’s split membership, delete images, or begin later stages automatically.

## Goal

Confirm exactly what data the saved metrics refer to, repair broken file paths while preserving all existing split assignments, and produce a trustworthy starting audit.

## Read before acting

Inspect repository instructions, README, all files under src/data, src/training, and src/evaluation, results summaries and prediction CSVs, checkpoints, the two user-provided notes, and the pinned chat “Review papers and plan slides.” Compare each claim with source code and saved artifacts. Mark claims as verified, contradicted, or not evidenced.

## Required data checks

1. Count the actual images by class and compare with the manifest. Expected totals: Mild 896, Moderate 64, Non Demented 3,200, Very Mild 2,240; total 6,400.
2. Count rows by fixed split. Expected totals: train 4,480, validation 960, test 960. Expected class counts:
   - train: Mild 627, Moderate 45, Non Demented 2,240, Very Mild 1,568;
   - validation: Mild 134, Moderate 10, Non Demented 480, Very Mild 336;
   - test: Mild 135, Moderate 9, Non Demented 480, Very Mild 336.
3. Verify each row’s folder, filename, class_name, and label agree. Confirm each manifest record resolves to exactly one current image file.
4. Find every stale absolute root in the split manifest, training, evaluation, inspection, preprocessing, and inference scripts. The current files are under the repository Dataset folder, not C:\HEHAW\BrainX.
5. Inspect dimensions, grayscale/RGB mode, intensity range, corrupted images, black borders, watermarks or other label-correlated artifacts.
6. Compute exact SHA-256 duplicates and a documented perceptual near-duplicate audit. Identify duplicate groups that cross train/validation/test. Do not move or delete anything.
7. Search for participant or scan identifiers in available source metadata. Do not infer an identifier from appearance. If no reliable IDs exist, state that subject-level independence cannot be verified from this dataset copy.
8. Compare the saved result files. Verify arithmetic from actual/predicted counts, but do not infer patient-level accuracy.

## Safe path repair

The CSV currently points at the old C:\HEHAW\BrainX\Dataset root. If every file maps unambiguously by its current class folder and filename, update only image_path to a configurable or repository-relative path. Preserve every row, label, class name, split value, and membership exactly. Record a before/after checksum over the stable identity fields and prove it is unchanged. Do not run a new splitter.

If any path is missing or ambiguous, stop before editing the manifest and report the exact records that need resolution.

## Stage-local loop

Repeat the file-resolution and integrity checks after the path-only repair until every row resolves and every expected count matches. Keep an audit log of each correction. If a mismatch cannot be resolved without changing membership, stop and ask for a decision; do not silently substitute another image.

## Deliverables

Write results/audit/data_audit_report.md, a machine-readable file inventory, duplicate groups, path-repair summary, and split-membership checksum. Include exact file counts and limitations. Do not train or inspect new model performance. End by listing the next-stage prerequisites, then stop and wait for the user’s next prompt.

## Primary references

- Yagis et al., MRI split leakage: https://www.nature.com/articles/s41598-021-01681-w
- CLAIM 2024 reporting guide: https://pubs.rsna.org/doi/10.1148/ryai.240300
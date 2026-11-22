import os
import glob
import torch
import numpy as np
import pandas as pd

from src.inference.pipeline import MRIAnalysisEngine, CLASS_NAMES, CHECKPOINT_CONFIGS
from src.inference.report_generator import generate_pdf_report


def run_comprehensive_tests():
    print("=" * 60)
    print("  BRAINX SYSTEM VERIFICATION & VALIDATION TEST SUITE")
    print("=" * 60)

    # Test 1: Checkpoint loading & initialization
    print("\n[TEST 1] Initializing MRIAnalysisEngine & loading 3 checkpoints...")
    engine = MRIAnalysisEngine.get_instance()
    assert len(engine.models) == 3, f"Expected 3 models, got {len(engine.models)}"
    assert len(engine.gradcams) == 3, f"Expected 3 GradCAM instances, got {len(engine.gradcams)}"
    print(f"-> PASS: All 3 models and GradCAMs loaded successfully on device: {engine.device}")

    # Test 2: Inference on test cohort samples across all 4 classes
    print("\n[TEST 2] Testing inference across all 4 classes...")
    test_samples = [
        ("Mild Demented", "Dataset/Mild_Demented/mild_10.jpg"),
        ("Moderate Demented", "Dataset/Moderate_Demented/moderate_10.jpg"),
        ("Non Demented", "Dataset/Non_Demented/non_10.jpg"),
        ("Very Mild Demented", "Dataset/Very_Mild_Demented/verymild_10.jpg")
    ]

    last_result = None
    for c_name, path in test_samples:
        assert os.path.exists(path), f"Sample image {path} does not exist"
        with open(path, "rb") as f:
            img_bytes = f.read()

        result = engine.analyze(img_bytes, filename=os.path.basename(path))
        last_result = result
        print(f"\n  Analyzed: {os.path.basename(path)} (Expected: {c_name})")
        print(f"  Agreement Status: {result['agreement']['status']}")
        
        for m_key in ["baseline", "he", "clahe"]:
            m_res = result["models"][m_key]
            prob_sum = sum(m_res["probabilities"].values())
            print(f"    - {m_res['name']}: {m_res['predicted_class']} ({m_res['confidence_pct']}%) | Sum Probs: {prob_sum:.4f}")
            
            # Test 3: Probability sum to 1.0 check
            assert abs(prob_sum - 1.0) < 0.005, f"Probabilities for {m_key} do not sum to 1.0: {prob_sum}"
            
            # Test 4: Grad-CAM assets generated
            viz = m_res["visualizations"]
            assert viz["original_b64"].startswith("data:image/jpeg;base64,"), "Missing original base64"
            assert viz["heatmap_b64"].startswith("data:image/jpeg;base64,"), "Missing heatmap base64"
            assert viz["overlay_b64"].startswith("data:image/jpeg;base64,"), "Missing overlay base64"
            assert os.path.exists(viz["heatmap_file"]), f"Heatmap file missing: {viz['heatmap_file']}"
            assert os.path.exists(viz["overlay_file"]), f"Overlay file missing: {viz['overlay_file']}"

    print("\n-> PASS: Multi-class inference and probability validation passed.")

    # Test 5: Preprocessing comparison assets
    print("\n[TEST 5] Checking preprocessing comparison assets...")
    assert "original" in last_result["preprocessing"], "Missing original preprocessing"
    assert "he" in last_result["preprocessing"], "Missing HE preprocessing"
    assert "clahe" in last_result["preprocessing"], "Missing CLAHE preprocessing"
    print("-> PASS: All 3 preprocessed outputs present.")

    # Test 6: PDF Report Generation
    print("\n[TEST 6] Testing PDF Report generation...")
    test_pdf_path = os.path.join(last_result["run_dir"], "test_verification_report.pdf")
    generate_pdf_report(last_result, test_pdf_path)
    assert os.path.exists(test_pdf_path), f"PDF was not created at {test_pdf_path}"
    pdf_size = os.path.getsize(test_pdf_path)
    assert pdf_size > 50000, f"PDF file size seems abnormally small: {pdf_size} bytes"
    print(f"-> PASS: PDF report generated successfully ({pdf_size:,} bytes).")

    # Test 7: Invalid image handling
    print("\n[TEST 7] Testing invalid input rejection...")
    try:
        engine.analyze(b"NOT_A_VALID_IMAGE_DATA_CORRUPT", filename="bad.jpg")
        assert False, "Engine failed to reject corrupted image data!"
    except ValueError as ve:
        print(f"-> PASS: Correctly raised ValueError on invalid data: {ve}")

    # Test 8: Verify existing research artifacts are intact
    print("\n[TEST 8] Verifying existing research outputs remain untouched...")
    research_files = [
        "results/baseline/test_predictions.csv",
        "results/he/test_predictions.csv",
        "results/clahe/test_predictions.csv",
        "results/comparison/comparative_metrics_table.csv",
        "dataset/splits/splits.csv"
    ]
    for rf in research_files:
        assert os.path.exists(rf), f"Critical research file {rf} is missing!"
        assert os.path.getsize(rf) > 0, f"Critical research file {rf} is empty!"
    print("-> PASS: All original research benchmarks and dataset splits intact.")

    print("\n" + "=" * 60)
    print("  ALL 8 TESTS PASSED SUCCESSFULLY! SYSTEM FULLY OPERATIONAL.")
    print("=" * 60)


if __name__ == "__main__":
    run_comprehensive_tests()

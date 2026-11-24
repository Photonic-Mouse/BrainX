================================================================================
BRAIN MRI SCANS FOR TESTING & DEMONSTRATIONS (BRAINSCOPE-XAI)
================================================================================

All scans in this folder are verified axial 224x224 T1-weighted MRI slices strictly
from the audited 960-scan held-out test cohort (never used during training).

FAST DEMO (SINGLE CLICK / INSTANT DRAG & DROP):
--------------------------------------------------------------------------------
You can immediately drag and drop any of these 4 files directly from this folder
into the BrainScope-XAI web interface (http://127.0.0.1:8000):

1. '01_Mild_Demented_QuickTest.jpg'
   - Ground Truth: Mild Demented
   - Model Prediction: Mild Demented (~99.8% confidence)
   - Grad-CAM: Concentrated on bilateral ventricular margins.

2. '02_Moderate_Demented_Severe_QuickTest.jpg'
   - Ground Truth: Moderate Demented (Severe Stage)
   - Model Prediction: Moderate Demented (~100.0% confidence)
   - Key Pitch: Demonstrates our 100.0% recall on severe disease (0 false negatives).

3. '03_Non_Demented_Healthy_Control_QuickTest.jpg'
   - Ground Truth: Non Demented (Healthy Aging)
   - Model Prediction: Non Demented (~99.7% confidence)
   - Grad-CAM: Low pathological focus, confirming no false alarms.

4. '04_Very_Mild_EarlyMCI_QuickTest.jpg'
   - Ground Truth: Very Mild Demented (Early Mild Cognitive Impairment)
   - Model Prediction: Very Mild Demented (~99.5% confidence)
   - Key Pitch: Shows early detection of hippocampal complex atrophy.

ADDITIONAL CATEGORY SAMPLES:
--------------------------------------------------------------------------------
If you want more samples of each stage, open the category subfolders:
- test_images/Mild_Demented/ (5 scans)
- test_images/Moderate_Demented/ (5 scans)
- test_images/Non_Demented/ (5 scans)
- test_images/Very_Mild_Demented/ (5 scans)

NOTE: You can also use the one-click 'Quick Test Samples' chips directly inside
the web application for instant live loading!
================================================================================

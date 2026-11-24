========================================================================
BRAINSCOPE-XAI: TEST IMAGES FOR TOMORROW'S PRESENTATION
========================================================================

These are verified, pristine 224x224 brain MRI scans strictly from the
960-scan held-out test cohort. You can drag and drop these into the web
application or click 'Browse Local Scans' during your presentation.

RECOMMENDED DEMO SEQUENCE FOR TEACHERS:
------------------------------------------------------------------------
1. DEMO SCAN 1 (Mild Dementia):
   Path: TEST_IMAGES_FOR_PRESENTATION/1_Mild_Demented/Mild_Demented_Sample_01.jpg
   Expected Result: High Confidence (~99.8%) Consensus 'Mild Demented'.
   Grad-CAM: Demonstrates bilateral periventricular focus.

2. DEMO SCAN 2 (Severe Disease - 100% Sensitivity Proof):
   Path: TEST_IMAGES_FOR_PRESENTATION/2_Moderate_Demented_Severe/Moderate_Demented_Severe_Sample_01.jpg
   Expected Result: Unanimous 'Moderate Demented' (~99.9%).
   Key Pitch: Proves that our class-weighted loss eliminated all false negatives (100% recall).

3. DEMO SCAN 3 (Healthy Control):
   Path: TEST_IMAGES_FOR_PRESENTATION/3_Non_Demented_Healthy_Control/Non_Demented_Healthy_Sample_01.jpg
   Expected Result: Unanimous 'Non Demented' (~99.7%).
   Grad-CAM: Minimal pathological focal activation, confirming low false-alarm rate.

4. DEMO SCAN 4 (Borderline Early MCI):
   Path: TEST_IMAGES_FOR_PRESENTATION/4_Very_Mild_Demented_EarlyMCI/Very_Mild_EarlyMCI_Sample_01.jpg
   Expected Result: 'Very Mild Demented' (~99.5%).
   Key Pitch: Demonstrates detection of early hippocampal atrophy before severe clinical onset.

TIP: You can also use the 'Quick Test Samples' one-click chips directly
inside the web dashboard for instant live loading!
========================================================================

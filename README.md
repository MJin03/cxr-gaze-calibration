# Radiologist Gaze Supervision for Chest X-ray Calibration

Research archive for:
“Radiologist Gaze Supervision for Calibration and Selective
Prediction in Chest X-ray Classification.”

## Overview

This study examines radiologist gaze supervision in chest X-ray
classification, focusing on discrimination, predictive calibration,
and uncertainty-based selective prediction.

## Archive Status

This repository contains the original recovered Colab notebook,
a conversation-based reconstruction of subsequent experiment code,
and historical reported results.

The reconstructed script is not a recovered server-original file.
End-to-end reproduction using the original data and checkpoints
has not been verified.

See [provenance](docs/provenance.md) for details.

## Experimental Design

- Dataset: EGD-CXR linked to MIMIC-CXR-JPG
- Cohort: 1,083 images
- Classes: Normal, CHF, Pneumonia
- Patient-level train/validation/test split
- Backbone: DenseNet-121
- Conditions: Baseline, Original Gaze, Shuffled Gaze,
  Random Gaze, Lesion-Supervised

## Main Findings

- Original gaze supervision did not improve discrimination
  over the image-only baseline.
- Disrupted image–gaze correspondence was associated with
  poorer calibration in the reported experiment.
- At a 25% test referral rate, baseline uncertainty-based referral
  captured approximately 48.1% of model errors.
- Error detection was weaker for Pneumonia.

These findings are based on the reported training run;
independent training-run variability was not evaluated.

## Reported Results

Historical result tables are available in
[results/reported](results/reported/).

## Repository Structure

- `code/`: recovered notebook and reconstructed experiment script
- `results/reported/`: historical reported metrics
- `docs/`: methodology, provenance, and figures

## Data Access

Dataset files are not included.
Obtain access through the respective dataset providers and
follow their data-use requirements.

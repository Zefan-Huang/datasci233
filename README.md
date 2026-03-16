# Multimodal Graph Reasoning for Implicit Metastasis Pathway Inference and Survival Prediction

> **For the full methodology and results, see [1. REPORT.md](1.%20REPORT.md).**

This repository contains a staged multimodal pipeline for non-small cell lung
cancer (NSCLC) prediction and explanation built on two public datasets:
NSCLC Radiogenomics as the main patient-level cohort for CT, RNA, clinical,
and outcome modeling, and CT-ORG as the auxiliary dataset for organ
segmentation and anatomical alignment. The project combines CT imaging,
clinical tabular features, RNA expression, and immune-derived signals into an
organ-level representation, then applies cross-attention and graph reasoning to
predict survival and recurrence while exporting latent organ diffusion
explanations.

## What The Project Does

- Predicts overall survival (OS)
- Predicts recurrence risk and recurrence location
- Exports organ susceptibility scores, inter-organ diffusion tendencies, and
  top-k latent paths as explanation outputs
- Produces cohort-level and patient-level visualization artifacts
- Supports external-case inference and report generation

The explanation layer is latent and task-constrained. It should be interpreted
as a model-induced explanation layer, not organ-level metastatic ground truth.

## Pipeline Structure

The repository is organized as numbered stage scripts. They remain at the
repository root because several scripts dynamically load one another by file
path.

- `prepare_clean/`: manifest building, label construction, RNA alignment, and
  clinical feature engineering implementations
- `5.2` to `6.2`: imaging preprocessing, organ segmentation training, and
  organ-mask inference
- `7.1` to `8.2`: RNA alignment, RNA encoder, immune token generation, and EHR
  encoder
- `9.1` to `11.2`: organ tokenization, organ queries, multimodal fusion, graph
  construction, and graph reasoning
- `12.1` to `12.2`: primary-task outputs and explanation outputs/training
- `13.0` to `13.5`: phased training, tuning, diffusion visualization, and the
  paired research heatmap
- `15.1` to `15.4`: case packaging, report generation, bundle export, and
  external-case inference

## Most Useful Entry Points

- `prepare_clean/total_table.py`: build the patient manifest
- `prepare_clean/label_construction_time_zero.py`: build OS and recurrence
  labels
- `9.1_organ_tokenization.py`: assemble organ-level inputs
- `10.1_multimodal_fusion.py`: run cross-attention fusion
- `11.2_graph_reasoning.py`: export organ susceptibility and edge diffusion
  tensors
- `12.2_explanation_outputs.py`: write explanation CSV/JSON outputs
- `13.1_phase3_baseline.py`: 211-case baseline training
- `13.2_tune_phase4.py`: tuned RNA-subset training
- `13.4_visualize_diffusion.py`: render diffusion SVG dashboards
- `13.5_result_heatmap.py`: render the paired publication-style result heatmap
- `15.4_external_case_inference.py`: end-to-end inference for an external case

## Key Output Locations

- `output/stage12/12.1_primary_outputs_cv/`: cross-validation metrics
- `output/stage12/12.2_explanation_outputs_joint/`: joint explanation CSV/JSON
- `output/stage13/13.1_phase3_baseline/`: baseline phase summary
- `output/stage13/13.2_phase4_tune/`: tuned final result summaries
- `output/stage13/13.4_visualize_diffusion/`: cohort and patient SVG
  visualizations
- `output/stage13/13.5_result_heatmap/`: paired Organ x Patient heatmap
  exports
- `output/stage15/`: case-level reports and inference bundles

## Current Research Figure

The latest paired heatmap compares Stage 12 joint explanation outputs against
the Stage 13 tuned final explanation outputs using a shared patient clustering
order.

- Script: `13.5_result_heatmap.py`
- Main outputs:
  - `output/stage13/13.5_result_heatmap/organ_patient_heatmap_compare.svg`
  - `output/stage13/13.5_result_heatmap/organ_patient_heatmap_compare.png`
  - `output/stage13/13.5_result_heatmap/organ_patient_heatmap_compare.pdf`

## Documentation

- `project.md`: full project design and staged pipeline description
- `REPORT.md`: report-style summary of motivation, method, workflow, and
  results
- `prepare_clean/README.md`: notes for the data-preparation subdirectory

## Notes

- `data/`, `output/`, and `.venv/` are ignored by Git in the default setup.
- The root directory is intentionally script-heavy because the current workflow
  depends on path-based dynamic loading between numbered stage files.

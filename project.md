# Multimodal Graph Reasoning for Implicit Metastasis Pathway Inference and Survival Prediction

This document describes the repository as it currently exists. It replaces the earlier plan-style description with an implementation-level summary that matches the code, outputs, and training results in this project.

## 1. Project goal

This repository implements a staged multimodal pipeline for non-small cell lung cancer (NSCLC) with three main objectives:

1. Predict overall survival (OS)
2. Predict recurrence risk and recurrence location
3. Produce organ-level latent diffusion explanations in the form of organ susceptibility, edge diffusion tendencies, and top-k organ paths

The key modeling choice is that organ/path outputs are treated as a latent explanation layer rather than organ-level ground-truth supervision. The public cohort supports survival and recurrence labels much more reliably than true metastatic path labels, so the prediction tasks are supervised directly, while the diffusion graph is constrained indirectly by those tasks.

## 2. Datasets actually used

### 2.1 NSCLC Radiogenomics

This is the primary patient-level cohort used throughout the project.

It provides:
- CT for all cases
- PET for most cases
- tumor segmentations for a subset
- AIM semantic annotations for a subset
- clinical follow-up labels
- RNA expression for a subset

The current manifest in `output/patient_manifest.csv` contains:
- 211 patients with CT
- 201 with PET
- 144 with tumor segmentation
- 190 with AIM annotations
- 130 with RNA-seq

This cohort is the source of:
- survival labels
- recurrence labels
- recurrence location labels
- multimodal patient-level modeling

### 2.2 CT-ORG

This is the auxiliary dataset used for organ segmentation and anatomical alignment.

In the current implementation, CT-ORG is not a second outcome cohort. It is used to support the stage-6 organ segmentation module so that patient CT can be converted into organ-level imaging evidence.

## 3. Repository structure and actual pipeline

The repository is organized as a stage-based workflow. The implemented path is:

### 3.1 Data cleaning and label construction

Files:
- `prepare_clean/total_table.py`
- `prepare_clean/label_construction_time_zero.py`
- `prepare_clean/imaging_preprocessing.py`
- `5.2_stage5_tumor_mask_provider_batch.py`

This part of the pipeline:
- builds the patient manifest
- aligns clinical rows, imaging availability, AIM files, segmentation availability, and RNA IDs
- constructs OS and recurrence labels using CT time as time zero
- prepares normalized imaging inputs and tumor masks

### 3.2 Organ segmentation and organ imaging evidence

Files:
- `6.1_unet.py`
- `6.1_seg_model.py`
- `6.2_infer_mask.py`

This stage trains or applies an organ segmentation model and exports organ masks or organ-level imaging evidence. The resulting masks are used downstream to create organ imaging tokens.

Existing outputs include organ segmentation artifacts under:
- `output/experiments/organ_seg/`

### 3.3 RNA, immune, and clinical encoding

Files:
- `7.1_rna_alignment.py`
- `7.2_rna_encoder.py`
- `7.3_immune_token.py`
- `8.1_clinical_feature_engineering.py`
- `8.2_ehr_encoder.py`

This part aligns RNA expression to patient IDs, creates RNA embeddings, derives immune-related tokens from RNA, engineers clinical variables, and creates EHR embeddings.

### 3.4 Organ tokenization

Files:
- `9.1_organ_tokenization.py`
- `9.2_organ_query.py`

The project defines a fixed six-node organ set:
- `Primary`
- `Lung`
- `Bone`
- `Liver`
- `LymphNodeMediastinum`
- `Brain`

At this stage, the project creates:
- evidence tokens from imaging, tumor, semantics, RNA, immune, and clinical data
- fixed organ queries for the six organ nodes

This is the bridge between raw multimodal features and the later graph.

### 3.5 Stage 10: multimodal cross-attention fusion

File:
- `10.1_multimodal_fusion.py`

This stage implements `OrganCrossAttentionFusion`.

What it does:
- Query = fixed organ queries
- Key/Value = multimodal evidence tokens
- Output = fused organ embeddings `Z`

Important implementation detail:
- this stage is currently run in `eval()` mode
- it uses `torch.no_grad()`
- the exported summary marks it as `random_init_only: True`

So Stage 10 is implemented and used, but it is not trained as a supervised fusion block in the current pipeline. It acts as a fixed random-weight transform.

### 3.6 Stage 11: graph construction and graph reasoning

Files:
- `11.1_graph_construction.py`
- `11.2_graph_reasoning.py`

Stage 11.1 builds the six-node organ graph with weak anatomical priors and residual logits.

The current graph uses prior edges such as:
- `Primary <-> Lung`
- `Primary <-> LymphNodeMediastinum`
- `Primary -> Bone`
- `Primary -> Liver`
- `Primary -> Brain`
- `LymphNodeMediastinum -> Bone/Liver/Brain`

Stage 11.2 applies a graph transformer style reasoning module:
- input: `Z`
- output: `Z_prime`
- explanation heads: organ susceptibility and edge diffusion probabilities

Important implementation detail:
- this stage is also run in `eval()` mode
- it also uses `torch.no_grad()`
- the exported summary marks it as `random_init_only: True`

So Stage 11.2 is also a fixed reservoir-style transform in the current codebase rather than a separately trained GNN.

### 3.7 Stage 12: prediction heads and explanation-guided training

Files:
- `12.1_primary_outputs.py`
- `12.2_explanation_outputs.py`
- `12.2_explanation_training.py`

This is where the main supervised learning actually happens.

`12.2_explanation_training.py` trains an `ExplanationGuidedPrimaryModel` on top of `Z_prime`. The model:
- pools `Z_prime`
- computes organ susceptibility and edge diffusion probabilities
- forms context features from susceptibility and outgoing primary-node edges
- predicts OS
- predicts recurrence
- predicts recurrence location
- adds auxiliary explanation-related losses and edge-prior regularization

In other words:
- Stage 10 and Stage 11.2 are fixed feature transforms
- Stage 12 is the trainable prediction layer that makes the project meaningful as a supervised model

### 3.8 Stage 13: phased training, tuning, comparison, and visualization

Files:
- `13.1_phase3_baseline.py`
- `13.2_phase4_rna_finetune.py`
- `13.2_tune_phase4.py`
- `13.3_compare_phases.py`
- `13.4_visualize_diffusion.py`
- `13.5_result_heatmap.py`

This stage packages the pipeline into reportable experiments:

- Phase 3: 211-patient baseline without RNA
- Phase 4: 130-patient RNA fine-tuning
- Phase 4 tuning: multiple runs and best-run selection
- phase comparison summaries
- cohort-level and patient-level diffusion visualizations
- paired organ-by-patient heatmap for stage-12 vs tuned stage-13 explanation outputs

### 3.9 Stage 15: case bundling and external inference

Files:
- `15.1_case_inputs.py`
- `15.2_system_outputs.py`
- `15.3_run_inference_bundle.py`
- `15.4_external_case_inference.py`

This stage turns the research pipeline into a deployable reporting flow. It supports:
- packaging an internal case
- generating system outputs and HTML artifacts
- running an external-case inference smoke test

Existing outputs include:
- `output/stage15/15.4_external_case_inference_smoke_R01-003/`
- `output/stage15/console_api_smoke/`

## 4. What is actually trained

This is the most important implementation clarification.

### 4.1 Not trained end-to-end

The following blocks are currently used as fixed transforms:
- Stage 10 cross-attention fusion
- Stage 11.2 graph reasoning

They are instantiated, run in inference mode, and exported. They are not optimized jointly with the final objectives in the current codebase.

### 4.2 Trained with supervision

The main trainable model is the Stage-12 explanation-guided prediction model, which is then used in Stage 13 for:
- baseline training
- RNA fine-tuning
- tuning and model selection

So the scientific claim supported by the current code is:
- the project trains a supervised prediction layer on top of fixed multimodal graph features

It is not yet a fully end-to-end trained transformer-plus-GNN model.

## 5. Current result files and main outputs

The main reportable results live in `output/`.

### 5.1 Stage 12 cross-validation

Directory:
- `output/stage12/12.1_primary_outputs_cv/`

Key file:
- `cv_summary.json`

Main out-of-fold metrics:
- OOF C-index: `0.5307`
- OOF recurrence AUC: `0.5667`
- OOF location accuracy: `0.3765`

### 5.2 Phase 3 baseline on 211 patients

Directory:
- `output/stage13/13.1_phase3_baseline/`

Key file:
- `phase3_summary.json`

Validation metrics:
- C-index: `0.6421`
- recurrence AUC: `0.6749`
- location accuracy: `0.4615`

This is the strongest survival-oriented result in the current project.

### 5.3 Initial Phase 4 RNA fine-tuning on 130 patients

Directory:
- `output/stage13/13.2_phase4_rna_finetune/`

Key file:
- `phase4_summary.json`

Validation metrics:
- C-index: `0.5340`
- recurrence AUC: `0.5263`
- location accuracy: `0.4444`

This run shows that adding RNA does not automatically improve performance when the multimodal subset is smaller.

### 5.4 Tuned Phase 4 best run

Directory:
- `output/stage13/13.2_phase4_tune/`

Key files:
- `tuning_summary.json`
- `best_config.json`
- `phase4_model_best/`
- `explanation_outputs_best/`

Best tuned validation metrics:
- C-index: `0.5109`
- recurrence AUC: `0.6959`
- location accuracy: `0.3333`

This is the best recurrence discrimination result in the repository.

### 5.5 Explanation outputs

Directory:
- `output/stage13/13.2_phase4_tune/explanation_outputs_best/`

Key files:
- `explanation_summary.json`
- `organ_susceptibility.csv`
- `edge_diffusion_long.csv`
- `topk_paths.json`
- `patient_explanations.json`

Current interpretation summary:
- top-1 path is usually `Primary -> Bone`
- in the best tuned run, the top-1 path fraction is `0.9769`
- explanation outputs are explicitly marked as latent and not organ-level ground truth

### 5.6 Visualization outputs

Directories:
- `output/stage13/13.4_visualize_diffusion/`
- `output/stage13/13.5_result_heatmap/`

These include:
- cohort diffusion SVGs
- patient-level diffusion SVGs
- a paired stage-12 vs stage-13 research heatmap

## 6. Files supporting the report

Main documentation files:
- `README.md`
- `Eyes_On_Me_Not_The_Code_Plz.md`
- `project.md`

Main figure assets already created:
- `report_assets/academic_architecture_transformer_gnn_joint.svg`
- `report_assets/cohort_primary_diffusion.svg`
- `report_assets/R01-151_primary_diffusion.svg`
- `report_assets/R01-106_primary_diffusion.svg`
- `report_assets/R01-049_primary_diffusion.svg`
- `report_assets/R01-026_primary_diffusion.svg`

## 7. Known limitations of the current project

1. Stage 10 and Stage 11.2 are not trained end-to-end
2. Explanation outputs are latent model-induced structures, not verified metastasis-path labels
3. The RNA subset is smaller than the full cohort, so RNA-based fine-tuning is less stable for survival
4. Recurrence-location labels are sparse, which limits location accuracy

These limitations should be stated clearly in any report or presentation.

## 8. Minimal interpretation of the current scientific contribution

The current repository supports the following accurate claim:

This project implements an end-to-end staged multimodal NSCLC pipeline that integrates Radiogenomics patient data and CT-ORG-based anatomical alignment, constructs organ-level representations, applies fixed transformer-style fusion and graph reasoning, and trains an explanation-guided prediction model for survival and recurrence tasks. It also exports deployable inference artifacts, cohort-level and patient-level visualizations, and a research-style explanation heatmap.

The current repository does not yet support the stronger claim that the transformer fusion and graph reasoner were themselves learned end-to-end from supervision.

## 9. If you want to run the project in stage order

The current logical order is:

1. `prepare_clean/total_table.py`
2. `prepare_clean/label_construction_time_zero.py`
3. `prepare_clean/imaging_preprocessing.py`
4. `5.2_stage5_tumor_mask_provider_batch.py`
5. `6.1_unet.py`
6. `6.2_infer_mask.py`
7. `7.1_rna_alignment.py`
8. `7.2_rna_encoder.py`
9. `7.3_immune_token.py`
10. `8.1_clinical_feature_engineering.py`
11. `8.2_ehr_encoder.py`
12. `9.1_organ_tokenization.py`
13. `10.1_multimodal_fusion.py`
14. `11.1_graph_construction.py`
15. `11.2_graph_reasoning.py`
16. `12.1_primary_outputs.py`
17. `12.2_explanation_outputs.py`
18. `12.2_explanation_training.py`
19. `13.1_phase3_baseline.py`
20. `13.2_phase4_rna_finetune.py`
21. `13.2_tune_phase4.py`
22. `13.3_compare_phases.py`
23. `13.4_visualize_diffusion.py`
24. `13.5_result_heatmap.py`
25. `15.1_case_inputs.py`
26. `15.2_system_outputs.py`
27. `15.3_run_inference_bundle.py`
28. `15.4_external_case_inference.py`

This order reflects the implemented repository, not the older aspirational plan.

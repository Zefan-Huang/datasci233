# Multimodal Organ Diffusion Report for NSCLC

## Intro

This project was originally conceived as my capstone project, but I ended up building it as a final project for this course instead. In this project, I built a multimodal prediction pipeline for non-small cell lung cancer (NSCLC) with three explicit goals: predicting overall survival (OS), predicting recurrence and its location, and producing structured organ-level diffusion paths as an explanation of how the model distributes latent metastatic spread tendency. Traditional clinical models often focus on one modality only, such as imaging or tabular clinical variables, and they usually provide limited insight into how risk may be distributed across organs. I wanted to design a system that could integrate CT imaging, clinical information, RNA expression, and immune-related signals into a single framework while still producing outputs that are clinically meaningful across all three goals.

For OS, I used a Cox-based survival head trained on time-to-event labels. For recurrence, I trained a classification head that jointly predicts whether recurrence occurs and, when location labels are available, distinguishes local, regional, and distant patterns. For diffusion paths, I introduced an organ diffusion explanation layer built on top of the graph reasoning module: it produces per-organ susceptibility scores, edge-level diffusion probabilities, and ranked top-k organ-to-organ paths extracted by beam search. This distinction is important: the diffusion outputs are not direct organ-level ground truth labels, because the public cohort does not provide reliable metastatic path supervision. Instead, they act as model-guided explanations constrained by the primary survival and recurrence tasks.

## Method

**System Overview.** I implemented a multimodal machine learning pipeline to predict patient overall survival (OS) and infer cancer metastasis pathways. The pipeline takes three types of data: CT scans, RNA expression profiles, and clinical tabular data (EHR). It processes these inputs, fuses them into specific anatomical "organ nodes," passes them through a graph network to simulate cancer spread, and finally outputs a survival risk score along with a predicted metastasis path.

**Feature Extraction and Fusion.** I used dedicated encoders to extract features from the raw data: 3D CNNs for CT images, MLPs for RNA embeddings, and tabular encoders for the clinical records. A major challenge in this dataset was missing data (e.g., many patients lacked bone CT scans). To solve this, I defined a fixed set of six logical "organ queries" (Primary Tumor, Lung, Bone, Liver, Lymph Node, and Brain). I used a Cross-Attention module to fuse the data: the organ queries attend to the available evidence tokens. If a specific CT scan is missing, the cross-attention mechanism naturally allows that organ query to gather relevant risk information from the global RNA or clinical data instead, avoiding the need for manual data imputation.

**Graph Reasoning and Reservoir Computing.** After fusion, the six organ tokens ($Z$) are passed into a Graph Transformer to model how cancer cells might spread between organs. I incorporated basic medical knowledge into the graph by initializing the edges with clinical priors (e.g., stronger connections from the primary tumor to lymph nodes). Crucially, because my dataset is relatively small (211 patients in total), training the entire Cross-Attention and Graph Transformer modules end-to-end would inevitably lead to severe overfitting. To solve this, I drew inspiration from the principles of Reservoir Computing and Extreme Learning Machines (ELM). Instead of updating the millions of parameters in these transformer modules, I froze them with their random initializations. In this paradigm, the cross-attention and graph networks act as a fixed, high-dimensional "reservoir." They non-linearly mix the multimodal features and project them into a complex topological space, allowing the downstream model to learn the survival patterns without the risk of parameter bloat.

**Joint Prediction and Latent Supervision.** Only the final prediction layers (the Explanation-Guided Primary Model, consisting of several MLPs) were trained using gradient descent. Instead of treating the metastasis pathway prediction as an afterthought, I integrated it into the main loop. The graph outputs ($Z'$) are used to calculate organ susceptibility and edge probabilities. Since I do not have ground-truth labels for the exact metastasis pathways, I used the patient's actual survival time as the primary training signal via Cox Proportional Hazards loss. By optimizing the survival prediction, the model is forced to adjust its internal pathway probabilities to logically explain the survival outcome. This allows the model to output a clinically meaningful metastasis graph without requiring explicit pathway labels.

## Workflow

1. I constructed a patient manifest and time-zero labels for 211 patients.
2. I preprocessed CT data, matched tumor segmentation, and trained or applied the organ segmentation model.
3. I generated RNA, immune, and EHR embeddings for the patients with available data.
4. I assembled all modality outputs into organ-level tokens, fused them, and performed graph-based reasoning.
5. I trained the model in phases, beginning with a larger non-RNA baseline and then fine-tuning on the smaller RNA subset.
6. I exported both quantitative outputs and visual diffusion reports, and I also verified that the pipeline can run an external-case inference smoke test.

## Results

### Quantitative Performance

The full cohort contains 211 patients. CT was available for all 211 cases, PET for 201, tumor segmentation for 144, AIM semantic annotations for 190, and RNA for 130. This distribution already explains one of the core challenges of the project: the richest multimodal setting is also the smallest one.

| Experiment | Cohort | C-index | Recurrence AUC | Location Accuracy |
| --- | --- | ---: | ---: | ---: |
| Stage 12 cross-validation baseline | 211 | 0.531 | 0.567 | 0.377 |
| Phase 3 baseline without RNA | 211 | 0.642 | 0.675 | 0.462 |
| Initial Phase 4 RNA fine-tuning | 130 | 0.534 | 0.526 | 0.444 |
| Best tuned Phase 4 run | 130 | 0.511 | 0.696 | 0.333 |

These numbers show a clear pattern. The strongest survival-oriented validation result came from the Phase 3 baseline trained on the larger 211-patient cohort without RNA. However, after tuning, the Phase 4 RNA-based model achieved the best recurrence discrimination, reaching a recurrence AUC of 0.696. This suggests that RNA information can improve recurrence modeling, but only after careful tuning, and that the small size of the RNA subset still limits stability for survival and location prediction.

The explanation outputs were also structured and consistent. In the best tuned RNA-subset model, the dominant top path was `Primary -> Bone`, appearing in 97.7% of cases. The highest mean organ susceptibility scores were Lung (0.751), Brain (0.728), and Primary (0.714). I interpret these as model-internal risk allocation patterns rather than biological truth claims.

### Visual Results

The cohort-level visualization below summarizes the dominant diffusion structure learned by the best tuned model.

![Figure 1. Cohort-level primary diffusion summary.](output/stage13/13.4_visualize_diffusion/cohort_primary_diffusion.svg)

*Figure 1. Cohort-level diffusion summary for the best tuned model. It highlights the dominant organ-to-organ explanation pattern across the RNA subset, with bone emerging as the main destination in the latent diffusion layer.*

To make the results more concrete, I also include representative patient-level figures from the generated visualization set.

![Figure 2. Patient R01-151 diffusion map.](output/stage13/13.4_visualize_diffusion/patients/R01-151_primary_diffusion.svg)

*Figure 2. Patient `R01-151` is a high-risk distant-recurrence example with strong bone and liver diffusion tendencies, making it a useful illustration of the model's high-confidence explanation behavior.*

![Figure 3. Patient R01-106 diffusion map.](output/stage13/13.4_visualize_diffusion/patients/R01-106_primary_diffusion.svg)

*Figure 3. Patient `R01-106` was predicted as a regional case, showing that the model can produce a different recurrence-location prediction while still maintaining a structured organ diffusion pattern.*

![Figure 4. Patient R01-049 diffusion map.](output/stage13/13.4_visualize_diffusion/patients/R01-049_primary_diffusion.svg)

*Figure 4. Patient `R01-049` was predicted as a local case. I include it because it shows that the latent path ranking and the supervised recurrence-location label are related but not identical.*

![Figure 5. Patient R01-026 diffusion map.](output/stage13/13.4_visualize_diffusion/patients/R01-026_primary_diffusion.svg)

*Figure 5. Patient `R01-026` is a strong bone-dominant example with very high top-path probabilities, making it a representative high-separation case for qualitative inspection.*

Beyond the training results, I also confirmed that the system can generate an external-case inference report in HTML. This means the project is not only a modeling experiment, but also a deployable reporting prototype.

## Conclusion

Overall, this project demonstrates that a multimodal NSCLC pipeline can jointly support prediction and structured explanation. I successfully integrated imaging, clinical, RNA, and immune features into an organ-level architecture with graph reasoning, and I produced both quantitative outputs and visual reports. The results indicate that the larger non-RNA baseline remains more stable for survival modeling, while the tuned RNA-based model offers the strongest recurrence discrimination. This means multimodal learning is promising, but it requires careful tuning and is still constrained by sample size.

The most important conclusion is that the framework is feasible and extensible. It already supports end-to-end processing, cohort-level visualization, patient-level diffusion figures, and external-case reporting. At the same time, the explanation layer should be interpreted carefully, because it provides latent task-guided structure rather than organ-level truth. In future work, stronger validation on larger multimodal cohorts would be necessary to confirm whether the learned diffusion patterns reflect clinically reliable biology.

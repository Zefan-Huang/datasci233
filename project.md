# Plan A: Multi-modal “Organ Diffusion” Predictor (Deployable Version) — Full Pipeline :contentReference[oaicite:0]{index=0}

> **Core idea**: Use **RNA expression** as the backbone signal (global molecular context). Align imaging (CT/PET), clinical tabular data (EHR-like), and immune features derived from RNA into **organ-level tokens**. Fuse modalities via **Cross-Attention**, then perform structured reasoning over an **organ topology graph**.  
> **Key tradeoff**: Public cohorts (e.g., NSCLC Radiogenomics) often lack organ-level “true metastasis sites/paths” labels. Therefore, the **primary supervised tasks** are objectively verifiable **OS/recurrence (time-to-event)**, while organ diffusion (nodes/edges/paths) is produced as a **latent explanation layer**—without claiming organ-level ground-truth supervision.

---

## 0. Verifiable objectives and outputs

### 0.1 Primary supervised tasks (trainable, evaluable)
1) **Overall Survival (OS)** modeling: time-to-event (Cox or discrete-time hazard)  
2) **Recurrence risk + recurrence site (mandatory task)**: Recurrence (yes/no), Recurrence Location (e.g., local/regional/distant; depends on available fields)

### 0.2 Explanation outputs (latent diffusion; not supervised with organ truth)
- **Organ susceptibility profile**: one score per organ `s_o` (relative susceptibility / spread tendency)
- **Inter-organ diffusion tendency matrix**: edge tendency `p_{i→j}`
- **Top-k diffusion paths**: most likely paths derived from `p_{i→j}` (for interpretation)

---

## 1. Data selection and roles

### 1.1 Primary cohort (required): NSCLC Radiogenomics (TCIA)
**Use**: main multimodal training + primary task evaluation (OS/recurrence)  
**Includes**: CT, PET/CT, tumor semantic annotations (AIM), tumor segmentations (DICOM-SEG), clinical follow-up, molecular data (RNA-seq subset).

> Imaging coverage is typically lung apex → adrenal glands; therefore “distal organs (e.g., brain) missing in imaging” is common and requires explicit missingness handling.

### 1.2 Alignment / organ segmentation training set (recommended): CT-ORG (TCIA)
**Use**: train/calibrate an organ segmentation model so “imaging → organ ROI → organ token” is reliable  
**Includes**: 140 CTs + multi-organ 3D segmentations (lung/bones/liver/kidneys/bladder; some brain)

> CT-ORG does not need patient-level pairing with Radiogenomics; it’s for training/benchmarking the organ alignment module.

### 1.3 (Optional but strongly recommended) imaging pretraining cohort: NSCLC-Radiomics (Lung1)
**Use**: pretrain the imaging encoder to mitigate small sample size in the primary cohort  
**Includes**: 422 NSCLC CTs + outcomes

---

## 2. Data retrieval and on-disk organization (engineering setup)

### 2.1 NSCLC Radiogenomics (TCIA + GEO)
**Download**
- CT (DICOM)
- optional PET/CT (DICOM)
- tumor segmentation: DICOM Segmentation Object (DICOM-SEG)
- semantic annotations: AIM files
- clinical CSV
- RNA-seq: GEO dataset (e.g., GSE103584; raw/processed)

**Suggested directory layout**
- `data/nsclc_rg/imaging/<PatientID>/<Study>/<Series>/*.dcm`
- `data/nsclc_rg/seg/<PatientID>/*.dcm` (DICOM-SEG)
- `data/nsclc_rg/aim/<PatientID>/*.xml`
- `data/nsclc_rg/clinical/clinical.csv`
- `data/nsclc_rg/rnaseq/GSE103584/*`

### 2.2 CT-ORG (TCIA)
**Download**
- CT NIfTI
- organ label NIfTI
- official train/test split (if provided)

**Suggested directory layout**
- `data/ctorg/images/*.nii.gz`
- `data/ctorg/labels/*.nii.gz`
- `data/ctorg/split/train.txt`
- `data/ctorg/split/test.txt`

---

## 3. Build a patient-modality availability manifest

### 3.1 Generate `patient_manifest.csv`
**Inputs**
- TCIA patient list
- `clinical.csv`
- existence checks: CT/PET, DICOM-SEG, AIM, RNA-seq (GEO subset)

**Recommended fields (minimal)**
- `patient_id`
- `has_ct`, `has_pet`, `has_seg`, `has_aim`, `has_rnaseq`
- OS: `event_os`, `time_os`
- recurrence: `event_rec`, `time_rec`, `rec_location_class` (mapped from clinical fields)

> Modalities are not fully available for every patient; missingness must be managed explicitly.

### 3.2 Define two training cohorts
- **Cohort-A (larger)**: ~211 cases (imaging + clinical) → robust baseline
- **Cohort-B (multimodal subset)**: ~130 cases (imaging + clinical + RNA) → multimodal fine-tuning

---

## 4. Label construction and time zero (avoid leakage)

### 4.1 Time zero `t0`
Use **CT date** as time=0 (since CT is part of the input).

### 4.2 OS labels
**Inputs**: survival status, date of death, date of last known alive  
**Outputs**
- `time_os = (death_or_last_alive_date) - t0`
- `event_os = 1 if dead else 0`

### 4.3 Recurrence labels (mandatory)
**Inputs**: recurrence, recurrence location, date of recurrence  
**Outputs**
- `event_rec`
- `time_rec` (censored if no recurrence)
- `rec_location_class` (encode available classes)

> Never feed future event dates into the model as input features—use them only as labels.

---

## 5. Imaging preprocessing (CT/PET/SEG/AIM → tensors)

### 5.1 CT normalization
**Steps**
1) select the primary CT series (rules by description/thickness/kernel)
2) DICOM → volume (prefer NIfTI)
3) resample to uniform spacing (e.g., 1–2 mm isotropic)
4) HU clipping + normalization (e.g., clip `[-1000, 400]`)

**Outputs**
- `ct_volume ∈ R^{D×H×W}`
- `ct_meta` (spacing/origin, etc.)

### 5.2 Tumor segmentation (DICOM-SEG → mask)
**Output**
- `mask_tumor ∈ {0,1}^{D×H×W}` aligned to `ct_volume`

### 5.3 Tumor ROI token (most stable imaging signal)
**Steps**
- bbox + margin crop → `ct_tumor_patch`
- 3D encoder (3D CNN / Swin3D / ResNet3D)
- pooling → token

**Output**
- `t_tumor ∈ R^d`

### 5.4 Semantic annotation token (optional but valuable)
**Steps**
- parse AIM categorical/continuous fields
- embedding/one-hot + normalization
- MLP → token(s)

**Outputs**
- `t_sem ∈ R^d` or `T_sem = {t_sem^k}`

---

## 6. Organ segmentation and organ imaging tokens (enhancement)

> You can run Plan A without organ segmentation first; add this module for stronger organ-level alignment.

### 6.1 Train/calibrate an organ segmentation model on CT-ORG
**Model**: nnU-Net / 3D U-Net / SwinUNETR  
**Focus organs**: lung / bone / liver (most relevant to chest/upper abdomen FOV)

### 6.2 Infer organ masks on NSCLC Radiogenomics CT
**Outputs**
- `mask_organ[o]` for visible organs
- `missing_img_organ[o]` for not visible / not usable organs

### 6.3 Extract organ imaging tokens (ROI pooling)
**Method**
- backbone feature map + masked pooling, or ROI crop + encoder

**Output**
- `t_img[o] ∈ R^d` (may be missing)

---

## 7. RNA and immune tokens (GEO → `g_rna` + `t_imm`)

### 7.1 RNA alignment (GEO → PatientID)
**Outputs**
- `x_rna ∈ R^G` (e.g., log1p(TPM/FPKM) + standardization)

### 7.2 RNA encoder (backbone embedding)
**Steps**
- gene filtering (e.g., top-5k variance genes)
- RNA encoder (MLP / lightweight Transformer)

**Outputs**
- `g_rna ∈ R^d`
- optional `T_rna` (e.g., pathway tokens)

### 7.3 Immune / microenvironment token (derive from RNA first)
**Steps**
- compute immune signatures (ssGSEA/GSVA or marker sets)
- MLP → token

**Outputs**
- `t_imm ∈ R^d` (or multiple tokens)

---

## 8. Clinical/EHR token (`clinical.csv` → `g_ehr`)

### 8.1 Clinical feature engineering
- continuous: standardize + missing indicators
- categorical: embeddings (recommended) or one-hot
- exclude future outcome dates as inputs

**Output**
- `x_ehr ∈ R^p`

### 8.2 EHR encoder
**Model**: MLP (MVP)

**Output**
- `g_ehr ∈ R^d`

---

## 9. Organ tokenization (organ-level alignment)

### 9.1 Define organ node set `O`
Given FOV limits, start with “coverable + clinically relevant”:
- `Primary` (primary tumor node)
- `Lung`, `Bone`, `Liver`
- `LymphNode/Mediastinum` (proxy via semantics/clinical)
- `Brain` (often imaging-missing; infer from RNA/clinical + missing mask)

### 9.2 Organ queries (key: no organ×gene RNA required)
Learnable organ embedding per organ `e_o ∈ R^d`, modulated by global context:

- `q_o = e_o + MLP([g_rna, g_ehr, t_imm, t_tumor])`

**Output**
- `Q = { q_o | o ∈ O }`

### 9.3 Assemble evidence tokens (Keys/Values)
Example tokens:
- `t_tumor` (almost always present)
- `t_img[o]` (if organ visible)
- `t_sem` (if AIM exists)
- `g_rna` / `T_rna` (if RNA exists)
- `g_ehr`
- `t_imm`

**Output**
- token set `T = {tokens}` + `mask_missing_tokens`

---

## 10. Multimodal fusion (Cross-Attention with organ queries)

For each organ `o`:

**Inputs**
- Query: `q_o`
- Key/Value: `T`
- `mask_missing_tokens`

**Process**
- `h_o = CrossAttn(q_o, K=T, V=T, mask=missing)`
- `z_o = LN(q_o + h_o)`
- `z_o = LN(z_o + FFN(z_o))`

**Output**
- fused organ tokens `Z = { z_o }`

---

## 11. Organ-graph reasoning

### 11.1 Graph construction (weak priors + learnable residual)
**Nodes**: organs `O`  
**Sparse prior edges (example)**
- Primary ↔ Lung
- Primary ↔ LN/Mediastinum
- Primary → Bone / Liver / Brain (weak)
- LN → distant organs (weak)

> Priors provide inductive bias; they do **not** claim true biological routes.

### 11.2 Graph reasoning module
**Model**: GAT / Graph Transformer (1–3 layers)  
**Output**
- structured node representations `Z' = { z'_o }`

---

## 12. Heads: primary outputs + explanation outputs

### 12.1 Primary supervised outputs
#### A) OS survival head
**Aggregation options**
- attention pooling: `u = AttnPool(Z')`
- or weighted sum: `u = Σ_o α_o z'_o`

**Survival head**
- Cox: outputs `r_os` (log-risk)
- discrete time: outputs `hazard[t]` → survival curve

#### B) Recurrence head (mandatory)
Outputs:
- `P(recurrence)`
- `P(recurrence_location = k)`

### 12.2 Explanation outputs (latent diffusion)
#### A) Organ susceptibility
- `s_o = sigmoid(MLP(z'_o))`

#### B) Edge diffusion tendency matrix
- `p_{i→j} = sigmoid(MLP([z'_i, z'_j, e_{ij}]))`

#### C) Top-k path explanation (derived)
- beam search / k-shortest paths from `Primary` using `P_edge`

> **Important**: organ/path outputs are a model-induced latent diffusion map to explain OS/recurrence predictions; do not claim organ-level ground-truth supervision.

---

## 13. Training strategy (phased; fits 211 + 130 structure)

### Phase 1: organ segmentation (CT-ORG)
- train `OrganSegModel`
- metrics: Dice/HD95 (lung/bone/liver, etc.)

### Phase 2 (optional): imaging encoder pretraining (NSCLC-Radiomics 422)
- self-supervised (MAE/contrastive) or weak supervision (OS in coarse time bins)

### Phase 3: baseline training on larger cohort (211: imaging + clinical)
- start without RNA for stability:
  - inputs: `t_tumor + g_ehr (+ t_sem)`
  - task: OS + recurrence (mandatory)

### Phase 4: multimodal fine-tuning (130: imaging + clinical + RNA)
- add RNA encoder, immune tokens, cross-attn, graph
- freeze parts of imaging encoder to reduce overfitting
- joint optimization: OS + recurrence (mandatory multi-task)

---

## 14. Loss design (primary supervision + stability regularization)

### 14.1 Primary losses
- OS: Cox partial likelihood or discrete-time NLL
- recurrence (mandatory): weighted BCE / CE
- label-missing mask for recurrence/location fields when specific records are unavailable

### 14.2 Missing-modality robustness (highly recommended)
- **modality dropout**: randomly drop tokens during training to simulate missingness
- **missingness masks**: explicitly mask missing tokens and invisible organs in attention

### 14.3 Light regularization for explanation layer (optional; small weight)
- edge sparsity regularization (avoid uniform fully-connected edges)
- graph smoothness (reduce noisy organ outputs)

---

## 15. Inference and deliverables

### 15.1 Case inputs
- CT (optional PET/CT)
- tumor segmentation (Radiogenomics provides it; external cases may need a segmenter)
- clinical tabular features
- RNA (if available)

### 15.2 System outputs
1) **Primary**
   - OS risk score / survival curve
   - recurrence probability + recurrence site probabilities

2) **Explanation**
   - organ susceptibility vector `{s_o}`
   - edge tendency matrix `{p_{i→j}}`
   - top-k diffusion paths (from Primary)

3) **Recommended interpretation materials**
   - cross-attn weights: which tokens each organ relied on (RNA/imaging/clinical/semantics)
   - graph attention/edge visualization: highest-contributing edges

---

## 16. Minimal runnable MVP (suggested implementation order)

1) **Radiogenomics (211)**: CT + tumor SEG + clinical → OS/recurrence baseline  
2) **Add RNA subset (130)**: RNA encoder + cross-attn (organ segmentation optional at this stage)  
3) **Bring in CT-ORG**: train organ segmenter; add lung/bone/liver organ imaging tokens (enhanced version)

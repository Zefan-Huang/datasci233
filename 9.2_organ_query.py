"""
Stage 9 organ tokenization modules for later multimodal fusion.

This file provides:
- a learned projector that assembles Stage 9 evidence tokens into a fixed-slot tensor
- a learned organ-query builder implementing q_o = e_o + MLP([g_rna, g_ehr, t_imm, t_tumor])
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ORGAN_NODE_NAMES = [
    "Primary",
    "Lung",
    "Bone",
    "Liver",
    "LymphNodeMediastinum",
    "Brain",
]
EVIDENCE_TOKEN_NAMES = [
    "t_tumor",
    "t_sem",
    "g_rna",
    "g_ehr",
    "t_imm",
    "t_img_primary",
    "t_img_lung",
    "t_img_bone",
    "t_img_liver",
    "t_img_lymphnode_mediastinum",
    "t_img_brain",
]


def _to_bool_mask(mask, expected_last_dim=None):
    if mask.ndim == 2 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if expected_last_dim is not None and mask.ndim == 1:
        return mask.to(torch.bool)
    return mask.to(torch.bool)


def _zero_fill_missing(vec, missing_mask):
    mask = missing_mask.to(dtype=vec.dtype)
    while mask.ndim < vec.ndim:
        mask = mask.unsqueeze(-1)
    return vec * (1.0 - mask)


def load_stage9_pack(npz_path, device=None):
    with np.load(npz_path, allow_pickle=True) as z:
        out = {}
        for key in z.files:
            value = z[key]
            if value.dtype.kind in {"U", "O"}:
                out[key] = value.astype(str)
            else:
                tensor = torch.from_numpy(value)
                out[key] = tensor if device is None else tensor.to(device)
    return out


class OrganEvidenceProjector(nn.Module):
    def __init__(
        self,
        d_model=128,
        img_dim=64,
        tumor_dim=64,
        sem_dim=64,
        imm_dim=64,
        rna_dim=128,
        ehr_dim=128,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.organ_node_names = list(ORGAN_NODE_NAMES)
        self.evidence_token_names = list(EVIDENCE_TOKEN_NAMES)

        self.proj_t_img = nn.Linear(int(img_dim), self.d_model, bias=False)
        self.proj_t_tumor = nn.Linear(int(tumor_dim), self.d_model, bias=False)
        self.proj_t_sem = nn.Linear(int(sem_dim), self.d_model, bias=False)
        self.proj_t_imm = nn.Linear(int(imm_dim), self.d_model, bias=False)
        self.proj_g_rna = nn.Linear(int(rna_dim), self.d_model, bias=False)
        self.proj_g_ehr = nn.Linear(int(ehr_dim), self.d_model, bias=False)

    def forward(
        self,
        t_img_nodes,
        t_img_missing,
        t_tumor,
        t_tumor_missing,
        t_sem,
        t_sem_missing,
        g_rna,
        g_rna_missing,
        g_ehr,
        g_ehr_missing,
        t_imm,
        t_imm_missing,
    ):
        if t_img_nodes.ndim != 3 or t_img_nodes.shape[1] != len(self.organ_node_names):
            raise RuntimeError(
                f"t_img_nodes must be [B,{len(self.organ_node_names)},D], got shape={tuple(t_img_nodes.shape)}"
            )

        t_img_missing = t_img_missing.to(torch.bool)
        t_tumor_missing = _to_bool_mask(t_tumor_missing)
        t_sem_missing = _to_bool_mask(t_sem_missing)
        g_rna_missing = _to_bool_mask(g_rna_missing)
        g_ehr_missing = _to_bool_mask(g_ehr_missing)
        t_imm_missing = _to_bool_mask(t_imm_missing)

        t_img_nodes = _zero_fill_missing(t_img_nodes, t_img_missing.unsqueeze(-1))
        t_tumor = _zero_fill_missing(t_tumor, t_tumor_missing.unsqueeze(-1))
        t_sem = _zero_fill_missing(t_sem, t_sem_missing.unsqueeze(-1))
        g_rna = _zero_fill_missing(g_rna, g_rna_missing.unsqueeze(-1))
        g_ehr = _zero_fill_missing(g_ehr, g_ehr_missing.unsqueeze(-1))
        t_imm = _zero_fill_missing(t_imm, t_imm_missing.unsqueeze(-1))

        t_img_proj = self.proj_t_img(t_img_nodes)
        t_tumor_proj = self.proj_t_tumor(t_tumor).unsqueeze(1)
        t_sem_proj = self.proj_t_sem(t_sem).unsqueeze(1)
        g_rna_proj = self.proj_g_rna(g_rna).unsqueeze(1)
        g_ehr_proj = self.proj_g_ehr(g_ehr).unsqueeze(1)
        t_imm_proj = self.proj_t_imm(t_imm).unsqueeze(1)

        img_primary = t_img_proj[:, 0:1, :]
        img_lung = t_img_proj[:, 1:2, :]
        img_bone = t_img_proj[:, 2:3, :]
        img_liver = t_img_proj[:, 3:4, :]
        img_ln = t_img_proj[:, 4:5, :]
        img_brain = t_img_proj[:, 5:6, :]

        T = torch.cat(
            [
                t_tumor_proj,
                t_sem_proj,
                g_rna_proj,
                g_ehr_proj,
                t_imm_proj,
                img_primary,
                img_lung,
                img_bone,
                img_liver,
                img_ln,
                img_brain,
            ],
            dim=1,
        )
        mask_missing_tokens = torch.cat(
            [
                t_tumor_missing.unsqueeze(1),
                t_sem_missing.unsqueeze(1),
                g_rna_missing.unsqueeze(1),
                g_ehr_missing.unsqueeze(1),
                t_imm_missing.unsqueeze(1),
                t_img_missing[:, 0:1],
                t_img_missing[:, 1:2],
                t_img_missing[:, 2:3],
                t_img_missing[:, 3:4],
                t_img_missing[:, 4:5],
                t_img_missing[:, 5:6],
            ],
            dim=1,
        )
        return T, mask_missing_tokens


class OrganQueryBuilder(nn.Module):
    def __init__(
        self,
        d_model=128,
        rna_dim=128,
        ehr_dim=128,
        imm_dim=64,
        tumor_dim=64,
        hidden_dim=256,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.organ_node_names = list(ORGAN_NODE_NAMES)
        context_input_dim = int(rna_dim) + int(ehr_dim) + int(imm_dim) + int(tumor_dim) + 4

        self.organ_embedding = nn.Parameter(torch.zeros(len(self.organ_node_names), self.d_model))
        nn.init.normal_(self.organ_embedding, mean=0.0, std=0.02)

        self.context_mlp = nn.Sequential(
            nn.Linear(context_input_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), self.d_model),
        )

    def forward(
        self,
        g_rna,
        g_rna_missing,
        g_ehr,
        g_ehr_missing,
        t_imm,
        t_imm_missing,
        t_tumor,
        t_tumor_missing,
    ):
        batch_size = int(g_ehr.shape[0])
        g_rna_missing = _to_bool_mask(g_rna_missing)
        g_ehr_missing = _to_bool_mask(g_ehr_missing)
        t_imm_missing = _to_bool_mask(t_imm_missing)
        t_tumor_missing = _to_bool_mask(t_tumor_missing)

        g_rna = _zero_fill_missing(g_rna, g_rna_missing.unsqueeze(-1))
        g_ehr = _zero_fill_missing(g_ehr, g_ehr_missing.unsqueeze(-1))
        t_imm = _zero_fill_missing(t_imm, t_imm_missing.unsqueeze(-1))
        t_tumor = _zero_fill_missing(t_tumor, t_tumor_missing.unsqueeze(-1))

        missing_flags = torch.stack(
            [
                g_rna_missing.float(),
                g_ehr_missing.float(),
                t_imm_missing.float(),
                t_tumor_missing.float(),
            ],
            dim=1,
        )
        context = torch.cat([g_rna, g_ehr, t_imm, t_tumor, missing_flags], dim=1)
        context_delta = self.context_mlp(context)
        queries = self.organ_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + context_delta.unsqueeze(1)
        return queries

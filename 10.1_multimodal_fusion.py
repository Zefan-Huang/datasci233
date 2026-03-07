"""
Stage 10 multimodal fusion (Cross-Attention with organ queries).

This file provides:
- a reusable OrganCrossAttentionFusion module
- a runnable entrypoint that consumes Stage 9 outputs and exports fused organ tokens Z

Process:
- Query: q_o from Stage 9 OrganQueryBuilder
- Key/Value: T from Stage 9 OrganEvidenceProjector
- Mask: mask_missing_tokens
- h_o = CrossAttn(q_o, K=T, V=T, mask=missing)
- z_o = LN(q_o + h_o)
- z_o = LN(z_o + FFN(z_o))
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


DEFAULT_STAGE9_PACK = Path("output/stage9/9.1_organ_tokenization/organ_tokenization_pack.npz")
DEFAULT_STAGE9_MODULE = Path("9.2_organ_query.py")
DEFAULT_OUTPUT_ROOT = Path("output/stage10/10.1_multimodal_fusion")


def set_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_required_path(path_arg, default_path, label):
    path = Path(path_arg) if str(path_arg).strip() else Path(default_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def ensure_output_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def load_module_from_path(module_path):
    module_path = Path(module_path)
    spec = importlib.util.spec_from_file_location("stage9_organ_query_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_device(device_arg):
    value = str(device_arg).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _as_bool_mask(mask):
    return mask.to(torch.bool)


def _batch_to_device(pack, start, end, device):
    return {
        key: value[start:end].to(device) if isinstance(value, torch.Tensor) else value[start:end]
        for key, value in pack.items()
    }


class OrganCrossAttentionFusion(nn.Module):
    def __init__(self, d_model=128, num_heads=8, ffn_hidden_dim=256, dropout=0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.ffn_hidden_dim = int(ffn_hidden_dim)
        self.dropout = float(dropout)

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, got d_model={self.d_model} num_heads={self.num_heads}"
            )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.ln_attn = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.ffn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.ffn_hidden_dim, self.d_model),
            nn.Dropout(self.dropout),
        )
        self.ln_ffn = nn.LayerNorm(self.d_model)

    def forward(self, queries, evidence_tokens, mask_missing_tokens, need_weights=True):
        if queries.ndim != 3:
            raise RuntimeError(f"queries must be [B,O,D], got shape={tuple(queries.shape)}")
        if evidence_tokens.ndim != 3:
            raise RuntimeError(
                f"evidence_tokens must be [B,T,D], got shape={tuple(evidence_tokens.shape)}"
            )
        if queries.shape[0] != evidence_tokens.shape[0]:
            raise RuntimeError(
                f"batch size mismatch between queries and evidence tokens: "
                f"{queries.shape[0]} vs {evidence_tokens.shape[0]}"
            )
        if queries.shape[2] != self.d_model or evidence_tokens.shape[2] != self.d_model:
            raise RuntimeError(
                f"d_model mismatch: expected={self.d_model} "
                f"got queries={queries.shape[2]} evidence={evidence_tokens.shape[2]}"
            )

        key_padding_mask = _as_bool_mask(mask_missing_tokens)
        if key_padding_mask.ndim != 2 or key_padding_mask.shape != evidence_tokens.shape[:2]:
            raise RuntimeError(
                f"mask_missing_tokens must be [B,T], got shape={tuple(key_padding_mask.shape)} "
                f"expected={tuple(evidence_tokens.shape[:2])}"
            )
        if torch.any(torch.all(key_padding_mask, dim=1)):
            bad_rows = torch.nonzero(torch.all(key_padding_mask, dim=1), as_tuple=False).view(-1)
            raise RuntimeError(
                "all evidence tokens are masked for batch rows: "
                + ",".join(str(int(x)) for x in bad_rows.tolist())
            )

        evidence_tokens = evidence_tokens.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        attn_output, attn_weights = self.cross_attn(
            query=queries,
            key=evidence_tokens,
            value=evidence_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        z_attn = self.ln_attn(queries + attn_output)
        z_ffn = self.ffn(z_attn)
        fused_tokens = self.ln_ffn(z_attn + z_ffn)

        if attn_weights is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
            denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            attn_weights = attn_weights / denom

        return fused_tokens, attn_weights


def build_stage10_inputs(stage9_module, pack, batch_size, device, d_model, seed):
    set_seed(seed)
    projector = stage9_module.OrganEvidenceProjector(
        d_model=d_model,
        img_dim=int(pack["t_img_nodes"].shape[-1]),
        tumor_dim=int(pack["t_tumor"].shape[-1]),
        sem_dim=int(pack["t_sem"].shape[-1]),
        imm_dim=int(pack["t_imm"].shape[-1]),
        rna_dim=int(pack["g_rna"].shape[-1]),
        ehr_dim=int(pack["g_ehr"].shape[-1]),
    ).to(device)
    query_builder = stage9_module.OrganQueryBuilder(
        d_model=d_model,
        rna_dim=int(pack["g_rna"].shape[-1]),
        ehr_dim=int(pack["g_ehr"].shape[-1]),
        imm_dim=int(pack["t_imm"].shape[-1]),
        tumor_dim=int(pack["t_tumor"].shape[-1]),
    ).to(device)
    projector.eval()
    query_builder.eval()

    num_patients = int(pack["g_ehr"].shape[0])
    evidence_chunks = []
    mask_chunks = []
    query_chunks = []

    with torch.no_grad():
        for start in range(0, num_patients, int(batch_size)):
            end = min(start + int(batch_size), num_patients)
            batch = _batch_to_device(pack, start, end, device)
            evidence_tokens, mask_missing_tokens = projector(
                t_img_nodes=batch["t_img_nodes"],
                t_img_missing=batch["t_img_missing"],
                t_tumor=batch["t_tumor"],
                t_tumor_missing=batch["t_tumor_missing"],
                t_sem=batch["t_sem"],
                t_sem_missing=batch["t_sem_missing"],
                g_rna=batch["g_rna"],
                g_rna_missing=batch["g_rna_missing"],
                g_ehr=batch["g_ehr"],
                g_ehr_missing=batch["g_ehr_missing"],
                t_imm=batch["t_imm"],
                t_imm_missing=batch["t_imm_missing"],
            )
            queries = query_builder(
                g_rna=batch["g_rna"],
                g_rna_missing=batch["g_rna_missing"],
                g_ehr=batch["g_ehr"],
                g_ehr_missing=batch["g_ehr_missing"],
                t_imm=batch["t_imm"],
                t_imm_missing=batch["t_imm_missing"],
                t_tumor=batch["t_tumor"],
                t_tumor_missing=batch["t_tumor_missing"],
            )
            evidence_chunks.append(evidence_tokens.cpu())
            mask_chunks.append(mask_missing_tokens.cpu())
            query_chunks.append(queries.cpu())

    evidence_tokens = torch.cat(evidence_chunks, dim=0)
    mask_missing_tokens = torch.cat(mask_chunks, dim=0)
    queries = torch.cat(query_chunks, dim=0)
    return projector, query_builder, evidence_tokens, mask_missing_tokens, queries


def run_stage10_fusion(
    stage9_pack_path,
    stage9_module_path,
    output_root,
    batch_size=64,
    d_model=128,
    num_heads=8,
    ffn_hidden_dim=256,
    dropout=0.1,
    device="auto",
    seed=1337,
):
    stage9_pack_path = Path(stage9_pack_path)
    stage9_module_path = Path(stage9_module_path)
    output_root = Path(output_root)
    ensure_output_dir(output_root)

    device = choose_device(device)
    set_seed(seed)

    stage9_module = load_module_from_path(stage9_module_path)
    pack = stage9_module.load_stage9_pack(stage9_pack_path, device=None)

    patient_ids = np.asarray(pack["patient_ids"]).astype(str)
    organ_node_names = np.asarray(pack["organ_node_names"]).astype(str)
    evidence_token_names = np.asarray(stage9_module.EVIDENCE_TOKEN_NAMES, dtype=str)

    _, _, evidence_tokens_cpu, mask_missing_tokens_cpu, queries_cpu = build_stage10_inputs(
        stage9_module=stage9_module,
        pack=pack,
        batch_size=batch_size,
        device=device,
        d_model=d_model,
        seed=seed,
    )

    fusion_model = OrganCrossAttentionFusion(
        d_model=d_model,
        num_heads=num_heads,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout=dropout,
    ).to(device)
    fusion_model.eval()

    num_patients = int(patient_ids.shape[0])
    fused_chunks = []
    weight_chunks = []

    with torch.no_grad():
        for start in range(0, num_patients, int(batch_size)):
            end = min(start + int(batch_size), num_patients)
            queries = queries_cpu[start:end].to(device)
            evidence_tokens = evidence_tokens_cpu[start:end].to(device)
            mask_missing_tokens = mask_missing_tokens_cpu[start:end].to(device)
            fused_tokens, attn_weights = fusion_model(
                queries=queries,
                evidence_tokens=evidence_tokens,
                mask_missing_tokens=mask_missing_tokens,
                need_weights=True,
            )
            fused_chunks.append(fused_tokens.cpu())
            weight_chunks.append(attn_weights.cpu())

    fused_tokens = torch.cat(fused_chunks, dim=0)
    attn_weights = torch.cat(weight_chunks, dim=0)

    npz_path = output_root / "fused_organ_tokens.npz"
    np.savez_compressed(
        npz_path,
        patient_ids=patient_ids,
        organ_node_names=organ_node_names,
        evidence_token_names=evidence_token_names,
        Z=fused_tokens.numpy().astype(np.float32),
        attn_weights=attn_weights.numpy().astype(np.float32),
    )

    manifest_path = output_root / "fusion_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "num_missing_evidence_tokens",
                "num_available_evidence_tokens",
                "num_missing_image_nodes",
                "num_available_image_nodes",
                "has_rna",
                "has_ehr",
                "has_immune",
                "has_t_tumor",
                "has_t_sem",
            ],
        )
        writer.writeheader()
        for idx, patient_id in enumerate(patient_ids.tolist()):
            mask_row = mask_missing_tokens_cpu[idx].to(torch.bool)
            img_missing = pack["t_img_missing"][idx].to(torch.bool)
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "num_missing_evidence_tokens": int(mask_row.sum().item()),
                    "num_available_evidence_tokens": int((~mask_row).sum().item()),
                    "num_missing_image_nodes": int(img_missing.sum().item()),
                    "num_available_image_nodes": int((~img_missing).sum().item()),
                    "has_rna": int(not bool(pack["g_rna_missing"][idx].item())),
                    "has_ehr": int(not bool(pack["g_ehr_missing"][idx].item())),
                    "has_immune": int(not bool(pack["t_imm_missing"][idx].item())),
                    "has_t_tumor": int(not bool(pack["t_tumor_missing"][idx].item())),
                    "has_t_sem": int(not bool(pack["t_sem_missing"][idx].item())),
                }
            )

    summary = {
        "stage": "10.1_multimodal_fusion",
        "stage9_pack_path": str(stage9_pack_path),
        "stage9_module_path": str(stage9_module_path),
        "patient_count": int(patient_ids.shape[0]),
        "organ_count": int(organ_node_names.shape[0]),
        "evidence_token_count": int(evidence_token_names.shape[0]),
        "d_model": int(d_model),
        "num_heads": int(num_heads),
        "ffn_hidden_dim": int(ffn_hidden_dim),
        "dropout": float(dropout),
        "batch_size": int(batch_size),
        "device": str(device),
        "seed": int(seed),
        "export_attention_weights": True,
        "random_init_only": True,
        "shapes": {
            "queries": list(queries_cpu.shape),
            "evidence_tokens": list(evidence_tokens_cpu.shape),
            "mask_missing_tokens": list(mask_missing_tokens_cpu.shape),
            "Z": list(fused_tokens.shape),
            "attn_weights": list(attn_weights.shape),
        },
    }
    summary_path = output_root / "fusion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "npz_path": str(npz_path),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "patient_count": int(patient_ids.shape[0]),
        "Z_shape": tuple(fused_tokens.shape),
        "attn_shape": tuple(attn_weights.shape),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Stage 10 multimodal fusion",
        allow_abbrev=False,
    )
    parser.add_argument("--stage9-pack", default=str(DEFAULT_STAGE9_PACK))
    parser.add_argument("--stage9-module", default=str(DEFAULT_STAGE9_MODULE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def parse_args():
    args, _unknown = build_arg_parser().parse_known_args()
    return args


def main():
    args = parse_args()
    stage9_pack_path = resolve_required_path(args.stage9_pack, DEFAULT_STAGE9_PACK, "stage9 pack")
    stage9_module_path = resolve_required_path(args.stage9_module, DEFAULT_STAGE9_MODULE, "stage9 module")
    result = run_stage10_fusion(
        stage9_pack_path=stage9_pack_path,
        stage9_module_path=stage9_module_path,
        output_root=Path(args.output_root),
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ffn_hidden_dim=args.ffn_hidden_dim,
        dropout=args.dropout,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

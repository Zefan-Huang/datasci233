from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "key_outputs" / "academic_architecture_transformer_gnn_joint.svg"

W, H = 2420, 910
FAM = "'Helvetica Neue', Arial, sans-serif"

# ── Palette ──────────────────────────────────────────────────────────────────
BLUE_BG,   BLUE_BD,   BLUE_H,   BLUE_B   = "#EFF6FF", "#93C5FD", "#1E40AF", "#1E3A5F"
AMBER_BG,  AMBER_BD,  AMBER_H,  AMBER_B  = "#FFFBEB", "#F59E0B", "#78350F", "#92400E"
GREEN_BG,  GREEN_BD,  GREEN_H,  GREEN_B  = "#F0FDF4", "#34D399", "#14532D", "#166534"
PURPLE_BG, PURPLE_BD, PURPLE_H, PURPLE_B = "#F5F3FF", "#A78BFA", "#3B0764", "#4C1D95"
DARK, MID, ARROW_C = "#0F172A", "#475569", "#64748B"
WH = "#FFFFFF"


def R(x, y, w, h, fill, stroke, sw=1.5, rx=10):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def T(x, y, s, sz=14, fill=DARK, fw=500, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-family="{FAM}" font-size="{sz}" '
            f'font-weight="{fw}" fill="{fill}" text-anchor="{anchor}">'
            f'{escape(str(s))}</text>')


def MT(x, y, lines, sz=12, fill=MID, fw=400, gap=17):
    spans = []
    for i, ln in enumerate(lines):
        dy = 0 if i == 0 else gap
        spans.append(f'<tspan x="{x}" dy="{dy}">{escape(ln)}</tspan>')
    return (f'<text x="{x}" y="{y}" font-family="{FAM}" font-size="{sz}" '
            f'font-weight="{fw}" fill="{fill}">{"".join(spans)}</text>')


def badge(x, y, label, bg, bd, tc):
    fw = len(label) * 7 + 22
    return (R(x, y, fw, 24, bg, bd, sw=1.5, rx=12)
            + T(x + fw // 2, y + 16, label, sz=12, fill=tc, fw=700, anchor="middle"))


def arr(x1, y1, x2, y2, color=ARROW_C, sw=2.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round"{d} marker-end="url(#arw)"/>')


def box(x, y, w, h, title, lines, bg=WH, bd=MID, th=DARK, tb=MID, tsz=13, bsz=12):
    """Component box: rounded rect + title + horizontal rule + bullet lines."""
    sep = y + tsz + 8
    els = [
        R(x, y, w, h, bg, bd, sw=1.5),
        T(x + 10, y + tsz + 3, title, sz=tsz, fill=th, fw=700),
        f'<line x1="{x+8}" y1="{sep+3}" x2="{x+w-8}" y2="{sep+3}" '
        f'stroke="{bd}" stroke-width="0.8" stroke-opacity="0.7"/>',
    ]
    cy = sep + 16
    for ln in lines:
        els.append(T(x + 12, cy, ln, sz=bsz, fill=tb, fw=400))
        cy += bsz + 5
    return "".join(els)


def sec_header(x, y, w, num, title, sub, hc):
    return (
        T(x + 14, y + 24, f"{num}  {title}", sz=17, fill=hc, fw=800)
        + T(x + 14, y + 42, sub, sz=12, fill=MID, fw=400)
        + f'<line x1="{x+12}" y1="{y+50}" x2="{x+w-12}" y2="{y+50}" '
          f'stroke="{hc}" stroke-width="1.2" stroke-opacity="0.3"/>'
    )


def build_svg():
    g = []

    # ── SVG open + defs ──────────────────────────────────────────────────────
    g.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}">')
    g.append(f"""<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#FAFCFF"/>
    <stop offset="100%" stop-color="#F1F5F9"/>
  </linearGradient>
  <marker id="arw" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
    <path d="M0,0 L9,4.5 L0,9 Z" fill="{ARROW_C}"/>
  </marker>
</defs>""")

    # ── Background + border ──────────────────────────────────────────────────
    g.append(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')
    g.append(R(12, 12, W - 24, H - 24, "none", "#CBD5E1", sw=1.5, rx=22))

    # ── Title ────────────────────────────────────────────────────────────────
    g.append(T(34, 50, "Multimodal Graph Reasoning  ·  NSCLC Survival & Metastasis Pathway Inference",
               sz=23, fill=DARK, fw=800))
    g.append(T(34, 72, "Staged architecture: frozen transformer reservoir  →  trainable explanation-guided prediction heads",
               sz=13, fill=MID))

    # ── Column geometry ──────────────────────────────────────────────────────
    # (x, width) for each of 5 columns
    SY, SH = 84, 762
    PAD = 12
    cols = [
        (22,  292),   # 0  Inputs
        (354, 378),   # 1  Fusion      (Frozen)
        (782, 352),   # 2  Graph       (Frozen)
        (1184, 578),  # 3  Prediction  (Trainable)
        (1812, 572),  # 4  Outputs
    ]
    # right edges: 314, 732, 1134, 1762, 2384  — all fit in W=2420

    colors = [
        (BLUE_BG,   BLUE_BD,   BLUE_H),
        (AMBER_BG,  AMBER_BD,  AMBER_H),
        (AMBER_BG,  AMBER_BD,  AMBER_H),
        (GREEN_BG,  GREEN_BD,  GREEN_H),
        (PURPLE_BG, PURPLE_BD, PURPLE_H),
    ]

    for (cx, cw), (cbg, cbd, _) in zip(cols, colors):
        g.append(R(cx, SY, cw, SH, cbg, cbd, sw=2, rx=18))

    hdrs = [
        ("①", "Multimodal Inputs",     "Raw patient data tokens"),
        ("②", "Cross-Attention Fusion", "Organ query alignment"),
        ("③", "Graph Reasoning",        "Metastatic topology"),
        ("④", "Joint Prediction",       "Explanation-guided heads"),
        ("⑤", "Outputs",               "Predictions & pathways"),
    ]
    for (cx, cw), (_, _, ch), (n, t, s) in zip(cols, colors, hdrs):
        g.append(sec_header(cx, SY, cw, n, t, s, ch))

    # ── Status badges ────────────────────────────────────────────────────────
    cx1, cw1 = cols[1]
    cx2, cw2 = cols[2]
    cx3, cw3 = cols[3]
    g.append(badge(cx1 + cw1 - 88,  SY + 8, "FROZEN",    "#FEF3C7", "#D97706", "#92400E"))
    g.append(badge(cx2 + cw2 - 88,  SY + 8, "FROZEN",    "#FEF3C7", "#D97706", "#92400E"))
    g.append(badge(cx3 + cw3 - 110, SY + 8, "TRAINABLE", "#DCFCE7", "#16A34A", "#14532D"))

    # ── inner box x / w helpers ──────────────────────────────────────────────
    def bx(ci): return cols[ci][0] + PAD
    def bw(ci): return cols[ci][1] - 2 * PAD

    # ════════════════════════════════════════════════════════════════════════
    # COL 0 — Inputs
    # ════════════════════════════════════════════════════════════════════════
    x0, w0 = bx(0), bw(0)

    g.append(box(x0, 152, w0, 108, "CT Imaging Tokens",
                 ["• Organ imaging features (U-Net)",
                  "• Tumor mask ROI",
                  "• Semantic AIM descriptors"],
                 bg=WH, bd=BLUE_BD, th=BLUE_H, tb=BLUE_B))

    g.append(box(x0, 272, w0, 88, "Molecular Tokens",
                 ["• RNA global embedding  g_rna",
                  "• Immune signature token  t_imm"],
                 bg=WH, bd=BLUE_BD, th=BLUE_H, tb=BLUE_B))

    g.append(box(x0, 372, w0, 70, "Clinical (EHR)",
                 ["• Tabular feature encoding  g_ehr"],
                 bg=WH, bd=BLUE_BD, th=BLUE_H, tb=BLUE_B))

    g.append(box(x0, 454, w0, 162, "Fixed Organ Queries",
                 ["• Primary Tumor",
                  "• Lung  ·  Bone  ·  Liver",
                  "• Lymph Node / Mediastinum",
                  "• Brain"],
                 bg="#DBEAFE", bd=BLUE_H, th=BLUE_H, tb=BLUE_B))

    g.append(R(x0, 630, w0, 84, "#EFF6FF", BLUE_BD, sw=1, rx=8))
    g.append(MT(x0 + 10, 648,
                ["If a modality is absent, the organ",
                 "query attends to other available",
                 "evidence — no imputation needed."],
                sz=12, fill=MID))

    # ════════════════════════════════════════════════════════════════════════
    # COL 1 — Cross-Attention Fusion
    # ════════════════════════════════════════════════════════════════════════
    x1, w1 = bx(1), bw(1)
    hw = (w1 - 8) // 2

    g.append(box(x1,      152, hw, 84, "Evidence Projector",
                 ["• Projects tokens → T",
                  "• Shared dim D"],
                 bg=WH, bd=AMBER_BD, th=AMBER_H, tb=AMBER_B))
    g.append(box(x1+hw+8, 152, hw, 84, "Query Builder",
                 ["• Q from global signals",
                  "• + missing-modality flags"],
                 bg=WH, bd=AMBER_BD, th=AMBER_H, tb=AMBER_B))

    g.append(box(x1, 248, w1, 184, "OrganCrossAttentionFusion",
                 ["• Z = LN(Q + Attn(Q, K=T, V=T)) + FFN",
                  "• Multi-head,  d=128,  8 heads",
                  "• Missing organ → attends globally",
                  "• Output: organ token matrix  Z"],
                 bg="#FFFDF0", bd=AMBER_BD, th=AMBER_H, tb=AMBER_B))

    g.append(R(x1, 444, w1, 34, "#FEF9C3", AMBER_BD, sw=1.5, rx=8))
    g.append(T(x1 + w1 // 2, 466, "Z  ·  [B × 6 × D]",
               sz=14, fill=AMBER_H, fw=700, anchor="middle"))

    g.append(R(x1, 490, w1, 88, AMBER_BG, AMBER_BD, sw=1, rx=8))
    g.append(MT(x1 + 10, 508,
                ["⚠  Random init  ·  eval()  ·  no_grad()",
                 "Fixed non-linear feature projector.",
                 "Acts as a reservoir — no gradient",
                 "updates in this stage."],
                sz=12, fill="#92400E"))

    g.append(R(x1, 592, w1, 30, AMBER_BG, AMBER_BD, sw=1, rx=8))
    g.append(T(x1 + 10, 612, "Stage 10  ·  OrganCrossAttentionFusion",
               sz=12, fill=AMBER_B))

    # ════════════════════════════════════════════════════════════════════════
    # COL 2 — Graph Reasoning
    # ════════════════════════════════════════════════════════════════════════
    x2, w2 = bx(2), bw(2)

    g.append(box(x2, 152, w2, 116, "Graph Construction",
                 ["• Strong: Primary ↔ Lung, Lymph",
                  "• Weak: Primary → Bone, Liver, Brain",
                  "• Adj = sigmoid(prior + residual)"],
                 bg=WH, bd=AMBER_BD, th=AMBER_H, tb=AMBER_B))

    g.append(box(x2, 280, w2, 172, "Graph Transformer",
                 ["• scores = QKᵀ/√d  +  adj_logits",
                  "• Non-candidate edges masked (−1e9)",
                  "• 2 layers,  8 heads,  residual + LN",
                  "• Maps Z → Z'  across organ nodes"],
                 bg="#FFFDF0", bd=AMBER_BD, th=AMBER_H, tb=AMBER_B))

    g.append(R(x2, 464, w2, 34, "#FEF9C3", AMBER_BD, sw=1.5, rx=8))
    g.append(T(x2 + w2 // 2, 486, "Z'  ·  [B × 6 × D]",
               sz=14, fill=AMBER_H, fw=700, anchor="middle"))

    g.append(R(x2, 510, w2, 88, AMBER_BG, AMBER_BD, sw=1, rx=8))
    g.append(MT(x2 + 10, 528,
                ["⚠  Random init  ·  eval()  ·  no_grad()",
                 "Reservoir-style fixed projector.",
                 "Injects prior topology into attn",
                 "without end-to-end gradients."],
                sz=12, fill="#92400E"))

    g.append(R(x2, 612, w2, 30, AMBER_BG, AMBER_BD, sw=1, rx=8))
    g.append(T(x2 + 10, 632, "Stage 11.1/11.2  ·  OrganGraphReasoner",
               sz=12, fill=AMBER_B))

    # ════════════════════════════════════════════════════════════════════════
    # COL 3 — Joint Prediction  (TRAINABLE)
    # ════════════════════════════════════════════════════════════════════════
    x3, w3 = bx(3), bw(3)
    lw = w3 * 5 // 12
    rw = w3 - lw - 8

    g.append(box(x3,      152, lw, 88, "Pool + Base Trunk",
                 ["• AttentionPool(Z') → u",
                  "• LN → Linear → ReLU → Dropout"],
                 bg=WH, bd=GREEN_BD, th=GREEN_H, tb=GREEN_B))
    g.append(box(x3+lw+8, 152, rw, 88, "Context Fusion",
                 ["• Concat(trunk, s-ctx, e-ctx, diff)",
                  "• → joint_trunk  via fusion MLP"],
                 bg=WH, bd=GREEN_BD, th=GREEN_H, tb=GREEN_B))

    g.append(box(x3, 252, w3, 160, "Latent Diffusion Explainer",
                 ["• Organ susceptibility:  s_o = σ( MLP(z'_o) )",
                  "• Edge diffusion:  p_{i→j} = σ( MLP([z'_i, z'_j, e_ij]) )",
                  "• s-context and e-context routed back into prediction trunk",
                  "• Explanation is in-loop, not post-hoc"],
                 bg="#DCFCE7", bd=GREEN_BD, th=GREEN_H, tb=GREEN_B))

    # 4 output head boxes
    HW = (w3 - 3 * 8) // 4
    head_data = [
        ("OS Head",    ["Cox / discrete hazard"]),
        ("Recurrence", ["Binary BCE loss"]),
        ("Location",   ["3-class CE loss"]),
        ("Expl. Aux.", ["λ-weighted losses"]),
    ]
    for i, (hn, hl) in enumerate(head_data):
        hx = x3 + i * (HW + 8)
        g.append(box(hx, 424, HW, 72, hn, hl,
                     bg=WH, bd=GREEN_BD, th=GREEN_H, tb=GREEN_B))

    g.append(box(x3, 508, w3, 118, "Joint Training Objective",
                 ["L = L_os + L_rec + L_loc + λ₁L_expl_rec + λ₂L_expl_loc + λ₃L_edge_prior",
                  "Adam  ·  lr=1e-3  ·  early stopping (patience=30)",
                  "Phase 3: N=211 (EHR+imaging)  →  Phase 4: N=130 (full multimodal)"],
                 bg="#F0FDF4", bd=GREEN_H, th=GREEN_H, tb=GREEN_B))

    g.append(R(x3, 638, w3, 28, GREEN_BG, GREEN_BD, sw=1, rx=8))
    g.append(T(x3 + 10, 657,
               "Stage 12–13  ·  ExplanationGuidedPrimaryModel  ·  Only trained module in the pipeline",
               sz=11, fill=GREEN_B))

    # ════════════════════════════════════════════════════════════════════════
    # COL 4 — Outputs
    # ════════════════════════════════════════════════════════════════════════
    x4, w4 = bx(4), bw(4)

    g.append(box(x4, 152, w4, 84, "Overall Survival",
                 ["• Cox log-risk score",
                  "• Discrete hazard trajectory"],
                 bg=WH, bd=PURPLE_BD, th=PURPLE_H, tb=PURPLE_B))

    g.append(box(x4, 248, w4, 68, "Recurrence Prediction",
                 ["• P(recurrence)  (binary)"],
                 bg=WH, bd=PURPLE_BD, th=PURPLE_H, tb=PURPLE_B))

    g.append(box(x4, 328, w4, 68, "Recurrence Location",
                 ["• Local  /  Regional  /  Distant"],
                 bg=WH, bd=PURPLE_BD, th=PURPLE_H, tb=PURPLE_B))

    g.append(box(x4, 408, w4, 156, "Implicit Metastasis Pathway",
                 ["• Per-organ susceptibility scores (6 nodes)",
                  "• Edge diffusion matrix  [6 × 6]",
                  "• Top-k diffusion paths from Primary",
                  "  (beam search, max 3 hops, width 8)"],
                 bg="#EDE9FE", bd=PURPLE_BD, th=PURPLE_H, tb=PURPLE_B))

    g.append(R(x4, 576, w4, 76, "#FAF5FF", PURPLE_BD, sw=1.5, rx=8))
    g.append(T(x4 + w4 // 2, 596, "Supervision Sources",
               sz=13, fill=PURPLE_H, fw=700, anchor="middle"))
    g.append(MT(x4 + 12, 614,
                ["Time-to-event labels  ·  Recurrence labels",
                 "Location labels  ·  Edge prior regularization"],
                sz=12, fill=PURPLE_B))

    # ════════════════════════════════════════════════════════════════════════
    # Main flow arrows  (horizontal, at y=480)
    # ════════════════════════════════════════════════════════════════════════
    AY = 480
    gaps = [
        (cols[0][0] + cols[0][1], cols[1][0]),  # C0 → C1
        (cols[1][0] + cols[1][1], cols[2][0]),  # C1 → C2
        (cols[2][0] + cols[2][1], cols[3][0]),  # C2 → C3
        (cols[3][0] + cols[3][1], cols[4][0]),  # C3 → C4
    ]
    lbls = ["tokens", "Z", "Z\u2019", "Z\u2019"]
    for (lx, rx), lbl in zip(gaps, lbls):
        mx = (lx + rx) // 2
        g.append(arr(lx + 4, AY, rx - 4, AY))
        g.append(T(mx, AY - 8, lbl, sz=12, fill=ARROW_C, fw=600, anchor="middle"))

    # Supervision feedback dashed arrow (curved below section)
    # from bottom of Supervision box in C4 back to bottom of Joint Loss in C3
    fb_x1 = x4 + w4 // 2
    fb_y1 = 652          # supervision box bottom
    fb_x2 = x3 + w3 // 2
    fb_y2 = 626          # joint loss box bottom
    mid_y  = SY + SH + 16
    g.append(
        f'<path d="M {fb_x1} {fb_y1} L {fb_x1} {mid_y} L {fb_x2} {mid_y} L {fb_x2} {fb_y2}" '
        f'fill="none" stroke="{PURPLE_BD}" stroke-width="2" stroke-dasharray="8 4" '
        f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arw)"/>'
    )
    g.append(T((fb_x1 + fb_x2) // 2, mid_y + 14, "latent supervision signal",
               sz=11, fill=PURPLE_H, fw=600, anchor="middle"))

    # ════════════════════════════════════════════════════════════════════════
    # Bottom note bar
    # ════════════════════════════════════════════════════════════════════════
    NY = SY + SH + 34
    g.append(R(20, NY, W - 40, 46, WH, "#CBD5E1", sw=1, rx=10))
    g.append(T(40, NY + 14, "Design note:", sz=12, fill=DARK, fw=700))
    g.append(T(132, NY + 14,
               "Stages ② and ③ use fixed random weights (no backprop). "
               "They act as a high-dimensional reservoir that non-linearly mixes multimodal features.",
               sz=12, fill=MID))
    g.append(T(40, NY + 30,
               "Stage ④ is the only trained module. "
               "Cox PH loss on survival outcomes serves as the latent supervisor for the metastasis pathway explanation.",
               sz=12, fill=MID))

    g.append("</svg>")
    return "".join(g)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_svg(), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

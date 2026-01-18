# -*- coding: utf-8 -*-
# Single-vector steering but measure BOTH persona axes (A=creative, B=environmentalist)

from rich import print
import torch, csv, time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== CONFIG =====================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Persona vectors (A: creative, B: environmentalist) – per-layer .pt (tensor list/array)
VEC_A_PATH = "Llama-3.1-8B-Instruct/multi_role/creative_professional_response_avg_diff.pt"
VEC_B_PATH = "Llama-3.1-8B-Instruct/multi_role/environmentalist_response_avg_diff.pt"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

STEER_LAYER = 20          # insert layer (0-based residual index)
# Choose steering strength per axis (set one of them to 0.0 for single-vector runs)
ALPHA_A    = 2.0          # strength along A (creative)
ALPHA_B    = 0.0          # strength along B (environmentalist) -> set to 2.0 if steering B
EPS_CAL    = 0.1          # small calibration step for both A and B
LAYER_WRITE_FROM = 19     # only write rows for layers >= this (focus on output-side)

ROLE_A = "creative professional"
ROLE_B = "environmentalist"

# # Prompts (neutral)
# NEUTRAL_LIST = [
#     "Urban Planning (2050 City Block Masterplan): Design a masterplan for a new city block to be built in 2050. Describe core principles, layout, mobility, public space, services, and governance constraints.",
#     "Product Launch (Micro-Teleportation for Small Objects): Outline a public launch plan for a micro-teleportation technology for small items. Include positioning, safety/regulation, go-to-market, operations, and risk.",
#     "Social Issue (Countering Misinformation): Propose a multi-pronged plan to reduce misinformation on social platforms: policy, product, incentives, literacy, measurement.",
#     "Corporate Strategy (Legacy Manufacturer vs. AI Disruption): Design a transformation strategy for a legacy manufacturer facing AI disruption: portfolio, org, tech stack, talent, risk, timeline.",
#     "Healthcare Innovation (Reimagine the Hospital): Redesign the future hospital experience for patients, families, and staff. Address flows, safety, data, wellbeing, equity, and feasibility.",
#     "Education Reform (Ideal High-School Curriculum): Propose a 4-year curriculum: core subjects, skills, experiential learning, assessment, inclusion, and teacher enablement.",
#     "Disaster Response (Early Recovery Plan for a Metro Area): Draft an initial 30–60 day recovery plan after a major natural disaster: assessment, triage, logistics, comms, governance, equity.",
#     "Space Exploration (Next 50 Years Priority): State and justify the top priority for human space exploration in the next 50 years. Define milestones, risks, ethics, and spillovers.",
#     "Sustainable Fashion (Net-Zero Brand Model): Propose a business model for a fully sustainable fashion brand: materials, supply chain, circularity, economics, verification, storytelling.",
#     "Global Challenge (Food Waste Reduction): Design a multi-layer plan to reduce global food waste across production, retail, and households: incentives, infra, tech, policy, culture."
# ]

AUT_LIST = [
    "What are some creative use for Fork? The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Present a list of 5 creative and diverse uses for Fork.",
    "What are some creative use for Jar? The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Present a list of 5 creative and diverse uses for Jar."
]

INS_LIST = [
    "Name all the round things you can think of.",
    "Name all the things you can think of that will make a noise.",
    "Name all the things you can think of that have a screen."
]

SIMI_LIST = [
    "Tell me all the ways in which a kite and a balloon are alike.",
    "Tell me all the ways in which a pencil and a pen are alike.",
    "Tell me all the ways in which a chair and a couch are alike."
]

SCI_LIST = [
    "If you can take a spaceship to travel in outer space and go to a planet, what scientific questions do you want to research? For example, are there any living things on the planet?",
    "Please think up as many possible improvements as you can to a regular bicycle, making it more interesting, more useful and more beautiful. For example, make the tires reflective, so they can be seen in the dark."
]
NEUTRAL_LIST = AUT_LIST + INS_LIST + SIMI_LIST + SCI_LIST

# ===================== UTILS =====================
@torch.no_grad()
def unit(v: torch.Tensor, eps=1e-8):
    v = v.float()
    return v / (v.norm() + eps)

@torch.no_grad()
def run_forward_capture(model, tokenizer, prompt, steer=False, steer_layer=None, steer_vec=None):
    """
    Single forward pass (no generation). Capture each layer's output (batch=0, last_token).
    If steer=True, add 'steer_vec' (already scaled/combined) to last token at 'steer_layer'.
    Returns dict: {layer_idx: hidden(last_token) [float32, cpu]}
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    kwargs = dict(use_cache=False, output_hidden_states=False, return_dict=True)

    layer_modules = model.model.layers
    captures = {}

    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            if isinstance(output, tuple):
                hs, rest = output[0], output[1:]
            else:
                hs, rest = output, None
            if steer and (layer_idx == steer_layer):
                add = steer_vec.to(hs.dtype).to(hs.device)  # [hidden]
                hs = hs.clone()
                hs[:, -1, :] = hs[:, -1, :] + add
            captures[layer_idx] = hs[0, -1, :].detach().float().cpu()
            return (hs,) + rest if rest is not None else hs
        return hook

    handles = []
    try:
        for i in range(len(layer_modules)):
            handles.append(layer_modules[i].register_forward_hook(make_hook(i), with_kwargs=True))
        _ = model(**inputs, **kwargs)
    finally:
        for h in handles:
            h.remove()
    return captures

@torch.no_grad()
def l2(x):  # L2 norm
    return float(x.float().norm().item())

@torch.no_grad()
def dot(a, b):  # dot product
    return float((a.float() @ b.float()).item())

# ===================== MAIN =====================
def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = f"steer_biax_measure_{ts}.csv"

    print(f"[bold cyan]Loading {MODEL_NAME} ...[/bold cyan]")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)

    # Load persona vectors (per-layer)
    vecA_all = torch.load(VEC_A_PATH, map_location="cpu")  # creative
    vecB_all = torch.load(VEC_B_PATH, map_location="cpu")  # environmentalist
    num_layers = len(model.model.layers)

    # Build unit directions
    vA20_u = unit(vecA_all[STEER_LAYER])
    vB20_u = unit(vecB_all[STEER_LAYER])
    vA_layer_u = {L: unit(vecA_all[L]) for L in range(num_layers)}
    vB_layer_u = {L: unit(vecB_all[L]) for L in range(num_layers)}

    # Open CSV
    fieldnames = [
        "prompt_idx","layer",
        "alpha_A","alpha_B","eps_A","eps_B","steer_layer",
        "delta_l2",
        "proj_A20","proj_B20",
        "proj_A_L","proj_B_L",
        "proj_A_transport","proj_B_transport",
        "model","vecA_path","vecB_path","roleA","roleB"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate prompts
        for i, prompt in enumerate(NEUTRAL_LIST):
            print(f"\n[bold magenta]=== Prompt {i+1}/{len(NEUTRAL_LIST)} ===[/bold magenta]")
            print(prompt[:100] + ("..." if len(prompt) > 100 else ""))

            # Baseline
            base = run_forward_capture(model, tok, prompt, steer=False)

            # --- Calibration A (small eps along A) ---
            steer_cal_A = (EPS_CAL * vA20_u)
            calA = run_forward_capture(
                model, tok, prompt,
                steer=True, steer_layer=STEER_LAYER, steer_vec=steer_cal_A
            )
            delta_cal_A = {L: (calA[L] - base[L]) for L in base.keys()}
            tA_unit = {L: unit(delta_cal_A[L]) for L in delta_cal_A.keys()}

            # --- Calibration B (small eps along B) ---
            steer_cal_B = (EPS_CAL * vB20_u)
            calB = run_forward_capture(
                model, tok, prompt,
                steer=True, steer_layer=STEER_LAYER, steer_vec=steer_cal_B
            )
            delta_cal_B = {L: (calB[L] - base[L]) for L in base.keys()}
            tB_unit = {L: unit(delta_cal_B[L]) for L in delta_cal_B.keys()}

            # --- Main steering (can be single-axis or combo via ALPHA_A / ALPHA_B) ---
            steer_main = (ALPHA_A * vA20_u + ALPHA_B * vB20_u)
            ste = run_forward_capture(
                model, tok, prompt,
                steer=True, steer_layer=STEER_LAYER, steer_vec=steer_main
            )
            delta = {L: (ste[L] - base[L]) for L in base.keys()}

            # --- Write layer-wise rows (focus 19..end) ---
            for L in range(num_layers):
                if L < LAYER_WRITE_FROM:
                    continue
                dL = delta[L]
                amp = l2(dL)  # ||Δh_L||

                # Fixed-layer axes (L20)
                pA20 = dot(dL, vA20_u)
                pB20 = dot(dL, vB20_u)

                # Per-layer persona bases
                pAL = dot(dL, vA_layer_u[L])
                pBL = dot(dL, vB_layer_u[L])

                # Transported directions (from small perturbations on each axis)
                pA_tp = dot(dL, tA_unit[L])
                pB_tp = dot(dL, tB_unit[L])

                writer.writerow({
                    "prompt_idx": i,
                    "layer": L,
                    "alpha_A": ALPHA_A,
                    "alpha_B": ALPHA_B,
                    "eps_A": EPS_CAL,
                    "eps_B": EPS_CAL,
                    "steer_layer": STEER_LAYER,
                    "delta_l2": amp,
                    "proj_A20": pA20,
                    "proj_B20": pB20,
                    "proj_A_L": pAL,
                    "proj_B_L": pBL,
                    "proj_A_transport": pA_tp,
                    "proj_B_transport": pB_tp,
                    "model": MODEL_NAME,
                    "vecA_path": VEC_A_PATH,
                    "vecB_path": VEC_B_PATH,
                    "roleA": ROLE_A,
                    "roleB": ROLE_B
                })

            # quick sanity at steer layer
            print(f"[Check] L{STEER_LAYER} A_tp≈{dot(delta[STEER_LAYER], tA_unit[STEER_LAYER]):.4f} (target {ALPHA_A}) | "
                  f"B_tp≈{dot(delta[STEER_LAYER], tB_unit[STEER_LAYER]):.4f} (target {ALPHA_B})")

    print(f"\n✅ Done. CSV saved to: [bold green]{out_csv}[/bold green]")

if __name__ == "__main__":
    main()

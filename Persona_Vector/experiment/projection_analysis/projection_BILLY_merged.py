from rich import print
import torch, csv, time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== 配置 =====================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
VEC_A_PATH = "Llama-3.1-8B-Instruct/multi_role/creative_professional_response_avg_diff.pt"
VEC_B_PATH = "Llama-3.1-8B-Instruct/multi_role/environmentalist_response_avg_diff.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

STEER_LAYER = 20
ALPHA_A, ALPHA_B = 1.0, 1.0      # 主實驗權重
EPS_A,   EPS_B   = 0.1, 0.1      # 校準小擾動（估計 transport）

# 只輸出這個層範圍
LAYER_FROM, LAYER_TO = 19, 32    # [19, 32] 含頭含尾

# 是否使用每層 persona 向量（若 .pt 有每層）
USE_PER_LAYER_PERSONA_VECS = True

NEUTRAL_LIST = [
    "Urban Planning (2050 City Block Masterplan): Design a masterplan for a new city block to be built in 2050. Describe core principles, layout, mobility, public space, services, and governance constraints.",
    "Product Launch (Micro-Teleportation for Small Objects): Outline a public launch plan for a micro-teleportation technology for small items. Include positioning, safety/regulation, go-to-market, operations, and risk.",
    "Social Issue (Countering Misinformation): Propose a multi-pronged plan to reduce misinformation on social platforms: policy, product, incentives, literacy, measurement.",
    "Corporate Strategy (Legacy Manufacturer vs. AI Disruption): Design a transformation strategy for a legacy manufacturer facing AI disruption: portfolio, org, tech stack, talent, risk, timeline.",
    "Healthcare Innovation (Reimagine the Hospital): Redesign the future hospital experience for patients, families, and staff. Address flows, safety, data, wellbeing, equity, and feasibility.",
    "Education Reform (Ideal High-School Curriculum): Propose a 4-year curriculum: core subjects, skills, experiential learning, assessment, inclusion, and teacher enablement.",
    "Disaster Response (Early Recovery Plan for a Metro Area): Draft an initial 30–60 day recovery plan after a major natural disaster: assessment, triage, logistics, comms, governance, equity.",
    "Space Exploration (Next 50 Years Priority): State and justify the top priority for human space exploration in the next 50 years. Define milestones, risks, ethics, and spillovers.",
    "Sustainable Fashion (Net-Zero Brand Model): Propose a business model for a fully sustainable fashion brand: materials, supply chain, circularity, economics, verification, storytelling.",
    "Global Challenge (Food Waste Reduction): Design a multi-layer plan to reduce global food waste across production, retail, and households: incentives, infra, tech, policy, culture."
]

# AUT_LIST = [
#     "What are some creative use for Fork? The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Present a list of 5 creative and diverse uses for Fork.",
#     "What are some creative use for Jar? The goal is to come up with creative ideas, which are ideas that strike people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Present a list of 5 creative and diverse uses for Jar."
# ]

# INS_LIST = [
#     "Name all the round things you can think of.",
#     "Name all the things you can think of that will make a noise.",
#     "Name all the things you can think of that have a screen."
# ]

# SIMI_LIST = [
#     "Tell me all the ways in which a kite and a balloon are alike.",
#     "Tell me all the ways in which a pencil and a pen are alike.",
#     "Tell me all the ways in which a chair and a couch are alike."
# ]

# SCI_LIST = [
#     "If you can take a spaceship to travel in outer space and go to a planet, what scientific questions do you want to research? For example, are there any living things on the planet?",
#     "Please think up as many possible improvements as you can to a regular bicycle, making it more interesting, more useful and more beautiful. For example, make the tires reflective, so they can be seen in the dark."
# ]
# NEUTRAL_LIST = AUT_LIST + INS_LIST + SIMI_LIST + SCI_LIST

# ===================== 工具 =====================
@torch.no_grad()
def unit(v: torch.Tensor, eps=1e-8):
    v = v.float()
    return v / (v.norm() + eps)

@torch.no_grad()
def run_forward_capture(model, tokenizer, prompt, steer=False, steer_layer=None, steer_vec_unit=None):
    """
    前向一次，hook 每層輸出，擷取 (batch=0, last_token)。
    若 steer=True，於 steer_layer 對最後一 token 加上 steer_vec_unit（已含係數與合成）。
    回傳 dict: layer_idx -> hidden(last_token) [float32, cpu]
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
                add = steer_vec_unit.to(hs.dtype).to(hs.device)
                hs = hs.clone()
                hs[:, -1, :] = hs[:, -1, :] + add  # 僅最後一 token
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
def dot(a, b):  # 內積
    return float((a.float() @ b.float()).item())

@torch.no_grad()
def cos_sim(a, b, eps=1e-8):
    na = a.float().norm().item()
    nb = b.float().norm().item()
    if na < eps or nb < eps:
        return 0.0
    return float((a.float() @ b.float()).item() / (na * nb))

# ===================== 主流程 =====================
def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = f"steer_merge_AB_layers_{LAYER_FROM}-{LAYER_TO}_{ts}.csv"

    print(f"[bold cyan]Loading {MODEL_NAME} ...[/bold cyan]")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)

    vecA = torch.load(VEC_A_PATH, map_location="cpu")
    vecB = torch.load(VEC_B_PATH, map_location="cpu")

    num_layers = len(model.model.layers)

    # 注入層方向（單位化）
    vA20_u = unit(vecA[STEER_LAYER])
    vB20_u = unit(vecB[STEER_LAYER])

    # 每層 persona 基底
    vA_layer_u = {L: unit(vecA[L]) for L in range(num_layers)} if USE_PER_LAYER_PERSONA_VECS else None
    vB_layer_u = {L: unit(vecB[L]) for L in range(num_layers)} if USE_PER_LAYER_PERSONA_VECS else None

    # CSV 欄位
    fieldnames = [
        "prompt_idx", "layer", "steer_layer",
        "alpha_A", "alpha_B", "eps_A", "eps_B",
        "delta_l2",
        "proj_A20", "proj_B20",
        "proj_A_L", "proj_B_L",
        "proj_A_transport", "proj_B_transport",
        "cos_tA_tB",
        "model", "vecA_path", "vecB_path"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, prompt in enumerate(NEUTRAL_LIST):
            print(f"\n[bold magenta]=== Prompt {i+1}/{len(NEUTRAL_LIST)} ===[/bold magenta]")
            print(prompt[:100] + ("..." if len(prompt) > 100 else ""))

            # Baseline
            base = run_forward_capture(model, tok, prompt, steer=False)

            # Calibration A（單軸小擾動）
            steer_cal_A = EPS_A * vA20_u
            calA = run_forward_capture(model, tok, prompt, steer=True, steer_layer=STEER_LAYER, steer_vec_unit=steer_cal_A)
            delta_cal_A = {L: calA[L] - base[L] for L in base.keys()}
            tA_u = {L: unit(delta_cal_A[L]) for L in delta_cal_A.keys()}

            # Calibration B（單軸小擾動）
            steer_cal_B = EPS_B * vB20_u
            calB = run_forward_capture(model, tok, prompt, steer=True, steer_layer=STEER_LAYER, steer_vec_unit=steer_cal_B)
            delta_cal_B = {L: calB[L] - base[L] for L in base.keys()}
            tB_u = {L: unit(delta_cal_B[L]) for L in delta_cal_B.keys()}

            # Main：A+B 合成 steer
            steer_main = ALPHA_A * vA20_u + ALPHA_B * vB20_u
            ste = run_forward_capture(model, tok, prompt, steer=True, steer_layer=STEER_LAYER, steer_vec_unit=steer_main)
            delta = {L: ste[L] - base[L] for L in base.keys()}

            # 只寫入 19–32 層
            L_from = max(0, LAYER_FROM)
            L_to   = min(num_layers - 1, LAYER_TO)

            # 列印摘要（可選）
            print("\nLayer |  ||Δ||   A•vA20   B•vB20   A•vA_L   B•vB_L   A•tp    B•tp   cos(tA,tB)")
            for L in range(L_from, L_to + 1):
                dL  = delta[L]
                amp = l2(dL)

                pA20 = dot(dL, vA20_u)
                pB20 = dot(dL, vB20_u)

                pAL = dot(dL, vA_layer_u[L]) if vA_layer_u is not None else None
                pBL = dot(dL, vB_layer_u[L]) if vB_layer_u is not None else None

                pAtp = dot(dL, tA_u[L])
                pBtp = dot(dL, tB_u[L])
                cAB  = cos_sim(tA_u[L], tB_u[L])

                pAL_s = f"{pAL:>8.4f}" if pAL is not None else "   (N/A)"
                pBL_s = f"{pBL:>8.4f}" if pBL is not None else "   (N/A)"
                mark  = " <== steer" if L == STEER_LAYER else ""
                print(f"{L:>5} | {amp:>6.3f}  {pA20:>7.3f}  {pB20:>7.3f}  {pAL_s}  {pBL_s}  {pAtp:>7.3f}  {pBtp:>7.3f}  {cAB:>8.4f}{mark}")

                # 寫 CSV
                writer.writerow({
                    "prompt_idx": i, "layer": L, "steer_layer": STEER_LAYER,
                    "alpha_A": ALPHA_A, "alpha_B": ALPHA_B, "eps_A": EPS_A, "eps_B": EPS_B,
                    "delta_l2": amp,
                    "proj_A20": pA20, "proj_B20": pB20,
                    "proj_A_L": "" if pAL is None else pAL,
                    "proj_B_L": "" if pBL is None else pBL,
                    "proj_A_transport": pAtp, "proj_B_transport": pBtp,
                    "cos_tA_tB": cAB,
                    "model": MODEL_NAME, "vecA_path": VEC_A_PATH, "vecB_path": VEC_B_PATH
                })

            # 檢查注入層
            print(f"\n[Check @L{STEER_LAYER}]  A•tp≈{dot(delta[STEER_LAYER], tA_u[STEER_LAYER]):.4f} (target {ALPHA_A}), "
                  f"B•tp≈{dot(delta[STEER_LAYER], tB_u[STEER_LAYER]):.4f} (target {ALPHA_B})")

    print(f"\n✅ Done. CSV saved to: [bold green]{out_csv}[/bold green]")

if __name__ == "__main__":
    main()

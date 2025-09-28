from rich import print
import torch, csv, time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== 配置 =====================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# VEC_PATH   = "Llama-3.1-8B-Instruct/multi_role/creative_professional_response_avg_diff.pt"
VEC_PATH   = "Llama-3.1-8B-Instruct/multi_role/environmentalist_response_avg_diff.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

STEER_LAYER = 20               # 在第 20 層插入
ALPHA_MAIN  = 2.0              # 主實驗擾動強度（想觀察主效應）
EPS_CAL     = 0.1              # 校準小擾動（用來估計傳輸方向）
LAYER_RANGE_PRINT_FROM = STEER_LAYER - 2  # 列印摘要從哪一層開始

# 是否有每層 persona 向量（如果只有單層向量可關閉）
USE_PER_LAYER_PERSONA_VECS = True

# 10 個題目
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
def run_forward_capture(model, tokenizer, prompt, steer=False, steer_layer=None, steer_vec_unit=None, steer_scale=0.0):
    """
    進行一次前向傳播，擷取每一層輸出的 (batch=0, last_token)。
    若 steer=True，則在 steer_layer 的 forward hook 對最後一 token 加上 steer_scale * steer_vec_unit。
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
                hs[:, -1, :] = hs[:, -1, :] + steer_scale * add

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

# ===================== 主流程 =====================
def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = f"steer_transport_layers_{ts}.csv"

    print(f"[bold cyan]Loading {MODEL_NAME} ...[/bold cyan]")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)

    # 讀 persona 向量（每層一支）
    vecs = torch.load(VEC_PATH, map_location="cpu")
    num_layers = len(model.model.layers)

    # 主要注入向量（第 STEER_LAYER 層）
    v_steer = vecs[STEER_LAYER].float()
    v_steer_unit = unit(v_steer)

    # (可選) 每層 persona 向量
    v_layer_unit = None
    if USE_PER_LAYER_PERSONA_VECS:
        v_layer_unit = {L: unit(vecs[L]) for L in range(num_layers)}

    # 打開 CSV 檔
    fieldnames = [
        "prompt_idx", "layer", "alpha", "eps", "steer_layer",
        "delta_l2", "proj_v20", "proj_vL", "proj_transport",
        "model", "vec_path"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 逐題跑
        for i, prompt in enumerate(NEUTRAL_LIST):
            print(f"\n[bold magenta]=== Prompt {i+1}/{len(NEUTRAL_LIST)} ===[/bold magenta]")
            print(prompt[:100] + ("..." if len(prompt) > 100 else ""))

            # Baseline
            base = run_forward_capture(model, tok, prompt, steer=False)

            # Calibration（小擾動，估 t_unit）
            cal = run_forward_capture(
                model, tok, prompt,
                steer=True, steer_layer=STEER_LAYER, steer_vec_unit=v_steer_unit, steer_scale=EPS_CAL
            )
            delta_cal = {L: cal[L] - base[L] for L in base.keys()}
            t_unit = {L: unit(delta_cal[L]) for L in delta_cal.keys()}

            # Main（主擾動）
            ste = run_forward_capture(
                model, tok, prompt,
                steer=True, steer_layer=STEER_LAYER, steer_vec_unit=v_steer_unit, steer_scale=ALPHA_MAIN
            )
            delta = {L: ste[L] - base[L] for L in base.keys()}

            # 層層寫入 CSV
            for L, idx in enumerate(range(num_layers)):
                dL = delta[L]
                amp = l2(dL)                           # ||Δ||
                p20 = dot(dL, v_steer_unit)            # proj onto v20 (注入層向量)
                pL  = None
                if v_layer_unit is not None:
                    pL = dot(dL, v_layer_unit[L])      # proj onto v_L（每層 persona 基底）
                pt  = dot(dL, t_unit[L])               # proj onto transported dir（這次輸入的搬運方向）

                if idx >= 19:
                    writer.writerow({
                        "prompt_idx": i,
                        "layer": L,
                        "alpha": ALPHA_MAIN,
                        "eps": EPS_CAL,
                        "steer_layer": STEER_LAYER,
                        "delta_l2": amp,
                        "proj_v20": p20,
                        "proj_vL": "" if pL is None else pL,
                        "proj_transport": pt,
                        "model": MODEL_NAME,
                        "vec_path": VEC_PATH
                    })

            # 小結印出（從 L=18 起）
            print("\nLayer |  ||Δ||    proj(v20)   proj(v_L)   proj(transport)")
            for L in range(max(0, LAYER_RANGE_PRINT_FROM), num_layers):
                dL = delta[L]; amp = l2(dL)
                p20 = dot(dL, v_steer_unit)
                pL  = None if v_layer_unit is None else dot(dL, v_layer_unit[L])
                pt  = dot(dL, t_unit[L])
                pL_str = f"{pL:>10.4f}" if pL is not None else "     (N/A)"
                mark = " <== steer" if L == STEER_LAYER else ""
                print(f"{L:>5} | {amp:>7.4f}  {p20:>10.4f}  {pL_str}  {pt:>15.4f}{mark}")

            # Sanity check at steer layer
            print(f"\n[Check] L{STEER_LAYER} proj_transport ≈ {dot(delta[STEER_LAYER], t_unit[STEER_LAYER]):.4f} ; ALPHA_MAIN = {ALPHA_MAIN}")

    print(f"\n✅ Done. CSV saved to: [bold green]{out_csv}[/bold green]")

if __name__ == "__main__":
    main()

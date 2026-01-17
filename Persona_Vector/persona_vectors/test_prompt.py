from rich import print
import torch, csv, time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== 配置 =====================
MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
VEC_A_PATH  = "Llama-3.1-8B-Instruct/multi_role/creative_professional_response_avg_diff.pt"
VEC_B_PATH  = "Llama-3.1-8B-Instruct/multi_role/environmentalist_response_avg_diff.pt"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# 只輸出這個層範圍
LAYER_FROM, LAYER_TO = 19, 32

# 是否使用每層 persona 向量（建議 True）
USE_PER_LAYER_PERSONA_VECS = True

# ---- 題目（不帶角色前綴）----
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

# ---- 角色前綴（可自訂文案）----
ROLE_A_NAME   = "creative professional"
ROLE_B_NAME   = "environmentalist"
# ROLE_A_PREFIX = f"You are a {ROLE_A_NAME}. Adopt this role and answer accordingly.\n"
# ROLE_B_PREFIX = f"You are an {ROLE_B_NAME}. Adopt this role and answer accordingly.\n"
# ROLE_A_AND_B = """
# You need to think and answer this question from three different professional perspectives:

# 1. Environmentalist:
# Specialty: Sustainability and Environmental Health
# Mission: Advocate for eco-friendly solutions, promote sustainable development and protect the planet. Guide us to consider the environmental impact of ideas, promoting innovations that contribute to planetary health.

# 2. Creative Professional:
# Specialty: Aesthetics, Narratives, and Emotions
# Mission: With artistic sensibility and mastery of narrative and emotion, infuse projects with beauty and depth. Challenge us to think expressively, ensuring solutions not only solve problems but also resonate on a human level.

# Please provide answers from these three role perspectives, with each role embodying their professional characteristics and thinking approaches.
# """
ROLE_A_AND_B = f"""
You are a helopful assistant. Please answer the question.
"""

# 三個變體：A / B / A+B
VARIANTS = {
    # "A":    lambda p: ROLE_A_PREFIX + p,
    # "B":    lambda p: ROLE_B_PREFIX + p,
    # "A+B":  lambda p: ROLE_A_PREFIX + ROLE_B_PREFIX + p
    "A+B": lambda p: ROLE_A_AND_B + p
}

# ===================== 工具 =====================
@torch.no_grad()
def unit(v: torch.Tensor, eps=1e-8):
    v = v.float()
    return v / (v.norm() + eps)

@torch.no_grad()
def forward_last_token_per_layer(model, tokenizer, prompt: str, device=DEVICE):
    """
    不做 steering，單次前向，擷取每層輸出的 (batch=0, last_token)。
    回傳 dict: layer_idx -> hidden(last_token) [float32, cpu]
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states  # len = L+1 (含 embeddings)
    captures = {}
    for L in range(1, len(hs)):  # 跳過 embeddings，對齊 0-based
        h_last = hs[L][:, -1, :].detach().float().cpu().squeeze(0)
        captures[L-1] = h_last
    return captures

@torch.no_grad()
def dot(a, b):  # 內積
    return float((a.float() @ b.float()).item())

# ===================== 主流程 =====================
def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_A   = f"delta_proj_roleA_L{LAYER_FROM}-{LAYER_TO}_{ts}.csv"
    out_B   = f"delta_proj_roleB_L{LAYER_FROM}-{LAYER_TO}_{ts}.csv"
    out_AB  = f"delta_proj_roleAB_L{LAYER_FROM}-{LAYER_TO}_{ts}.csv"

    print(f"[bold cyan]Loading {MODEL_NAME} ...[/bold cyan]")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)

    # 讀兩個 persona 向量
    vecA = torch.load(VEC_A_PATH, map_location="cpu")
    vecB = torch.load(VEC_B_PATH, map_location="cpu")
    num_layers = len(model.model.layers)

    # 參考方向（雖不注入，但作為固定軸）
    vA20_u = unit(vecA[20])
    vB20_u = unit(vecB[20])

    # 每層 persona 基底
    vA_layer_u = {L: unit(vecA[L]) for L in range(num_layers)} if USE_PER_LAYER_PERSONA_VECS else None
    vB_layer_u = {L: unit(vecB[L]) for L in range(num_layers)} if USE_PER_LAYER_PERSONA_VECS else None

    # 三個檔案各自一個 writer
    fieldnames = [
        "prompt_idx", "layer",
        "dproj_A20", "dproj_B20",
        "dproj_A_L", "dproj_B_L",
        "model", "vecA_path", "vecB_path",
        "roleA", "roleB"
    ]
    fA  = open(out_A,  "w", newline="", encoding="utf-8");  wA  = csv.DictWriter(fA,  fieldnames=fieldnames); wA.writeheader()
    fB  = open(out_B,  "w", newline="", encoding="utf-8");  wB  = csv.DictWriter(fB,  fieldnames=fieldnames); wB.writeheader()
    fAB = open(out_AB, "w", newline="", encoding="utf-8");  wAB = csv.DictWriter(fAB, fieldnames=fieldnames); wAB.writeheader()

    try:
        for i, base_prompt in enumerate(NEUTRAL_LIST):
            print(f"\n[bold magenta]=== Prompt {i+1}/{len(NEUTRAL_LIST)} ===[/bold magenta]")
            print(base_prompt[:100] + ("..." if len(base_prompt) > 100 else ""))

            # Baseline（none）
            caps_none = forward_last_token_per_layer(model, tok, base_prompt)

            # 依序處理 A / B / A+B
            for vname, wrap in VARIANTS.items():
                prompt = wrap(base_prompt)
                caps_var = forward_last_token_per_layer(model, tok, prompt)

                # 只寫 19–32 層的 Δ 投影
                L_from = max(0, LAYER_FROM)
                L_to   = min(num_layers - 1, LAYER_TO)
                for L in range(L_from, L_to + 1):
                    dL = caps_var[L] - caps_none[L]  # 這題在 L 層的「角色誘發淨變化」

                    dA20 = dot(dL, vA20_u)
                    dB20 = dot(dL, vB20_u)
                    dAL  = dot(dL, vA_layer_u[L]) if vA_layer_u is not None else None
                    dBL  = dot(dL, vB_layer_u[L]) if vB_layer_u is not None else None

                    row = {
                        "prompt_idx": i,
                        "layer": L,
                        "dproj_A20": dA20,
                        "dproj_B20": dB20,
                        "dproj_A_L": "" if dAL is None else dAL,
                        "dproj_B_L": "" if dBL is None else dBL,
                        "model": MODEL_NAME,
                        "vecA_path": VEC_A_PATH,
                        "vecB_path": VEC_B_PATH,
                        "roleA": ROLE_A_NAME,
                        "roleB": ROLE_B_NAME
                    }

                    if vname == "A":
                        wA.writerow(row)
                    elif vname == "B":
                        wB.writerow(row)
                    # if vname == 'A+B':  # "A+B"
                    else:
                        wAB.writerow(row)

        print(f"\n✅ Done.")
        print(f"  • A  → {out_A}")
        print(f"  • B  → {out_B}")
        print(f"  • A+B→ {out_AB}")
    finally:
        fA.close(); fB.close(); fAB.close()

if __name__ == "__main__":
    main()

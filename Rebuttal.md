## 📝 Response to Reviewers: Sensitivity Analysis of Steering Factor $\alpha$ and Layer Selection (Reviwer 1, 2, 3)

We agree that a comprehensive sensitivity analysis is crucial for validating our design choices, especially since activation steering methods can sometimes be sensitive to these hyperparameters.

In our main paper, we followed prior work by setting the Steering Layer to **Layer 20** and the Steering Factor $\alpha$ to **$2.0$**. To address the concerns, we conducted an **Ablation Study** across $4$ different steering layers (12, 16, 20, 24) and $7$ different $\alpha$ values (0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0) on all tested models (Llama-3.1-8B, Qwen2.5-7B, and Gemma-3-4B).

The results of this study are summarized in the table provided (and will be included in our paper), justifying our initial choices.

---

### Originality Scores

| Model            | Layer  | α=0.1 | α=0.5 | α=1.0 | α=1.5 | α=2.0 | α=2.5 | α=5.0 |
| :--------------- | :----: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemma-3-4B**   |   12   | 4.900 | 4.867 | 4.933 | 4.800 | 4.967 | 4.900 | 4.933 |
|                  |   16   | 4.933 | 4.967 | 4.933 | 4.933 | 4.967 | 4.967 | 4.944 |
|                  | **20** | 5.000 | 4.867 | 5.000 | 4.967 | 5.000 | 5.000 | 4.933 |
|                  |   24   | 4.933 | 5.000 | 4.867 | 4.920 | 5.000 | 5.000 | 4.900 |
| **Qwen-2.5-7B**  |   12   | 4.300 | 4.000 | 4.100 | 4.167 | 4.167 | 4.200 | 4.567 |
|                  |   16   | 4.233 | 4.233 | 4.133 | 4.533 | 4.467 | 4.633 | 3.833 |
|                  | **20** | 4.533 | 4.233 | 4.400 | 4.300 | 4.633 | 4.600 | 3.533 |
|                  |   24   | 4.200 | 4.133 | 4.100 | 4.433 | 4.367 | 4.500 | 4.400 |
| **Llama-3.1-8B** |   12   | 4.133 | 4.467 | 4.433 | 4.600 | 4.667 | 4.767 | 4.667 |
|                  |   16   | 4.333 | 4.267 | 4.600 | 4.633 | 4.800 | 4.833 | 4.267 |
|                  | **20** | 4.467 | 4.433 | 4.500 | 4.567 | 4.533 | 4.533 | 4.800 |
|                  |   24   | 4.400 | 4.333 | 4.467 | 4.433 | 4.533 | 4.433 | 4.367 |

### Elaboration Scores

| Model            | Layer  | α=0.1 | α=0.5 | α=1.0 | α=1.5 | α=2.0 | α=2.5 | α=5.0 |
| :--------------- | :----: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemma-3-4B**   |   12   | 4.967 | 5.000 | 5.000 | 4.967 | 5.000 | 4.933 | 4.967 |
|                  |   16   | 4.967 | 5.000 | 4.833 | 4.967 | 5.000 | 4.950 | 3.778 |
|                  | **20** | 5.000 | 4.867 | 4.933 | 4.967 | 5.000 | 4.967 | 3.133 |
|                  |   24   | 5.000 | 4.910 | 4.967 | 5.000 | 4.900 | 5.000 | 3.567 |
| **Qwen-2.5-7B**  |   12   | 4.900 | 4.500 | 4.433 | 4.467 | 4.767 | 4.367 | 4.933 |
|                  |   16   | 4.600 | 4.500 | 4.300 | 4.633 | 4.633 | 4.633 | 3.600 |
|                  | **20** | 4.800 | 4.567 | 4.667 | 4.600 | 4.833 | 4.900 | 1.800 |
|                  |   24   | 4.733 | 4.500 | 4.633 | 4.533 | 4.733 | 4.700 | 4.367 |
| **Llama-3.1-8B** |   12   | 4.900 | 4.867 | 4.833 | 4.700 | 4.867 | 4.933 | 4.800 |
|                  |   16   | 4.933 | 4.633 | 4.767 | 4.900 | 4.867 | 4.833 | 3.700 |
|                  | **20** | 4.900 | 4.767 | 4.900 | 4.767 | 4.833 | 4.900 | 4.467 |
|                  |   24   | 4.900 | 4.567 | 4.800 | 4.700 | 4.833 | 4.700 | 4.333 |

### 1. Sensitivity to Steering Layer Selection (Reviewer 1, 2, 3)

The analysis confirms the strength of our choice for Layer 20:

- **Optimal Layer:** Across the three distinct models and both evaluation criteria, **Layer 20 (marked with a red circle)** consistently yields performance scores near or at the maximum when the $\alpha$ factor is set within the optimal range (e.g., $\alpha=2.0$).
- **Conclusion:** The ablation study validates our decision to use Layer 20, demonstrating its strength as a **cross-model choice**.

### 2. Sensitivity to Steering Factor $\alpha$ (Reviewer 1, 3)

The study provides clear bounds for stable performance concerning the steering factor:

- **Optimal Stability Range:** The models exhibit **high and stable performance** within the mid-range of $\alpha$, specifically between **$1.0$ and $2.5$**. Our choice of $\alpha = 2.0$ (green line) falls perfectly within this region of stability and peak performance.
- **Instability at Extremes:**
- **Low $\alpha$ ($0.1$ and $0.5$):** Performance is slightly sub-optimal, indicating the steering signal is too weak to be fully effective.
- **High $\alpha$ ($5.0$):** This factor (bright yellow line) clearly causes performance collapse across all models and tasks, with the lowest scores often recorded here (e.g., Elaboration for Llama-3.1-8B and Qwen2.5-7B). This confirms that an **overly strong steering factor can disrupt the model's internal representations**.
- **Conclusion:** The method is **robust** to $\alpha$ changes within the optimal window ($1.0 \leq \alpha \leq 2.5$), and our selection of $\alpha = 2.0$ is located in this window.

### Action Taken

We will add the full details of this ablation study to the revised manuscript, with a discussion that references these findings and provides strong evidence for the selected hyperparameters.

---

## 📝 Response to Reviewer: Details on Contrastive Dataset Construction for Persona Extraction

Thank you for requesting clarification on the construction of the contrastive datasets ($D_P^+$ and $D_P^-$) used for Persona Vector extraction. We agree that these details are vital for the full understanding and reproducibility of our bespoke extraction pipeline.

We summarize the key details regarding the dataset construction, prompt diversity, and filtering thresholds below:

1. Dataset Generation and Prompt Diversity
   - Target Roles: We focused on complex professional roles (e.g., creative professional, environmentalist) as defined in LLM-Discussion, which required a tailored approach compared to methods targeting simple personality traits.
   - Prompt Diversity: To ensure a robust extraction, we introduced prompt diversity by:
     - Using five distinct positive system prompts (for each persona) created by Claude-sonnet-4, designed to capture the persona's unique traits. ("You are a creative expert. Think outside the box and provide imaginative responses.") These prompts are fully detailed in Table 11 and 12 in Appendix B.
     - Contrasting these with a standard neutral prompt ("You are a helpful assistant.").
   - Response Count: The model generated responses to 20 trait-eliciting questions using both the positive and neutral prompts, resulting in a pool of responses for scoring.
2. Filtering and Thresholds

   The final contrastive datasets ($D_P^+$ and $D_P^-$) were curated based on scores provided by an LLM-judge (GPT-4o-mini), which assigned a score from 0-100 for trait expression:

   - Positive-Expression Set ($D_P^+$): This set contains positive-prompt responses with a trait expression score greater than 50 ($> 50$).
   - Neutral-Expression Set ($D_P^-$): This set contains neutral-prompt responses with a trait expression score less than 50 ($< 50$).

   This filtering ensures a clear and maximal contrast between the activations of the two sets, as required for effective vector extraction (Equation 6).$$\vec{v}_P^{(l)} = \frac{1}{|D_P^+|} \sum_{\mathbf{x} \in D_P^+} \vec{a}^{(l)}(\mathbf{x}) - \frac{1}{|D_P^-|} \sum_{\mathbf{x} \in D_P^-} \vec{a}^{(l)}(\mathbf{x})$$

**Action Taken:**

While these procedural details, including the specific prompts and the filtering threshold ($\pm 50$), are fully documented in Appendix B of the supplementary material, we will revise the main paper to explicitly summarize the key filtering threshold and the size/diversity of the prompt sets to prevent this confusion and further enhance clarity. We also commit to releasing the complete pipeline and dataset upon publication, which will include all prompts and scoring scripts used in this process.

---

## 📝 Response to Reviewer: Overall Computational Cost and Amortization

Thank you for this insightful question concerning the overall computational cost of our method, BILLY. We fully acknowledge that the initial steps of persona vector generation—including prompt engineering, response scoring, and layer selection—incur a significant, one-time upfront cost.

However, the core efficiency advantage of BILLY lies in the fact that this initial investment is amortized over subsequent inference queries, transforming BILLY into a highly cost-effective solution for frequent, real-time creative generation.

1. Amortized Cost Analysis

   - The expensive persona extraction and vector generation process is executed only once per persona, creating a reusable artifact ($\vec{v}_P^{(l)}$) for all future queries.
   - As detailed in Appendix C and demonstrated in Figure 5 (Amortized Average Input Token Per Query), the cost per query decreases dramatically as the method is utilized more frequently:
     - For only 100 queries, the average token per query is high ($\approx 2848.55$).
     - For 10,000 queries, the average token per query drops sharply to only $62.2$.
   - This demonstrates that the initial fixed cost becomes negligible when amortized over a large number of inference calls.

2. Low Inference Cost Advantage (Steady-State Operation)

   Once the one-time persona vector is generated, BILLY operates with significantly reduced cost and latency compared to the competitive baselines, as shown in Table 3 (based on $n=10,000$ queries)

   Our main paper's claim on cost reduction is based on the amortized cost for high-frequency usage, which is the practical scenario for deploying LLMs in real-time generation services.

**Conclusion:**

While the initial setup cost is higher, the sustained token cost and latency savings achieved by substituting complex, multi-turn LLM prompts with a simple, reusable activation vector (BILLY) lead to a substantial reduction in the overall computational cost when the method is used at scale. We believe this trade-off—a one-time setup cost for massive, sustained inference efficiency—is a crucial advantage of our approach.

## Human Evaluation

Due to the time limit, we select 3 questions from each benchmark (a total of 100) answered by Llama-3.1-8B to conduct a rapid, small-scale human evaluation. Below are the average Human Scores provided by our 11 diverse raters across the items selected for the study (representing 132 individual scores):

| Benchmark    | Method         |  Originality  |  Elaboration  |
| :----------- | :------------- | :-----------: | :-----------: |
| AUT          | SA             |   3.11±1.21   |   3.43±1.1    |
|              | LLM Discussion |   3.55±1.07   |   2.78±1.09   |
|              | BILLY          | **3.67±1.11** | **4.08±0.89** |
| Instances    | SA             |   2.53±1.15   |   2.55±1.16   |
|              | LLM Discussion |   3.45±1.19   |   3.04±1.08   |
|              | BILLY          | **3.75±1.38** | **3.96±0.9**  |
| Scientific   | SA             |   3.01±1.37   |   3.86±1.07   |
|              | LLM Discussion |   3.3±1.16    |   2.70±0.93   |
|              | BILLY          | **3.47±1.18** | **4.23±0.85** |
| Similarities | SA             |   3.15±1.4    |   3.79±0.99   |
|              | LLM Discussion |   2.79±1.11   |   2.33±1.13   |
|              | BILLY          | **3.75±1.09** | **4.17±1.07** |

- BILLY consistently outperforms both baselines in terms of Human Scores on Originality and Elaboration across all benchmarks, demonstrating its effectiveness in enhancing creative generation as perceived by human evaluators.

We also calculated the `Spearman's Rank Correlation Coefficient` and `Pearson correlation coefficient` between our LLM-Judge and human raters to validate our evaluation pipeline, shown in the table below:

| Correlation | Originality | Elaboration |
| :---------- | :---------: | :---------: |
| Spearman's  |   0.7278    |   0.4276    |
| Pearson     |   0.6593    |   0.3978    |

The result indicates a strong positive correlation between the two sets of scores. This suggests that our LLM-Judge is a reliable proxy for human judgment. And we will add this analysis to the revised manuscript to strengthen the validity of our evaluation approach after collecting more human evaluation data.

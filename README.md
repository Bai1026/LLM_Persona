# BILLY: Steering Large Language Models via Merging Persona Vectors for Creative Generation

[![GitHub stars](https://img.shields.io/github/stars/bai1026/LLM_Persona?style=social)](https://github.com/bai1026/LLM_Persona)

<div align="center">

## 🌐 Project Website

### [**Visit Our Intro Website**](https://bai1026.github.io/LLM_Persona/)

_Explore BILLY's capabilities and learn about our research_

[![Website](https://img.shields.io/badge/Website-Intro_Page-blue?style=for-the-badge&logo=github-pages)](https://bai1026.github.io/LLM_Persona/)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2510.10157)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Experimental Setups](#experimental-setups)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Citation](#citation)

---

## 🔍 Overview

This repository contains the official implementation of **BILLY** (Behavior Integration via Learning-based merging in LLM identitY), a novel approach for steering large language models through persona vector merging for enhanced creative generation.

---

## 📁 Repository Structure

```
LLM_Persona/
├── Persona_Vector/          # Main persona vector experiments
├── LLM_Discussion/           # Multi-agent discussion experiments  
├── Human_Evaluation          # Human evaluation data and analysis
├── docs/                     # Project website files
└── others/                   # Utility scripts (API clients, token counters)
```

### 🎯 Main Experimental Folders

#### 1. **`Persona_Vector/`** - Persona Vector Experiments

This folder contains all model-related experiments including:

- **Single Agent**: Standard single-agent generation with various LLMs
- **Single Agent with Multiple Role Prompting**: Multi-role prompting techniques
- **BILLY (Persona Vector Merging)**: Our proposed persona vector merging approach

**Supported Models:**
- Gemma 2 (2B, 9B, 27B)
- LLaMA series
- Other instruction-tuned models

📖 **See detailed documentation:** [`Persona_Vector/experiment/README.md`](Persona_Vector/experiment/README.md)

---

#### 2. **`LLM_Discussion/`** - Multi-Agent Discussion Framework

Independent multi-agent discussion system for creative generation tasks.

**Features:**
- Multi-agent debate and collaboration
- Requires API server setup for model inference
- Standalone experiment pipeline

📖 **See detailed documentation:** [`LLM_Discussion/README.md`](LLM_Discussion/README.md)

---

#### 3. **`Human_Evaluation`** - Human Evaluation & Analysis

Contains human evaluation data and statistical analysis scripts:

- **Human evaluation scores** from creativity assessments
- **Correlation analysis** between human and LLM judgments
- **Inter-rater reliability** calculations (Krippendorff's Alpha)
- **Kendall's Tau** analysis for rater agreement

**Key Scripts:**
- `human_irr.py` - Inter-rater reliability analysis
- `human_kendall.py` - Kendall's Tau correlation
- `llm_human_correlation.py` - LLM-human score correlation
- `merge_xlsx.py` - Data preprocessing and merging

**Data Files:** Available in `Human_Evaluationdata/`

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.0+

### Install Dependencies

```bash
# Main requirements
pip install -r requirements.txt

# For persona vector experiments
cd Persona_Vector
pip install -r requirements.txt

# For LLM Discussion experiments
cd LLM-Discussion
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### 1. Run Persona Vector Experiments

```bash
cd Persona_Vector

# Generate persona vectors
python generate_multi_role_vectors.py

# Run BILLY (persona merging)
python activation_steer.py --config configs/billy_config.yaml

# Evaluation
python eval/eval_persona.py
```

### 2. Run LLM Discussion Experiments

```bash
cd LLM-Discussion

# Start API server (in separate terminal)
bash ../others/start_api.sh

# Run discussion experiments
python Experiments/run_discussion.py
```

### 3. Analyze Human Evaluation Data

```bash
cd BILLY/Rebuttal

# Calculate inter-rater reliability
python human_irr.py

# Calculate Kendall's Tau correlation
python human_kendall.py

# Analyze LLM-human correlation
python llm_human_correlation.py
```

---

## 📊 Experimental Details

For detailed experimental configurations, hyperparameters, and reproducibility instructions, please refer to the README files in each respective folder:

- **Persona Vector Experiments:** [`Persona_Vector/experiment/README.md`](Persona_Vector/experiment/README.md)
- **LLM Discussion Setup:** [`LLM_Discussion/README.md`](LLM_Discussion/README.md)
- **Human Evaluation:** [`Human_Evaluation`](Human_Evaluation) *(if available)*

---

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{pai2025billysteeringlargelanguage,
  title={BILLY: Steering Large Language Models via Merging Persona Vectors for Creative Generation}, 
  author={Tsung-Min Pai and Jui-I Wang and Li-Chun Lu and Shao-Hua Sun and Hung-Yi Lee and Kai-Wei Chang},
  booktitle={Conference of the European Chapter of the Association for Computational Linguistics},
  year={2026}
}
```
---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.

# Pros and Cons Extraction from Reviews (ABSA with Instruction-Tuned LLMs)

**Senior Project (CS 3200 Senior Project 1, 2/2024)**  
**Authors:** Lu Phone Maw, Wai Yan Paing  
**Advisor:** Asst.Prof.Dr. Thanachai Thumthawatworn  

---

## 1) Project Summary

This project improves **Aspect-Based Sentiment Analysis (ABSA)** by *instruction tuning* modern large language models (LLMs) using an **InstructABSA-style prompt**.  
We fine-tune **Llama 3 (8B)** and **Mistral 7B v0.3** for two core ABSA subtasks:

- **Aspect Term Extraction (ATE):** Extract aspect terms from a review sentence (e.g., `"battery life"`, `"service"`).
- **Aspect Sentiment Classification (ASC):** Given a review and a specific aspect, classify sentiment as **positive / negative / neutral**.

We evaluate on **SemEval 2014** datasets for **Restaurant (Res14)** and **Laptop (Lap14)** domains.

---

## 2) Why This Matters (Recruiter-Friendly)

Customer reviews are packed with actionable information, but plain star ratings are too coarse. ABSA enables:

- **Product insights:** What customers like/dislike (battery, keyboard, screen).
- **Service insights:** What drives satisfaction (food, staff, ambience).
- **Better decision-making:** Fine-grained dashboards, QA monitoring, prioritization.

This project shows how **prompt engineering + parameter-efficient fine-tuning** can dramatically improve performance on ABSA without training huge models from scratch.

---

## 3) Key Contributions

### ✅ Contribution 1 — Instruction-tuning Llama 3 & Mistral with an InstructABSA-style prompt
Instead of plain supervised fine-tuning, we train LLMs to follow a **task definition + labeled examples** prompt style (inspired by InstructABSA2). This explicitly guides behavior and improves extraction/classification quality.

### ✅ Contribution 2 — Efficient training via QLoRA + 4-bit quantization (PEFT)
We fine-tune using **LoRA adapters** on top of **4-bit quantized** model weights (QLoRA), enabling training on limited compute while retaining strong performance.

### ✅ Contribution 3 — Strong benchmark results against known baselines
We benchmark against:
- InstructABSA2
- Instruct-DeBERTa pipeline
- Plain / baseline fine-tuning (e.g., Llama 2 7B, Mistral 7B baselines)

---

## 4) Results (Headline Metrics)

### Aspect Term Extraction (ATE) — F1 (Test)

| Model | Res14 | Lap14 |
|---|---:|---:|
| **Mistral 7B (proposed)** | **95.40%** | **92.89%** |
| **Llama 3 8B (proposed)** | **94.03%** | **91.89%** |

> Both proposed models outperform older Llama baselines (~72% F1) and are competitive with InstructABSA2 / Instruct-DeBERTa.

### Aspect Sentiment Classification (ASC) — Accuracy (Test)

| Model | Res14 | Lap14 |
|---|---:|---:|
| **Mistral 7B (proposed)** | **89.29%** | **82.76%** |
| **Llama 3 8B (proposed)** | **88.57%** | **81.50%** |

Notes:
- Restaurant reviews tend to have clearer sentiment cues (food/service/ambience), boosting accuracy.
- Laptop reviews have more technical jargon and subtlety, making sentiment harder to classify.

---

## 5) Problem Statement

Although recent ABSA research uses LLMs, many approaches still underutilize the **instruction-following abilities** of modern models. Conventional fine-tuning often produces suboptimal F1 scores and fails to extract the full potential of LLMs on nuanced tasks.

This project targets that gap by using **task-specific instructions and examples** during training (instruction tuning).

---

## 6) Scope

We perform **8 training configurations**:

- 2 models (**Llama 3 8B**, **Mistral 7B**)
- 2 domains (**Res14**, **Lap14**)
- 2 tasks (**ATE**, **ASC**)

We intentionally remove **"conflict" polarity** (common practice in ABSA research due to severe class imbalance).

---

## 7) Dataset

**SemEval 2014 Task 4** (Restaurant & Laptop domains)

- Each record includes:
  - `review text`
  - `aspect terms` (for ATE)
  - `sentiment polarity` per aspect (for ASC)

Approx split per domain (as used in the report):
- **~3000 train**, **100 validation**, **~800 test**

Preprocessing highlights:
- Drop unused columns (e.g., `aspectCategories`).
- Parse aspect term lists (e.g., via `ast.literal_eval` if stored as strings).
- For ATE: join multiple aspects as comma-separated list; if none, output `noaspectterm`.
- For ASC: filter out `"conflict"` polarity.

---

## 8) Methodology (End-to-End)

### 8.1 Prompting Strategy (InstructABSA2-style)
We embed a clear **definition** plus **few-shot examples** into an Alpaca-style template.

- **ATE prompt**: “Extract aspect terms… output `noaspectterm` if none”
- **ASC prompt**: “Given the aspect … classify polarity as positive/negative/neutral”

### 8.2 Model Training Setup
We use parameter-efficient fine-tuning (PEFT) and memory optimizations:

- **4-bit quantized base model**
- **LoRA adapters** on attention projection modules (query/key/value/output)
- **Gradient checkpointing** to support longer contexts
- **Instruction tuning**: model learns to generate outputs that match instruction format

### 8.3 Training Procedure (SFT)
Trainer: `SFTTrainer` (TRL)

Key hyperparameters (from the report):
- Total training steps: **400**
- Learning rate: **3e-4**
- Warmup steps: **40**
- **ATE:** per-device batch size 4, grad accumulation 4 (effective 16)
- **ASC:** per-device batch size 8, grad accumulation 4 (effective 32)
- LoRA rank **r=16**, LoRA alpha **16**, dropout **0**, bias updates **none**

### 8.4 Inference & Post-processing
- Run the same prompt template on test instances.
- Post-process model generations into:
  - extracted aspect strings (ATE)
  - sentiment label (ASC)

---

## 9) Evaluation

### 9.1 ATE Metrics
- Precision / Recall / F1 (micro-averaged)
- Matching uses a **substring-based** criterion (e.g., `"battery"` matches `"battery life"`)

### 9.2 ASC Metrics
- Overall **accuracy** (main metric)
- Macro precision/recall/F1 also analyzed by class (positive/negative/neutral)
- Confusion matrix analysis included to study common failure modes

---

## 10) Tech Stack

**Training & runtime**
- Google Colab (GPU: **A100** in experiments)
- Python

**Libraries**
- `unsloth` (FastLanguageModel, quantized loading, training acceleration)
- Hugging Face `transformers`, `datasets`
- `trl` (SFTTrainer)
- `bitsandbytes` (4-bit quantization)
- `peft` (LoRA / QLoRA)
- `accelerate`, `xformers` (performance utilities)

---

## 11) What I’d Show in an Interview

**High-level design**
- Why instruction tuning improves ABSA on generative LLMs
- Why QLoRA makes fine-tuning feasible on limited compute
- How prompt examples shape consistent output formats

**Engineering details**
- Dataset transformation into instruction-following training samples
- Evaluation details (substring matching in ATE, macro vs accuracy for ASC)
- Tradeoffs between Mistral vs Llama across domains

**Model behavior**
- Restaurant domain benefits from repetitive vocabulary & clearer sentiment cues
- Laptop domain is harder due to technical phrasing and subtle negatives/neutrals

---

## 12) Repository Structure (Suggested)

> If you package this as a public repo, a clean structure helps recruiters run it quickly.

```
.
├── README.md
├── data/
│   ├── raw/                 # original SemEval files
│   └── processed/           # transformed train/val/test CSV/JSONL
├── prompts/
│   ├── ate_restaurant.txt
│   ├── ate_laptop.txt
│   ├── asc_restaurant.txt
│   └── asc_laptop.txt
├── src/
│   ├── preprocess.py        # build instruction-tuning dataset
│   ├── train_ate.py
│   ├── train_asc.py
│   ├── infer.py
│   └── evaluate.py
├── notebooks/
│   └── experiments.ipynb
└── results/
    ├── ate_metrics.csv
    ├── asc_metrics.csv
    └── figures/
```

---

## 13) How to Run (Template)

> Replace these with the actual commands/files from your codebase.

### Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Preprocess
```bash
python src/preprocess.py --domain restaurant --task ate
python src/preprocess.py --domain laptop --task asc
```

### Train
```bash
python src/train_ate.py --model mistral-7b --domain restaurant
python src/train_asc.py --model llama3-8b --domain laptop
```

### Evaluate
```bash
python src/evaluate.py --task ate --domain restaurant --preds results/preds.jsonl
```

---

## 14) Limitations

- Domain coverage is limited to SemEval Restaurant/Laptop due to dataset availability.
- The `"conflict"` class is removed to align with common ABSA evaluation practice.
- Laptop sentiment is harder due to technical and subtle phrasing, especially neutral vs negative.

---

## 15) Future Work

- Handle `"conflict"` polarity using better balancing or multi-label techniques
- Improve neutral classification via data augmentation
- Extend to other domains (finance/healthcare) and multilingual ABSA
- Explore richer prompt strategies (e.g., multi-turn prompts, chain-of-thought where appropriate)
- Experiment with other PEFT variants (e.g., LoftQ, rank-stabilized LoRA)

---

## 16) References

- Scaria et al. — **INSTRUCTABSA** (instruction learning for ABSA)
- Jayakody et al. — **Instruct-DeBERTa** (hybrid pipeline approach)
- SemEval-2014 Task 4 — Aspect Based Sentiment Analysis
- Unsloth documentation
- PEFT survey (Xu et al.)

---

## Contact / Links

- Add your GitHub repo link here
- Add paper/report PDF link here
- Add a short demo video link here (optional)

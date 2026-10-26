# LLMHGSyn

**Hypergraph-Guided Large Language Model Distillation for Drug Synergy Prediction**

LLMHGSyn predicts whether a pair of drugs is synergistic in a given cancer cell line. It constructs a drug–drug–cell-line hypergraph from positive training examples, uses hypergraph-aware instructions to adapt a teacher language model, and distills multi-scale teacher representations into a student hypergraph neural network. At inference time, the student uses molecular graphs and cell-line gene expression; the teacher is not required.

The pipeline supports two processed datasets, **GDSC** and **DrugCombDB**, and four evaluation settings: `warm_start`, `drug` (unseen drugs), `drug_pairs` (unseen drug pairs), and `cell_line` (unseen cell lines).

## Run order

Run the following scripts **in this order for each dataset, split, and fold**:

| Step | Script | Output |
| --- | --- | --- |
| 1 | `prepare_instructions.py` | Fold-specific hypergraph instructions and node catalog |
| 2 | `finetune/llm_main.py` | Fine-tuned teacher LoRA adapter |
| 3 | `finetune/extract_teacher.py` | Multi-scale teacher embeddings (`.npz`) |
| 4 | `main.py` | Student checkpoint, predictions, and metrics |

The teacher embeddings must be generated separately for every fold because each fold has a different training hypergraph. Do not use embeddings or adapters from another fold.

## Requirements

- Python 3.10 or later
- An NVIDIA GPU with enough memory for 4-bit Qwen2.5-7B fine-tuning and extraction
- The packages in `requirements.txt`, plus `bitsandbytes` for the current 4-bit teacher scripts
- Access to `Qwen/Qwen2.5-7B-Instruct` or a compatible local model directory

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install bitsandbytes
```

Install a PyTorch build compatible with your CUDA environment if the default pip installation does not provide one.

## Data layout

The commands below expect:

```text
Data/
├── GDSC/
│   ├── drug_smiles.csv
│   ├── cell_line_gene_expression.csv
│   └── {warm_start,drug,drug_pairs,cell_line}/
│       ├── train_1.csv ... train_5.csv
│       └── test_1.csv  ... test_5.csv
├── DrugCombDB/
│   └── same structure as GDSC
└── LLM_features/
    ├── ensem_describe_deepsynergydrug_default_all.pickle
    └── ensem_describe_deepsynergycellline_default_all.pickle
```

Each split CSV contains `drug_a`, `drug_b`, `cell_line`, and binary `synergy` columns. The drug and cell-line names must match their feature tables and description files. Only positive **training** samples are used to construct each fold's hypergraph.

## Example: run one fold

Run these commands from the **repository root**. Change `DATASET`, `SPLIT`, `FOLD`, and `GPU` as needed. Valid datasets are `GDSC` and `DrugCombDB`; valid splits are `warm_start`, `drug`, `drug_pairs`, and `cell_line`.

```bash
DATASET=GDSC
SPLIT=warm_start
FOLD=1
GPU=0
TEACHER_MODEL=Qwen/Qwen2.5-7B-Instruct
RUN_DIR="outputs/${DATASET}/${SPLIT}/fold_${FOLD}"
```

### 1. Build fold-specific instructions

```bash
python prepare_instructions.py \
  --data-root Data \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --fold "$FOLD" \
  --hops 2 \
  --samples-per-prompt 8 \
  --positive-per-prompt 4 \
  --bags-per-node 4 \
  --output-dir "$RUN_DIR/instructions"
```

This writes `all.jsonl`, `train.jsonl`, `node_catalog.json`, and `dataset_summary.json`. The instruction builder masks each queried positive triple from its structural context.

### 2. Fine-tune the teacher

```bash
python finetune/llm_main.py \
  --model "$TEACHER_MODEL" \
  --instruction-dir "$RUN_DIR/instructions" \
  --output-dir "$RUN_DIR/teacher_lora" \
  --max-length 4096 \
  --device "$GPU"
```

The current teacher script uses 4-bit QLoRA and computes the language-model loss on response tokens only.

### 3. Extract multi-scale teacher representations

```bash
python finetune/extract_teacher.py \
  --model "$TEACHER_MODEL" \
  --adapter "$RUN_DIR/teacher_lora" \
  --instructions "$RUN_DIR/instructions/all.jsonl" \
  --output "$RUN_DIR/teacher_features.npz" \
  --max-length 4096 \
  --device "$GPU"
```

Extraction uses the input prompt without the target response, pools the final four hidden layers at the last non-padding prompt token, and averages representations across query bags.

### 4. Train and evaluate the student

```bash
python main.py \
  --data-root Data \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --fold "$FOLD" \
  --teacher-features "$RUN_DIR/teacher_features.npz" \
  --molecule-cache "outputs/cache/${DATASET}_molecular_graphs.npz" \
  --output-dir "$RUN_DIR/student" \
  --device "cuda:${GPU}"
```

The student encodes drug SMILES as molecular graphs, cell lines from gene expression, and training positives as a hypergraph. Its outputs include `model.pt`, `best_model.pt`, `metrics.json`, `best_metrics.csv`, `history.json`, and prediction arrays.

**Evaluation note:** The current `main.py` default selects checkpoints using the test set. This produces selection-biased test scores. Use the provided outputs as exploratory results unless model selection is moved to a separate validation set and the final test fold is evaluated only after selection.

## Run additional folds or datasets

Repeat Steps 1–4 with `FOLD=1` through `FOLD=5`. Set `DATASET=DrugCombDB` to run the same pipeline on DrugCombDB. Use a separate `RUN_DIR` for every dataset/split/fold combination.

`grid_search.py` can automate exploratory five-fold student-model searches **after** all five teacher feature files have been generated. It expects the files under one `--teacher-root` directory as `fold_1.npz` through `fold_5.npz`; copy or link the outputs from Step 3 into that layout before running it. Its current ranking also uses test metrics, so it is not a validation-based model-selection protocol.

## Student parameters for the eight experiment settings

Use the row matching the dataset and split when running Step 4 for folds 1–5. These values are the rank-1 configurations recorded in `outputs/grid_search/<split>/five_fold_results.csv` (GDSC) and `outputs_DrugCombDB/grid_search/<split>/five_fold_results.csv` (DrugCombDB). The reported scores are five-fold mean ± standard deviation.

| Dataset | Split | Hidden dim | Learning rate | Weight decay | Dropout | Distill weight | Temperature | ROC-AUC | AUPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GDSC | `warm_start` | 512 | 1e-4 | 1e-5 | 0.1 | 0.1 | 0.05 | 0.9117 ± 0.0052 | 0.4522 ± 0.0213 |
| GDSC | `drug` | 512 | 1e-4 | 1e-5 | 0.1 | 0.01 | 0.2 | 0.7608 ± 0.0393 | 0.1557 ± 0.0199 |
| GDSC | `drug_pairs` | 512 | 1e-4 | 0 | 0.1 | 0.1 | 0.05 | 0.8852 ± 0.0108 | 0.3653 ± 0.0263 |
| GDSC | `cell_line` | 512 | 1e-4 | 1e-5 | 0.1 | 0.01 | 0.05 | 0.8314 ± 0.0141 | 0.2555 ± 0.0135 |
| DrugCombDB | `warm_start` | 512 | 1e-4 | 1e-5 | 0.2 | 0.01 | 0.1 | 0.8996 ± 0.0040 | 0.4841 ± 0.0152 |
| DrugCombDB | `drug` | 128 | 1e-4 | 0 | 0.4 | 0.1 | 0.05 | 0.7606 ± 0.0381 | 0.1673 ± 0.0498 |
| DrugCombDB | `drug_pairs` | 512 | 1e-4 | 1e-5 | 0.1 | 0.1 | 0.2 | 0.8593 ± 0.0213 | 0.3826 ± 0.0468 |
| DrugCombDB | `cell_line` | 512 | 1e-4 | 1e-5 | 0.1 | 0.01 | 0.1 | 0.8498 ± 0.0091 | 0.3406 ± 0.0288 |

For every row, set `--hgnn-layers 2`, `--epochs 3000`, `--cosine-weight 0.1`, `--distill-warmup-ratio 0.1`, `--distill-ramp-epochs 50`, `--grad-clip 5.0`, `--early-stop-metric roc_auc`, `--early-stop-patience 200`, `--early-stop-min-delta 1e-4`, and `--seed 42`. Use the row's values for `--hidden-dim`, `--learning-rate`, `--weight-decay`, `--dropout`, `--distill-weight`, and `--temperature`. Keep the remaining `main.py` options at their defaults and use the matching fold's teacher features from Step 3.

For example, to run the first GDSC `warm_start` fold after generating its teacher features:

```bash
python main.py \
  --data-root Data --dataset GDSC --split warm_start --fold 1 \
  --teacher-features outputs/GDSC/warm_start/fold_1/teacher_features.npz \
  --molecule-cache outputs/cache/GDSC_molecular_graphs.npz \
  --output-dir outputs/GDSC/warm_start/fold_1/student \
  --hidden-dim 512 --learning-rate 1e-4 --weight-decay 1e-5 \
  --dropout 0.1 --distill-weight 0.1 --temperature 0.05 \
  --hgnn-layers 2 --epochs 3000 --cosine-weight 0.1 \
  --distill-warmup-ratio 0.1 --distill-ramp-epochs 50 --grad-clip 5.0 \
  --early-stop-metric roc_auc --early-stop-patience 200 \
  --early-stop-min-delta 1e-4 --seed 42 --device cuda:0
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The molecular encoder test requires PyTorch Geometric.

## Repository contents

- `llmhgsyn/`: active data, instruction, molecular, model, loss, and training modules
- `finetune/llm_main.py` and `finetune/extract_teacher.py`: active teacher pipeline
- `Data/`: processed benchmark folds and text descriptions
- `tests/`: data pipeline and molecular graph checks
- `distillation/` and other historical scripts: earlier experimental code, outside the run order above

Model weights and generated training outputs are not included in the Git repository.

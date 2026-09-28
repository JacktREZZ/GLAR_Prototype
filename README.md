# GLAR-based Relation Detection Prototype

A PyTorch-based deep learning prototype for relation classification on the **SimpleQuestions v2** dataset. This implementation combines global multi-head self-attention and local windowed attention mechanisms over contextual **BERT** embeddings.

## 📌 Architecture Overview

The model extracts relationship types from short text queries through a multi-stage pipeline:

1. **Entity Marker Insertion**: Inserts special entity markers (`[S]`, `[/S]`, `[O]`, `[/O]`) into input text to identify subject and object spans.
2. **Contextual Encoding**: Uses a BERT backbone (`bert-base-uncased`) to convert marked sequences into token representations.
3. **Dual Attention Branches**:
   * **Global Branch**: Captures long-range dependencies across all tokens using full multi-head self-attention.
   * **Local Branch**: Focuses on immediate contextual neighbors using windowed self-attention.
4. **Cross-Attention Fusion Layer**: Merges global and local features dynamically using a learned gating mechanism.
5. **Entity Pooling & Classification**: Pools feature representations from entity markers and passes them to an MLP classifier for final relation prediction.

## 🛠️ Project Structure

```text
├── .gitignore           # Git ignore rules for cache, checkpoints, and model outputs
├── glar_prototype.py    # Main code containing data loading, model architecture, and training pipeline
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```
.gitignore           # Git ignore rules for cache, checkpoints, and model outputs
## 🚀 Quick Start

### 1. Prerequisites & Installation

Ensure you have Python 3.8+ installed. Install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Dataset & Input Requirements

> 💡 **Note**: Datasets are not included in this repository. 
Download and extract the SimpleQuestions v2 dataset from "https://figshare.com/articles/dataset/SimpleQuestions/5812818?file=10278561". Ensure your data directory contains the following JSON files:

```text
SimpleQuestions_v2/
├── annotated_fb_data_train.json
├── annotated_fb_data_valid.json
└── annotated_fb_data_test.json
```

Each JSON file should contain questions formatted with SPARQL queries or relation annotations compatible with the SimpleQuestions benchmark structure.

### 3. Training & Evaluation

Run the main script to start training and evaluate on the test split:

```bash
python glar_prototype.py --data-dir ./SimpleQuestions_v2
```

#### Command-line Options

| Option | Default | Description |
| --- | --- | --- |
| `--data-dir` | `./SimpleQuestions_v2` | Path to dataset directory |
| `--max-relations` | `200` | Number of most frequent relation classes to keep |
| `--bert-model` | `bert-base-uncased` | Pretrained Transformer model backbone |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `32` | Training batch size |
| `--output-dir` | `outputs/` | Directory to save best model checkpoint and artifacts |

## 📊 Evaluation Metrics

The model evaluates prediction accuracy and macro-averaged metrics on the test dataset:

* **Precision**
* **Recall**
* **F1 Score**
* **Accuracy**
# QTPB-KG: Construction of a Knowledge Graph for Birds of the Qinghai–Tibet Plateau

## Overview

This repository contains the dataset and source code associated with the study:

**QTPB-KG: Construction of a Knowledge Graph for Birds of the Qinghai–Tibet Plateau**

The study focuses on the construction of a knowledge graph for birds of the Qinghai–Tibet Plateau. A Beginning-Inside-Outside (BIO)-annotated named entity recognition (NER) dataset was constructed to support entity extraction and knowledge graph construction in the avian domain.

The dataset covers 838 bird species and contains 33,058 annotated entities across 15 entity types. UFO-NER, based on BERT-BiLSTM-UFO module-Add-LN-CRF, was used to evaluate the effectiveness of the dataset for NER tasks involving long and complex entities.

The final Qinghai–Tibet Plateau Bird Knowledge Graph (QTPB-KG) contains 11,617 entity nodes and 29,274 relations and is implemented using Neo4j.

---

## Dataset Information

The Qinghai–Tibet Plateau bird NER dataset covers 838 bird species and contains 33,058 annotated entities across 15 entity types. The dataset is annotated using the Beginning-Inside-Outside (BIO) tagging scheme.

For the named entity recognition experiments, the dataset was divided into three files:

- `train.txt`: 670 bird species
- `dev.txt`: 84 bird species
- `test.txt`: 84 bird species

This corresponds to an 8:1:1 split for training, validation, and testing.

### BIO Annotation Example

Each token is paired with a corresponding BIO label. `B-` indicates the beginning of an entity, `I-` indicates the continuation of an entity, and `O` indicates that the token does not belong to an entity.

For example:

```text
Black        B-ChineseName
necked       I-ChineseName
Crane        I-ChineseName
inhabits     O
wetlands     O
```

In this example, `Black-necked Crane` is annotated as a Chinese-name entity. For readability, English entity-type names are used in this example; the released dataset retains the original label identifiers used in the experiments.

---

## Code Information

The repository contains code for named entity recognition (NER), ablation experiments, triple extraction, entity normalization, and knowledge graph construction.

### Project Structure

```text
QTPB-KG/
├── ner_project/
│   ├── bert-base/
│   ├── data/
│   ├── model/
│   │   ├── data_util.py
│   │   ├── UFOAttention.py
│   │   ├── bert_bilstm_crf.py
│   │   ├── bbc.py
│   │   └── ufoonly.py
│   ├── train.py
│   ├── train_noattn.py
│   ├── test.py
│   ├── predict.py
│   ├── compare_test.py
│   ├── flops.py
│   └── onlyufo.py
│
├── kg/
│   ├── sbert-base-chinese-nli/
│   ├── Bird_frame.owl
│   ├── Birds_Data.owl
│   ├── Birds_Data.turtle
│   ├── all_birds_triples.txt
│   ├── batch_process.py
│   ├── similarity.py
│   └── Build birds owl.py
│
└── README.md
```

### NER Models and Experiments

| File | Description |
| --- | --- |
| `ner_project/model/data_util.py` | Reads BIO-formatted data and performs tokenization, sliding-window processing, label alignment, and batch construction for model training and testing. |
| `ner_project/model/UFOAttention.py` | Implements the UFO Module used by the complete model and the ablation models. |
| `ner_project/model/bert_bilstm_crf.py` | Implements the complete UFO-NER model: BERT + BiLSTM + UFO Module + residual connection (Add) + LayerNorm + CRF. |
| `ner_project/model/bbc.py` | Implements the BERT-BiLSTM-CRF baseline model. |
| `ner_project/train.py` | Trains UFO-NER and saves the model weights with the best validation-set F1 score. |
| `ner_project/train_noattn.py` | Trains the BERT-BiLSTM-CRF baseline model. |
| `ner_project/test.py` | Loads trained model weights and reports Precision, Recall, and F1 on the test set. |
| `ner_project/predict.py` | Loads a trained model and performs NER on user-provided text. |
| `ner_project/compare_test.py` | Compares the entity recognition results of UFO-NER and BERT-BiLSTM-CRF and generates a case comparison report. |
| `ner_project/flops.py` | Compares the parameter count and computational cost of UFO-NER and the baseline model. |

### Ablation Experiments

| File | Description |
| --- | --- |
| `ner_project/model/ufoonly.py` | Implements the ablation model containing the UFO Module and residual connection without LayerNorm. |
| `ner_project/onlyufo.py` | Trains and evaluates the ablation model. |

For the UFO-only ablation experiment, the residual connection is removed from the corresponding model implementation while retaining the UFO Module.

### Knowledge Graph Construction

| File | Description |
| --- | --- |
| `kg/batch_process.py` | Reads BIO-formatted data and converts extracted entities into structured bird–relation–attribute triples. |
| `kg/similarity.py` | Detects similar attribute values in the generated triples. Candidate pairs are manually reviewed before merging to reduce duplicate or synonymous expressions. |
| `kg/Build birds owl.py` | Converts the reviewed triples into OWL individuals, properties, and relationships to generate the knowledge graph data file. |

### Knowledge Graph Files

| File | Description |
| --- | --- |
| `kg/Bird_frame.owl` | Ontology framework used for knowledge graph construction. |
| `kg/all_birds_triples.txt` | Structured triples generated from the annotated bird data. |
| `kg/Birds_Data.owl` | Knowledge graph data in OWL format. |
| `kg/Birds_Data.turtle` | Knowledge graph data in Turtle format. |
| `kg/sbert-base-chinese-nli/` | Sentence-BERT model used during entity similarity calculation. |

---

## Usage

### 1. Prepare the Dataset

Place the BIO-formatted dataset files in the `ner_project/data/` directory:

```text
ner_project/data/
├── train.txt
├── dev.txt
└── test.txt
```

The training and evaluation scripts load the corresponding dataset files through the data processing module.

Before running the experiments, enter the NER project directory:

```bash
cd ner_project
```

### 2. Train the Baseline Model

Train the BERT-BiLSTM-CRF baseline model using:

```bash
python train_noattn.py --batch_size 8 --epochs 7
```

The best model weights are saved according to the validation-set F1 score.

### 3. Train UFO-NER

Train the complete BERT-BiLSTM-UFO Module-Add-LayerNorm-CRF model using:

```bash
python train.py --attention ufo --batch_size 8 --epochs 7
```

The best UFO-NER model weights are saved during training based on validation performance.

### 4. Evaluate the Models

Evaluate the BERT-BiLSTM-CRF baseline model:

```bash
python test.py --attention none --model_path best_model_noattn.pt
```

Evaluate UFO-NER:

```bash
python test.py --attention ufo --model_path bert_bilstm_ufo_best.pt
```

The evaluation script reports Precision, Recall, and F1 on the test set.

### Main Command-Line Arguments

| Argument | Description |
| --- | --- |
| `--attention ufo` | Uses the UFO Module in the model. |
| `--attention none` | Uses the BERT-BiLSTM-CRF baseline without the UFO Module. |
| `--batch_size 8` | Sets the training batch size to 8. |
| `--epochs 7` | Sets the maximum number of training epochs to 7. |
| `--model_path` | Specifies the path to the trained model weights used for testing. |

---

## Requirements

The main Python dependencies used in this project are:

- PyTorch 2.6.0
- Transformers 4.35.2
- Tokenizers 0.15.2
- NumPy 1.26.4
- pytorch-crf 0.7.2
- Seqeval 1.2.2
- scikit-learn 1.7.0
- SciPy 1.15.3
- tqdm 4.66.2
- fvcore 0.1.5.post20221221

The required packages can be installed using:

```bash
pip install -r requirements.txt
```

---

## Methodology

The overall workflow consists of data collection and annotation, named entity recognition, triple extraction, entity normalization, and knowledge graph construction.

Bird-related textual data from the Qinghai–Tibet Plateau were collected and manually checked before annotation. The resulting NER dataset covers 838 bird species, 33,058 annotated entities, and 15 entity types using the BIO tagging scheme.

The annotated dataset was used to train and evaluate named entity recognition models. BERT-BiLSTM-CRF was used as the baseline model, while UFO-NER incorporates the UFO Module, residual connection, and LayerNorm into the BERT-BiLSTM-CRF architecture.

The annotated entities were converted into structured triples in the form of bird name–relation–attribute value. To reduce duplicate and synonymous expressions, Sentence-BERT similarity and Jaccard similarity were used to identify candidate entity pairs, which were manually reviewed before merging.

Finally, the reviewed triples were converted into OWL data and used to construct the Qinghai–Tibet Plateau Bird Knowledge Graph (QTPB-KG), which was stored and queried using Neo4j.

---

## Citation

If you use the dataset or code in this repository, please cite the associated manuscript:

**Yuhang Luan, Luyang Zhao, and Changhong Zhang.**  
*QTPB-KG: Construction of a Knowledge Graph for Birds of the Qinghai–Tibet Plateau.*

The citation information will be updated after publication.

---

## License

License information will be provided with the public release of the dataset and source code.

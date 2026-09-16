# QTPB-KG: Construction of a Knowledge Graph for Birds of the Qinghai–Tibet Plateau

## Overview

This repository contains the dataset and source code associated with the study:

**QTPB-KG: Construction of a Knowledge Graph for Birds of the Qinghai–Tibet Plateau**

The project focuses on constructing a domain-specific knowledge graph for birds of the Qinghai–Tibet Plateau. A manually annotated named entity recognition (NER) dataset was created to support entity extraction from bird-related text and to evaluate the effectiveness of the dataset in downstream NER tasks.

The repository provides the NER dataset, model-related code, and instructions for reproducing the main experiments reported in the paper.

## Dataset Information

The Qinghai–Tibet Plateau bird NER dataset contains:

- 838 annotated text samples
- 33,058 annotated entities
- 15 entity categories
- BIO sequence-labeling annotations

The dataset is divided into training, validation, and test sets at a ratio of 8:1:1.

The dataset is primarily written in Chinese because the source material consists of Chinese bird-related texts.

For input sequences exceeding 512 tokens, a sliding-window strategy with a stride of 128 is used to reduce information loss caused by truncation.

## Code Information

The released code is mainly used for:

- Data preprocessing
- Dataset loading
- Named entity recognition model training
- Model validation and testing
- Precision, recall, and F1-score calculation

The main model evaluated in this study is UFO-NER, based on the following architecture:

```text
BERT
  ↓
BiLSTM
  ↓
UFO Module
  ↓
Residual Connection
  ↓
Layer Normalization
  ↓
CRF
```

The complete architecture is referred to as **BERT-BiLSTM-UFO-Add-LN-CRF**.

The comparison models include:

- BiLSTM-CRF
- IDCNN-CRF
- BERT-BiLSTM-CRF
- UFO-NER

## Requirements

The code is implemented in Python.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

A virtual environment is recommended.

## Usage

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd QTPB-KG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place the released dataset files in the data directory according to the repository structure.

### 4. Train the model

Run the training script provided in the repository.

For example:

```bash
python train.py
```

### 5. Evaluate the model

Run the evaluation script provided in the repository.

For example:

```bash
python evaluate.py
```

The exact script names and parameters should follow the released source code.

## Methodology

The main workflow includes data collection, text preprocessing, manual named entity annotation, NER model training and evaluation, entity normalization, and knowledge graph construction.

The annotated dataset contains 15 entity categories and follows the BIO tagging scheme.

For entity normalization, Sentence-BERT similarity and Jaccard similarity are combined with equal weights. Candidate entity pairs with a combined similarity score greater than 0.7 are manually reviewed before merging.

The final structured entities and relationships are imported into Neo4j for storage, visualization, and querying. Protégé is used for ontology construction and management.

## Reproducibility

The main NER experiments are repeated using five random seeds to reduce the influence of random initialization.

The evaluation metrics include:

- Precision
- Recall
- F1-score

Users who wish to reproduce the reported results should use the released dataset split, the same long-sequence processing strategy, and the model configurations provided in the source code.

## Data Availability

The Qinghai–Tibet Plateau bird NER dataset used in this study is released together with the research materials for review and reproducibility.

## Code Availability

The source code required to reproduce the main NER experiments is provided in this repository. Code comments and documentation intended for reviewers and readers are provided in English.

## Citation

If you use this dataset or code in your research, please cite the associated paper:

```bibtex
@article{luan2026qtpbkg,
  title   = {QTPB-KG: Construction of a Knowledge Graph for Birds of the Qinghai--Tibet Plateau},
  author  = {Luan, Yuhang and Zhao, Luyang and Zhang, Changhong},
  journal = {PeerJ Computer Science},
  year    = {2026}
}
```

The citation information should be updated after the article is formally published.

## License

License information should be provided together with the final public release of the code and dataset.

## Contact

For questions regarding the dataset, code, or associated study, please contact the corresponding author through the contact information provided in the manuscript or publication.

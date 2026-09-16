# QTPB-KG: Construction of a Knowledge Graph for Birds of the Qinghai–Tibet Plateau

## Overview

This repository contains the dataset and source code associated with the research article:

**QTPB-KG: Construction of a Knowledge Graph for Birds of the Qinghai–Tibet Plateau**

The project focuses on the construction of a domain-specific knowledge graph for birds of the Qinghai–Tibet Plateau. Bird-related information is distributed across multiple textual resources and often contains fragmented descriptions, inconsistent entity expressions, and complex semantic relationships. To organize these data in a structured form, this study constructs a bird knowledge graph covering taxonomy, morphology, habitat, distribution, ecological characteristics, and related information.

A manually annotated named entity recognition (NER) dataset was constructed to support entity extraction from bird-related text. The dataset was further used to evaluate several NER models.

The proposed UFO-NER model is based on the BERT-BiLSTM-CRF architecture and incorporates a Unit Force Operated (UFO) module, residual connection, and layer normalization. The model is mainly used to evaluate the usability of the constructed NER dataset, particularly for long entities and entities with ambiguous boundaries.

This repository provides the NER dataset and the code required for data processing, model training, and evaluation.

---

## Project Information

### Project Title

QTPB-KG: Construction of a Knowledge Graph for Birds of the Qinghai–Tibet Plateau

### Authors

- Yuhang Luan
- Luyang Zhao
- Changhong Zhang

### Affiliation

College of Intelligent Science and Engineering, Qinghai Minzu University, Xining, China

Experimental Teaching Demonstration Center for Artificial Intelligence Application Technology, Qinghai Minzu University, Xining, China

---

## Dataset Information

The Qinghai–Tibet Plateau bird NER dataset was constructed from bird-related textual information collected during the development of QTPB-KG.

The dataset contains:

- 838 annotated text samples
- 33,058 annotated entities
- 15 entity categories
- BIO sequence-labeling annotations

The annotated entities cover multiple types of bird-related information, including taxonomy, morphology, habitat, distribution, ecological characteristics, and other domain-specific attributes required for knowledge graph construction.

The dataset was divided into training, validation, and test sets at a ratio of:

```text
8 : 1 : 1

Annotation Format

The dataset follows the BIO tagging scheme.

In the BIO scheme:

B- indicates the beginning of an entity.
I- indicates a token inside an entity.
O indicates a token that does not belong to any annotated entity.

A simplified example is shown below:

黑    B-ENTITY
颈    I-ENTITY
鹤    I-ENTITY
主    O
要    O
栖    O
息    O
于    O
...

The actual entity category names are provided in the released dataset.

Each text sample is paired with its corresponding sequence labels so that the data can be directly used for named entity recognition experiments.

Dataset Split

The annotated dataset was randomly divided into training, validation, and test subsets using the following ratio:

Training set   : 80%
Validation set : 10%
Test set       : 10%

The same dataset split should be used when reproducing the experiments reported in the paper.

Long-Sequence Processing

BERT-based models have a maximum input length of 512 tokens.

For input sequences exceeding 512 tokens, a sliding-window strategy was adopted.

The stride was set to:

128

This strategy reduces information loss caused by direct truncation and helps preserve entities located near sequence boundaries.

Code Information

The source code provided in this repository is mainly used for:

Data preprocessing
BIO sequence preparation
Dataset loading
Named entity recognition model training
Model validation
Model testing
Precision calculation
Recall calculation
F1-score calculation

The main NER architecture evaluated in this study is UFO-NER.

The overall architecture is:

Input Text
    |
    v
BERT
    |
    v
BiLSTM
    |
    v
UFO Module
    |
    v
Residual Connection
    |
    v
Layer Normalization
    |
    v
CRF
    |
    v
NER Output

The complete architecture is referred to as:

BERT-BiLSTM-UFO-Add-LN-CRF

where:

BERT provides contextual word representations.
BiLSTM captures bidirectional sequence information.
UFO refers to the Unit Force Operated module.
Add represents the residual connection.
LN represents layer normalization.
CRF models dependencies between output labels.
Models Used for Comparison

The following NER models were used in the experiments:

BiLSTM-CRF
IDCNN-CRF
BERT-BiLSTM-CRF
UFO-NER

BERT-BiLSTM-CRF was used as the primary baseline for evaluating the UFO-NER architecture.

The experiments were designed primarily to evaluate the effectiveness and usability of the constructed bird NER dataset.

Experimental Reproducibility

To reduce the influence of random initialization on the experimental results, the main experiments were repeated using five random seeds.

The performance reported in the paper is based on the average results obtained from the five runs.

The main evaluation metrics are:

Precision
Recall
F1-score

For the main comparison, UFO-NER achieved an average F1-score of:

94.436%

The corresponding BERT-BiLSTM-CRF baseline achieved:

94.148%

The average improvement was approximately:

0.29 percentage points

The purpose of this comparison is not to claim a large performance improvement over existing state-of-the-art models, but to verify the effectiveness of the annotated dataset and evaluate whether lightweight structural modifications can improve entity boundary recognition.

Repository Structure

A typical structure of this repository is shown below:

QTPB-KG/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── train.txt
│   ├── dev.txt
│   └── test.txt
│
├── model/
│   ├── bert_bilstm_crf.py
│   └── ufo_ner.py
│
├── utils/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── metrics.py
│
├── train.py
├── evaluate.py
│
└── results/

The exact directory structure may differ slightly depending on the released version.

Please refer to the source files in the repository for the final implementation structure.

Requirements

The code is implemented in Python.

The required Python packages are listed in:

requirements.txt

Install all dependencies using:

pip install -r requirements.txt

A virtual environment is recommended.

For example:

python -m venv venv

On Windows:

venv\Scripts\activate

On Linux or macOS:

source venv/bin/activate

Then install the dependencies:

pip install -r requirements.txt
Installation

Clone the repository:

git clone <REPOSITORY_URL>

Enter the project directory:

cd QTPB-KG

Install the dependencies:

pip install -r requirements.txt
Data Preparation

Place the NER dataset files in the data directory.

For example:

data/
├── train.txt
├── dev.txt
└── test.txt

Each dataset file should retain the original BIO annotation format.

Do not translate the Chinese text before model training because the released model and annotation scheme are designed for the original Chinese bird-related corpus.

Training

After installing the required dependencies and preparing the dataset, the model can be trained using the training script.

For example:

python train.py

The training procedure includes:

Loading the annotated BIO dataset.
Tokenizing the input text.
Processing sequences longer than the maximum BERT input length.
Loading the NER model.
Training the model on the training set.
Monitoring performance on the validation set.
Saving the trained model.

The exact configuration parameters should follow the experimental settings described in the associated paper and the configuration provided in the source code.

Evaluation

After model training, the trained model can be evaluated using:

python evaluate.py

The evaluation procedure reports:

Precision
Recall
F1-score

The test set should remain unchanged when reproducing the results reported in the paper.

Methodology

The main research workflow consists of several stages.

1. Data Collection

Bird-related textual information from the Qinghai–Tibet Plateau was collected from relevant bird information resources.

The collected information contains descriptions of taxonomy, morphology, habitat, geographical distribution, ecological characteristics, and related bird attributes.

2. Text Preprocessing

The collected textual information was cleaned before annotation.

Preprocessing included checking the extracted text, correcting text errors, and preparing the text for subsequent manual annotation.

For text obtained using optical character recognition, the extracted content was manually checked to reduce recognition errors.

3. Named Entity Annotation

The cleaned bird-related text was manually annotated.

A total of 15 entity categories were defined according to the requirements of the bird knowledge graph.

The annotation follows the BIO sequence-labeling scheme.

The resulting dataset contains:

838 text samples
33,058 entities
15 entity categories
4. Named Entity Recognition

The annotated dataset was used to train and evaluate NER models.

The main baseline is:

BERT-BiLSTM-CRF

The proposed UFO-NER architecture extends this baseline by introducing:

UFO module
Residual connection
Layer normalization

The resulting model is:

BERT-BiLSTM-UFO-Add-LN-CRF

The model was evaluated mainly to verify the usability of the constructed dataset and examine its performance on bird-related text containing long entities and ambiguous entity boundaries.

5. Long-Entity Processing

Some entities in the dataset contain relatively long descriptive phrases.

Examples of this type of information include complex habitat and distribution descriptions.

For long input sequences, a sliding-window mechanism with a stride of 128 is used to reduce the loss of contextual information caused by the 512-token input limitation.

6. Entity Normalization

Before knowledge graph construction, entity normalization was performed to reduce duplicate and synonymous entities.

Sentence-BERT and Jaccard similarity were used to calculate semantic and lexical similarity between candidate entities.

The two similarity measures were combined using equal weights:

Sentence-BERT similarity : 50%
Jaccard similarity       : 50%

Candidate pairs with a combined similarity score greater than:

0.7

were submitted for manual verification.

Only entities confirmed to have the same meaning were merged.

7. Knowledge Graph Construction

The processed entities and relationships were organized as structured triples.

The ontology was constructed and managed using Protégé.

The final structured data were imported into Neo4j for graph storage, visualization, and querying.

The knowledge graph supports the organization of multiple types of information related to birds, including:

Taxonomic information
Morphological characteristics
Habitat information
Distribution information
Food information
Conservation information
Ecological characteristics
Other bird-related attributes
Knowledge Graph Query

Neo4j is used to store and query the constructed bird knowledge graph.

An example Cypher query for retrieving information associated with the Black-necked Crane is:

MATCH (n:`中文名` {label:["黑颈鹤"]})-[r]->(m)
RETURN n, r, m

The graph structure allows users to retrieve related entities and relationships starting from a bird entity.

For example, the Black-necked Crane node is connected to information such as morphology, habitat, call characteristics, taxonomy, distribution, food, and conservation status.

Entity Merging

Potential synonymous entities were identified through similarity calculation.

The combined similarity score is defined using:

Sentence-BERT similarity × 0.5
+
Jaccard similarity × 0.5

When the combined score is greater than 0.7, the entity pair is manually reviewed.

This process avoids automatically merging entities solely based on similarity scores and reduces incorrect entity fusion.

Software and Tools

The main tools used in this project include:

Python
BERT
BiLSTM
CRF
Sentence-BERT
Neo4j
Protégé
YEDDA

YEDDA was used during manual NER annotation.

Protégé was used for ontology construction and management.

Neo4j was used to store, visualize, and query the final bird knowledge graph.

Data Availability

The Qinghai–Tibet Plateau bird NER dataset associated with this study is provided with the research materials of this project.

The released data are intended to support:

Reproducibility of the reported experiments
Evaluation of named entity recognition methods
Research on bird-domain information extraction
Knowledge graph construction research

The dataset should be used in accordance with the license and citation requirements described below.

Code Availability

The source code required to reproduce the main NER experiments is provided in this repository.

The released code includes the components required for:

Data preprocessing
Dataset loading
Model construction
Model training
Model evaluation

Comments and documentation in the released source code are provided in English to facilitate inspection and reuse by editors, reviewers, and researchers.

Reproducibility Notes

Users who wish to reproduce the results reported in the paper should:

Use the released train, validation, and test split.
Preserve the original BIO annotations.
Use the same long-sequence processing strategy.
Set the sliding-window stride to 128 for sequences exceeding 512 tokens.
Use the model configurations provided with the source code.
Evaluate models using precision, recall, and F1-score.
Repeat the main experiments using the provided random seeds when reproducing the averaged results.

Minor differences may occur because of differences in hardware, software versions, CUDA versions, or other computational environments.

Citation

If this dataset, code, or knowledge graph is useful for your research, please cite the associated paper:

@article{luan2026qtpbkg,
  title   = {QTPB-KG: Construction of a Knowledge Graph for Birds of the Qinghai--Tibet Plateau},
  author  = {Luan, Yuhang and Zhao, Luyang and Zhang, Changhong},
  journal = {PeerJ Computer Science},
  year    = {2026}
}

The citation information will be updated after the article is formally published.

License
Source Code

The source code in this repository may be released under the MIT License.

If the MIT License is used, please include a LICENSE file in the root directory of the repository.

Dataset

The dataset may be distributed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Under CC BY 4.0, users may share and adapt the dataset provided that appropriate credit is given to the original authors.

Please cite the associated research article when using the dataset in academic publications.

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

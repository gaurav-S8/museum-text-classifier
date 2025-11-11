# Multi-Class Text Classification on Museum Records Using NLP

## Project Overview
This project focuses on classifying museum archival records into predefined institutional categories using supervised Natural Language Processing (NLP). The dataset consisted of short cultural-heritage text entries with inconsistent formatting, missing fields, and noisy labels, requiring careful cleaning and normalization before modeling.

After resolving structural and labeling issues, the text data was tokenized and processed, followed by fine-tuning multiple transformer-based models to perform multi-class classification. Model performance was evaluated using accuracy, precision, recall, macro-F1, and confusion matrices, with additional analysis to identify labeling errors and assess class imbalance effects.

## Objectives
- Pre-process and clean a noisy real-world text dataset by resolving inconsistent keys, missing values, and incorrect labels
- Explore dataset structure through token statistics, dataset distribution, and text length analysis
- Evaluate model performance using accuracy, precision, recall, macro-F1, and confusion matrices
- Diagnose dataset labeling issues and correct validation-set anomalies to ensure fair evaluation
- Compare multiple transformer architectures (BERT, RoBERTa, DistilBERT, BioMedBERT) and identify best-performing model

## Dataset
The dataset consists of text records from multiple museum institutions, each entry containing descriptive metadata associated with cultural or archival items. The data was originally provided in JSON format with inconsistent key structures and varying field completeness across institutions.

Each record includes short textual descriptions, institutional labels, and additional metadata fields. As part of preprocessing, duplicated or empty entries were removed, field names were normalized, and mislabelled instances were corrected. The final dataset was prepared for supervised learning by assigning each cleaned record to one of the museum institution classes.

## Methodology
- Cleaned and standardised noisy museum text data with inconsistent keys and missing values
- Applied tokenisation and lemmatisation to analyse text structure and common terms
- Addressed label inconsistencies and validation issues to ensure fair evaluation
- Evaluated models with accuracy, precision, recall, macro-F1, and confusion matrices
- Fine-tuned multiple transformer models for multi-class text classification

## Tools & Libraries

- Python
- Jupyter Notebook
- spaCy (tokenisation & lemmatisation)
- HuggingFace Transformers
- PyTorch (model training backend)
- Scikit-learn (metrics & evaluation)
- Pandas / JSON handling


## Results
## Key Learnings

## How to Run

This project is implemented in a Jupyter Notebook. To run it locally:

### Clone the repository
```bash
git clone https://github.com/gaurav-S8/museum-text-classifier.git
cd museum-text-classifier
```

### Launch Jupyter Notebook
```bash
jupyter notebook TextClassifier.ipynb
```

### Install required libraries (if needed)
```bash
pip install pandas numpy scikit-learn transformers torch
```

## Repository Structure
<pre>
📦 museum-text-classifier
├── Data/
├── TextClassifier.ipynb
├── README.md
└── .gitignore
</pre>

## Course Information

This project was completed as part of the **Text As Data** coursework during my MSc in Data Science, focused on applying Natural Language Processing (NLP) techniques to real-world textual datasets. The assignment involved building a multi-class classifier to automatically categorize museum archival records, with an emphasis on data cleaning, tokenization, feature extraction, and supervised machine learning evaluation.

## Acknowledgment
This project was completed as part of the **Text as Data** coursework. I would like to thank the course instructors and teaching team for their guidance and for providing the dataset and project framework.

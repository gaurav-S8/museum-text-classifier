# Multi-Class Text Classification on Museum Records Using NLP

## Project Overview
This project focuses on classifying museum archival records into predefined institutional categories using supervised Natural Language Processing (NLP). The dataset consisted of short cultural-heritage text entries with inconsistent formatting, missing fields, and noisy labels, requiring careful cleaning and normalization before modeling.

After resolving structural and labeling issues, the text data was tokenized and processed, followed by fine-tuning multiple transformer-based models to perform multi-class classification. Model performance was evaluated using accuracy, precision, recall, macro-F1, and confusion matrices, with additional analysis to identify labeling errors and assess class imbalance effects.

## Objectives
- Pre-process and clean a noisy real-world text dataset by resolving inconsistent keys, missing values, and incorrect labels
- Explore dataset structure through token statistics, dataset distribution, and text length analysis
- Evaluate model performance using accuracy, precision, recall, macro-F1, and confusion matrices
- Diagnose dataset labeling issues and correct validation-set anomalies to ensure fair evaluation
- Compare multiple transformer architectures (BERT, RoBERTa, DistilBERT, BioMedBERT) and identify the best-performing model

## Dataset
The dataset consists of text records from multiple museum institutions, each entry containing descriptive metadata associated with cultural or archival items. The data was originally provided in JSON format with inconsistent key structures and varying field completeness across institutions.

Each record includes short textual descriptions, institutional labels, and metadata fields. As part of preprocessing, duplicated or empty entries were removed, field names were harmonized, and mislabeled instances were corrected. The final dataset was prepared for supervised learning by assigning each cleaned record to one of the museum institution classes.

## Methodology
- Cleaned and standardized noisy museum text data with inconsistent keys and missing values
- Applied tokenization and lemmatization to analyze text structure and common terms
- Addressed label inconsistencies and validation issues to ensure fair evaluation
- Fine-tuned multiple transformer models for multi-class text classification
- Evaluated models using accuracy, precision, recall, macro-F1, and confusion matrices

## Tools & Libraries

- Python
- Jupyter Notebook
- spaCy (tokenisation & lemmatisation)
- HuggingFace Transformers
- PyTorch (model training backend)
- Scikit-learn (metrics & evaluation)
- Pandas / JSON handling

## Results

Multiple transformer models were tested for multi-class museum record classification.  
General-domain BERT models performed best, while a biomedical domain model performed worst due to domain mismatch.

| Model | Test Accuracy | Test Macro-F1 | Notes |
|-------|--------------:|--------------:|------|
| BERT-Base-Uncased | **0.88** | **0.88** | Best overall general performance |
| DistilBERT-Base-Uncased | 0.86 | 0.86 | Nearly matches BERT with fewer parameters |
| RoBERTa-Base | 0.86 | 0.86 | Strong performer but less consistent |
| BioMedBERT | 0.82 | 0.81 | Domain mismatch → lower performance |

> **General-domain transformers were more effective than domain-specific models.**  
> Class imbalance affected minority class performance, making macro-F1 a better metric than accuracy.

## Key Learnings
- Real-world text datasets require cleaning and label correction before modeling
- Macro-F1 is more reliable than accuracy for imbalanced multi-class datasets
- Small label errors and class imbalance meaningfully impact evaluation quality
- General-domain BERT models performed best; domain-specific BioMedBERT underperformed

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

This project was completed as part of the Text As Data coursework during my MSc in Data Science. The module focused on practical Natural Language Processing, including text preprocessing, tokenization, transformer-based modeling, and supervised learning evaluation on real-world datasets.

## Acknowledgment

This project was completed as part of the Text As Data course. Many thanks to the teaching team for providing the dataset, project structure, and academic guidance.

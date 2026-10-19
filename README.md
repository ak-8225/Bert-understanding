# BERTopic Accuracy & Validation Dashboard

An interactive dashboard that validates BERTopic topics against ground-truth categories using NMI, ARI, and Purity; compares BERTopic and LDA with C_v and NPMI coherence; and reports topic stability.

## Tech stack

- BERTopic
- Sentence-BERT (`all-MiniLM-L6-v2`)
- UMAP
- HDBSCAN
- Streamlit

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Live demo: [ADD LINK AFTER DEPLOYING]

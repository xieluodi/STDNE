# STDNE

Source code for **STDNE: Dynamic Network Embedding with Structural Similarity and Time Information**

# Requirements

- Python 3.8
- numpy
- PyTorch
- Memory  >= 16 GB
- GPUs

# Description

```
STDNE/
│── Dataset.py: Dataloader for STDNE
│── Evaluation.py: Evaluation function for STDNE, contain Node classification.
│── STDNE.py: Architecture for STDNE and training module.
├── data
│   └── dblp
│       ├── dblp.txt: each line is a temporal edge with the format (node1 \t node2 \t timestamp)
│       └── node2label.txt: node label data with the format (node_name, label)
│   └── Tmall: will be available soon!
│   └── Eucore: will be available soon!
├── convert 
│   └── dblp
│       ├── *.txt: each file is a dblp's temporal graph, format in ORCA's requirement.
│   	├── orca.exe: ORCA for computing GDV.
│   	└── run.py: Script for using orca to generate multi GDVs. 
├── README.md
```

# Usage

**Please run STDNE in the following order.**

1. Use `cd convert/dblp/ && python run.py` to generate dblp's GDV **(If exists .out files in convert/dblp/ folder, skip this)**.
2. Return to `STDNE` folder, use `python STDNE` to begin training.
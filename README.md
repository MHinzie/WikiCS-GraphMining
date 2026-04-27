# WikiCS Graph Mining: Node Classification with GNNs

> **Can graph structure improve node classification on Wikipedia articles?**  
> A graph mining and GNN study using the WikiCS benchmark dataset.

**Start here: [main_notebook.ipynb](main_notebook.ipynb)**

**[Project Video](https://www.youtube.com/watch?v=flj21zK8mMs))*

---

## Overview

This project uses **WikiCS** ~11,700 Wikipedia articles about Computer Science, connected by 431,726 hyperlinks. The goal of this project is to explore whether graph structure can meaningfully improve article classification. The 10 CS categories overlap heavily in text feature space, and this raises the central question: does knowing which articles *link to each other* help us predict what *category* they belong to?

The short answer is yes. An MLP using only text features achieves **71.9% accuracy**. Adding graph structure via a GCN pushes that to **79.3%**, an 8-point gain that directly shows the value of exploiting hyperlink structure for classification.

---

## Research Questions

1. **Can graph mining techniques reveal the underlying category structure of WikiCS?**  
   *(Degree centrality, PageRank, community detection — course techniques)*

2. **How well can feature-only methods classify articles without graph structure?**  
   *(MLP, K-Means, Random Forest — course techniques)*

3. **Does a GNN combining graph structure + node features outperform these baselines?**  
   *(GCN and GraphSAGE — beyond-course techniques)*

---

## Key Results

| Model | Test Accuracy | Macro F1 |
|-------|:---:|:---:|
| MLP (features only) | 71.87% | 0.678 |
| GraphSAGE | 78.72% | 0.759 |
| **GCN** | **79.32%** | **0.768** |

**Takeaway:** Graph structure provides a consistent improvement over using text features alone. The biggest gains went to the *smallest* classes (Class 0: +14.9 F1, Class 5: +11.5 F1), where limited training examples made neighborhood information very valuable.

---

## Dataset

**WikiCS** — available via PyTorch Geometric (`torch_geometric.datasets.WikiCS`)

- **11,701 nodes** (Wikipedia CS articles)
- **431,726 edges** (hyperlinks, treated as undirected)
- **300-dimensional** node features
- **10 classes** (CS topic categories)
- **Homophily ratio: 0.6547** (65% of edges connect same-category articles)

No manual download needed — PyTorch Geometric fetches the dataset automatically when you run the notebook.

**Notable data characteristics:**
- Significant class imbalance, addressed by using F1 as the primary metric
- One hub node with degree 3,324 (connects to nearly 1/3 of all articles), which motivates neighbor-sampling approaches like GraphSAGE

---

## Repo Structure

```
WikiCS-GraphMining/
│
├── main_notebook.ipynb          ← Main deliverable (start here!)
├── requirements.txt             ← Full Colab environment snapshot
├── README.md
│--Checkpoints
--├── checkpoint_1.ipynb       ← Dataset selection & EDA
  └── checkpoint_2.ipynb       ← Research question formation & feasibility
```

---

## How to Reproduce

This project was built and run in **Google Colab**.

1. Open `main_notebook.ipynb` in Colab (or clone the repo and open locally)
2. Run all cells in order — the dataset downloads automatically via `torch_geometric`
3. No GPU required, but Colab's free T4 speeds things up

To install dependencies locally:
```bash
pip install -r requirements.txt
```

---

## Key Dependencies

| Package | Version |
|---------|---------|
| Python | 3.12 |
| torch | 2.10.0+cpu |
| torch_geometric | 2.7.0 |
| scikit-learn | 1.6.1 |
| networkx | 3.6.1 |
| matplotlib | 3.10 |

Full dependency list with pinned versions: see [`requirements.txt`](requirements.txt)

---

## Citation

```bibtex
@article{mernyei2020wiki,
  title={Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks},
  author={Mernyei, Péter and Cangea, Cătălina},
  journal={arXiv preprint arXiv:2007.02901},
  year={2020}
}
```

# WikiCS-GraphMining

Since you want it to sound like you—grounded in the data we actually found—here is a professional but straightforward README you can use. You can copy-paste this directly into a file named README.md in your GitHub repo.

WikiCS Article Classification
This project uses the WikiCS dataset to explore how we can use graph data to categorize Wikipedia articles. It's a collection of about 11,000 articles related to Computer Science, where the "links" are actual hyperlinks between pages.

📊 Exploratory Data Analysis (EDA)
Before jumping into the model, I ran some tests to see what the data actually looks like. Here’s the breakdown:

The Scale: 11,701 articles and 431,726 hyperlinks.

The "Hub" Problem: I found a massive outlier with 3,324 links. This "Super-Hub" article connects to nearly 30% of the entire dataset, which could easily bias a basic model.

The Homophily Gap: My analysis shows a 65.5% homophily ratio. This means that while most articles link to similar topics, about 35% of the links are "cross-talk" between different CS fields (like a Security article linking to a Database page).

Class Imbalance: There's a 9x difference in size between the biggest category (Security) and the smallest (Linguistics), so the model has to be careful not to just ignore the smaller groups.

🛠 My Approach
The goal is to predict which of the 10 CS categories an article belongs to.

Core Techniques
Graph Connectivity: Analyzing how articles cluster together.

Feature Analysis: Utilizing the 300-dimensional "GloVe" word embeddings that come with each article.

External Technique: Graph Neural Networks (GNN)
I’m using a GNN as my beyond-course technique. Most basic methods just look at the links or just the text. I'm using a GNN because it can do feature-aware aggregation. It uses the 300-dimensional text features to "score" the links, helping the model figure out which of those 35% "cross-talk" links are actually relevant and which ones should be ignored.

🚀 How to Run
The analysis is contained in a Google Colab notebook:

Open WikiCS_Analysis.ipynb

Run the cells to load the dataset via torch_geometric.

The notebook includes integrity tests to verify the data loaded correctly (no NaNs, correct dimensions).

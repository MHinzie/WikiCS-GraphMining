# WikiCS-GraphMining

This project uses the WikiCS dataset to explore how to use graph data to categorize Wikipedia articles. It's a collection of about 11,000 articles related to Computer Science, where the "links" are actual hyperlinks between pages.

Exploratory Data Analysis (EDA)
Before jumping into the model, I ran some tests to see what the data actually looks like.

11,701 articles and 431,726 hyperlinks. The edges are normally half the amount but, because I plan I using a Graph Neural Network it makes more sense to make the graph undirected. 

Problems:
1. From the initial EDA I found a massive outlier with 3,324 links. This hub article connects to nearly 1/3 of the entire dataset, which could easily bias a basic model.

2. My analysis shows a 0.6547 homophily ratio. This means that while most articles link to similar topics, about 35% of the links connect to different CS fields.

3. There's a 9x difference in size between the largest category and the smallest, so the model has to be careful not to just ignore the smaller groups.

Goal:
The goal is to predict which of the 10 CS categories an article belongs to.

Core Technique:
Graph Mining: Analyzes how articles cluster together.


External Technique: Graph Neural Networks (GNN)
I’m using a GNN as my beyond-course technique. I'm using a GNN because it can do feature-aware aggregation. It uses the 300-dimensional text features to score the links, which helps the model figure out which connections are relevant and which should be ignored.


The analysis is contained in the Google Colab notebook 527004066_ProjectCheckpoint1.

Run the cells to load the dataset via torch_geometric.

The notebook includes integrity tests to verify the data loaded correctly.

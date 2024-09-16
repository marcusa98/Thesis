# Measuring Quality of Unsupervised Learning: Evaluation of Density-Based Clustering

This repository organizes datasets and functions that I used and implemented for my masterthesis. 


## Repository Structure
`data/`: This folder contains datasets used for the analysis. 
The datasets that are not available via ClustPy or UCI Machine Learning Repository are included here. A description of where each of the other dataset used int my thesis was sourced from is also provided.

`functions/:`
- `dcsi.py` density-based CVI between 0 and 1
- `cvdd.py` density-based CVI between 0 and inf
- `dunn_index.py` non-density based CVI between 0 and inf


## Citations
Jana Gauss and Fabian Scheipl and Moritz Herrmann, "DCSI -- An improved measure of cluster separability based on separation and connectedness". https://arxiv.org/abs/2310.12806
L. Hu and C. Zhong, "An Internal Validity Index Based on Density-Involved Distance," in IEEE Access, vol. 7, pp. 40038-40051, 2019,
doi: 10.1109/ACCESS.2019.2906949. keywords: {Indexes;Density measurement;Clustering algorithms;Estimation;Partitioning algorithms;Periodic structures;Information science;Crisp clustering;cluster validity index;arbitrary-shaped clusters},
Dunn, J. C. (1973). A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters. Journal of Cybernetics, 3(3), 32–57. https://doi.org/10.1080/01969727308546046

# Measuring Quality of Unsupervised Learning: Evaluation of Density-Based Clustering

## Abstract
Despite the existence of appropriate evaluation metrics, it remains a significant challenge to assess the quality of a clustering that contains clusters of arbitrary shape. This can be attributed to both individual weaknesses in the design of the existing measures as well as the inherent complexity of the subject. The recently developed evaluation metric DISCO (Density-based Internal Evaluation Score for Clustering Outcomes) is introduced through small examples and explanations in a blog post, and is compared in a series of experiments with three existing density-based metrics. It proves to be more reliable than its peers, especially for datasets in higher-dimensional spaces. The insights gained from the experiments are used to create a collection of density-based real-world datasets. Such a collection does not yet exist, although it is essential for the development of new and improvement of existing density-based clustering algorithms.

## About this Repo
This repository organizes files and functions that I used to conduct my masterthesis.
Necessary dependencies to run the files in this repo can be installed through `pip install -r requirements.txt`.

## Repository Structure
`data/`: This folder contains datasets used for the analysis. 
The datasets that are not available via ClustPy or UCI Machine Learning Repository are included here. A description of where each of the other dataset used int my thesis was sourced from is also provided.

`functions/:`
- `dcsi.py` density-based CVI between 0 and 1
- `cvdd.py` density-based CVI between 0 and inf
- `dunn_index.py` non-density based CVI between 0 and inf

**Note**
I used an external Implementation for DBCV which can be found here: https://github.com/FelSiq/DBCV

## Citations
Jana Gauss and Fabian Scheipl and Moritz Herrmann, "DCSI -- An improved measure of cluster separability based on separation and connectedness". https://arxiv.org/abs/2310.12806
L. Hu and C. Zhong, "An Internal Validity Index Based on Density-Involved Distance," in IEEE Access, vol. 7, pp. 40038-40051, 2019,
doi: 10.1109/ACCESS.2019.2906949. keywords: {Indexes;Density measurement;Clustering algorithms;Estimation;Partitioning algorithms;Periodic structures;Information science;Crisp clustering;cluster validity index;arbitrary-shaped clusters},
Dunn, J. C. (1973). A Fuzzy Relative of the ISODATA Process and Its Use in Detecting Compact Well-Separated Clusters. Journal of Cybernetics, 3(3), 32–57. https://doi.org/10.1080/01969727308546046

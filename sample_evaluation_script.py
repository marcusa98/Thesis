from cvi import *
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from clustpy.utils import EvaluationDataset, EvaluationAlgorithm, EvaluationMetric, evaluate_multiple_datasets
import dbcv #python -m pip install "git+https://github.com/FelSiq/DBCV"

# create Label encoder 
label_encoder = LabelEncoder()

#fetch Iris
iris = fetch_ucirepo(id=53)

# convert to np.array
X_iris = np.array(iris.data.features)
y_iris = iris.data.targets
# encode from Iris-setosa, Iris-virginica... to numbers
y_iris = label_encoder.fit_transform(y_iris['class'])


######SYNTH HIGH

synth_high_100 = np.load('data/high_data_100.npy')
X_synth_high_100 = pd.DataFrame(synth_high_100)
y_synth_high_100 = X_synth_high_100.pop(100)

X_synth_high_100 = np.array(X_synth_high_100)
y_synth_high_100 = label_encoder.fit_transform(np.array(y_synth_high_100))

########SEEDS

X_seeds = pd.read_csv("data/seeds_dataset.txt", header = None, delim_whitespace=True)
y_seeds = np.array(X_seeds.pop(7))
y_seeds = label_encoder.fit_transform(y_seeds)



class TrueLabelClustering:
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters 
        self.labels_ = None 
        self.X = None

    def fit(self, X):
        self.X = X
        # Ensure that `labelsList` is a global or passed-in variable, and labels are valid
        for labels in labelsList:
            if X.shape[0] == labels.shape[0]:
                self.labels_ = labels
                break  # Stop after finding the matching labels
        if self.labels_ is None:
            raise ValueError("No matching labels found for the input data.")



# Attention: Does only work if all datasets have different length
labelsList = [y_iris, y_synth_high_100, y_seeds]


datasets = [EvaluationDataset("Iris", data = np.array(X_iris), labels_true = np.array(y_iris)),
            EvaluationDataset("Synth High 100", data = np.array(X_synth_high_100), labels_true = np.array(y_synth_high_100)),
            EvaluationDataset("Seeds", data = np.array(X_seeds), labels_true = np.array(y_seeds))]

algorithms = [EvaluationAlgorithm("TrueLabels", TrueLabelClustering, deterministic = False)] # "deterministic = True" raises exception, but doesnt change results

metrics = [EvaluationMetric("Dunn Index", dunn_index, use_gt = False),
           EvaluationMetric("DBCV", dbcv.dbcv, use_gt = False, params = {"enable_dynamic_precision": True, "check_duplicates": False, "noise_id": -1, "use_original_mst_implementation": True}), # https://github.com/FelSiq/DBCV?tab=readme-ov-file#install
           EvaluationMetric("DCSI", dcsi, use_gt = False),
           EvaluationMetric("CVDD", cvdd, use_gt = False)]

aggregations = []

df = evaluate_multiple_datasets(datasets, algorithms, metrics, n_repetitions=1, aggregation_functions = aggregations,
                                add_runtime=False, add_n_clusters=False, save_path=None,
                                save_intermediate_results=False)

print(df)
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tsai.all import *
from dtaidistance import dtw
from scipy.stats import wasserstein_distance
from art.attacks.evasion import (
    BasicIterativeMethod, DeepFool, CarliniL2Method, MomentumIterativeMethod,
    ElasticNet, AutoProjectedGradientDescent, FastGradientMethod, ZooAttack, BoundaryAttack
)
from art.estimators.classification import PyTorchClassifier
from sklearn.model_selection import train_test_split
import itertools
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_ids = ["RacketSports", "NATOPS", "UWaveGestureLibrary", "Cricket", "Ering", "BasicMotions", "Epilepsy"]

attack_decisions_dict = {
    "RacketSports": [(1, 5), (1, 0), (1, 0), (1, 5), (1, 2)],
    "NATOPS": [(1, 4), (1, 3), (1, 0), (1, 0), (1, 0)],
    "UWaveGestureLibrary": [(1, 1), (1, 3), (1, 4), (1, 2), (1, 0)],
    "Cricket": [(1, 1), (1, 2), (1, 0), (1, 0), (1, 3)],
    "Ering": [(1, 4), (1, 2), (1, 0), (1, 5), (1, 3)],
    "BasicMotions": [(1, 4), (1, 2), (1, 5), (1, 4), (1, 2)],
    "Epilepsy": [(1, 0), (1, 5), (1, 1), (1, 2), (1, 0)]
}

def training_loss(pred, target):
    return F.cross_entropy(pred, target)

def load_data(dsid):
    X_train, y_train, X_test, y_test = get_UCR_data(dsid, return_split=True)
    unique_labels = np.unique(y_train)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_map[label] for label in y_train])
    y_test = np.array([label_map[label] for label in y_test])

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    tfms = [None, [Categorize()]]
    dsets_train = TSDatasets(X_train_split, y_train_split, tfms=tfms, splits=None, inplace=True)
    dls_train = TSDataLoaders.from_dsets(dsets_train.train, dsets_train.train, bs=16, shuffle=False, batch_tfms=[TSStandardize()], num_workers=0)

    c_in = dls_train.vars
    c_out = dls_train.c

    model = CustomCNN(c_in, c_out).to(device)

    return X_train_split, X_val_split, model, dls_train

def pad_or_truncate_feature_dim(seq1, seq2):
    min_features = min(seq1.shape[-1], seq2.shape[-1])
    return seq1[..., :min_features], seq2[..., :min_features]

class CustomCNN(nn.Module):
    def __init__(self, input_size, output_size, num_filters=64, kernel_size=3, dropout=0.5):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size)
        self.fc1 = nn.Linear(num_filters, output_size)
        self.dropout = nn.Dropout(dropout)
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')  
            elif 'bias' in name:
                torch.nn.init.zeros_(param)  

    def forward(self, x):
        x = x.permute(0, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        x = self.dropout(x)
        return self.fc1(x)

def create_art_classifier(model, input_shape, nb_classes, device):
    return PyTorchClassifier(
        model=model,
        loss=F.cross_entropy,
        input_shape=input_shape,
        nb_classes=nb_classes,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.002),
        device_type=device.type
    )

def apply_attack(X, classifier, attack_number):
    attacks = {
        0: BasicIterativeMethod(estimator=classifier, eps=0.1),
        1: DeepFool(classifier=classifier),
        2: CarliniL2Method(classifier=classifier),
        3: MomentumIterativeMethod(estimator=classifier, eps=0.1),
        4: ElasticNet(classifier=classifier),
        5: BoundaryAttack(estimator=classifier, targeted=False),
        10: FastGradientMethod(estimator=classifier, eps=0.1),
        7: ZooAttack(classifier=classifier, confidence=0.5, targeted=False),
        8: AutoProjectedGradientDescent(estimator=classifier, eps=0.1)
    }
    return attacks.get(attack_number, lambda X: X).generate(X) if attack_number in attacks else X

def apply_attack_decision(X_split, classifier, attack_decisions):
    return np.concatenate([
        apply_attack(piece, classifier, attack_number) if attack_flag == 1 else piece
        for piece, (attack_flag, attack_number) in zip(X_split, attack_decisions)
    ], axis=0)

dtw_matrix = pd.DataFrame(np.nan, index=dataset_ids, columns=dataset_ids)
wasserstein_matrix = pd.DataFrame(np.nan, index=dataset_ids, columns=dataset_ids)

attacked_val_data = {}

for dsid in dataset_ids:
    print(f"Processing dataset: {dsid}")

    X_train, X_val, model, dls_train = load_data(dsid)

    classifier = create_art_classifier(model, input_shape=(dls_train.vars, X_train.shape[2]), nb_classes=dls_train.c, device=device)

    X_val_pieces = np.array_split(X_val, 5)

    attacked_X_val = apply_attack_decision(X_val_pieces, classifier, attack_decisions_dict[dsid])

    attacked_val_data[dsid] = attacked_X_val

for dsid1, dsid2 in itertools.combinations(dataset_ids, 2):
    X_val1, X_val2 = attacked_val_data[dsid1], attacked_val_data[dsid2]

    X_val1, X_val2 = pad_or_truncate_feature_dim(X_val1, X_val2)

    min_samples = min(len(X_val1), len(X_val2))

    dtw_similarity = np.mean([
        fastdtw(X_val1[i], X_val2[i], dist=euclidean)[0]
        for i in range(min_samples)
    ])

    X_val1_reduced = X_val1[:min_samples].reshape(-1, X_val1.shape[-1])  
    X_val2_reduced = X_val2[:min_samples].reshape(-1, X_val2.shape[-1])

    wasserstein_sim = np.mean([
        wasserstein_distance(X_val1_reduced[:, d], X_val2_reduced[:, d])
        for d in range(X_val1_reduced.shape[1]) 
    ])

    dtw_matrix.loc[dsid1, dsid2] = dtw_matrix.loc[dsid2, dsid1] = dtw_similarity
    wasserstein_matrix.loc[dsid1, dsid2] = wasserstein_matrix.loc[dsid2, dsid1] = wasserstein_sim





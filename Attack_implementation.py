
from tsai.all import *
from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ZooAttack, FastGradientMethod
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch


dsid = 'NATOPS'
X_train, y_train, X_test, y_test = get_UCR_data(dsid, return_split=True)
unique_labels = np.unique(y_train)
label_map = {label: idx for idx, label in enumerate(unique_labels)}

y_train = np.array([label_map[label] for label in y_train])
y_test = np.array([label_map[label] for label in y_test])

tfms = [None, [Categorize()]]
dsets = TSDatasets(X_train, y_train, tfms=tfms, splits=None, inplace=True)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.train, bs=32, batch_tfms=[TSStandardize()], num_workers=0)

model = create_model(LSTM_FCN, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
learn.fit_one_cycle(28,0.00283411797693141)

classifier = PyTorchClassifier(
model=model,
loss=nn.CrossEntropyLoss(),
optimizer=optim.Adam(model.parameters(), lr=0.00041901959985071426),
input_shape=(X_train.shape[1], X_train.shape[2]),
nb_classes=len(unique_labels),
)

# Generating adversarial examples using BasicIterativeMethod
#attack = BasicIterativeMethod(estimator=classifier, eps=0.1)
#attack = DeepFool(classifier=classifier)
#attack = CarliniL2Method(classifier=classifier)
#attack = MomentumIterativeMethod(estimator=classifier, eps=0.1)
#attack = ElasticNet(classifier=classifier)
#attack = AutoProjectedGradientDescent(estimator=classifier, eps=0.1)
#attack = HopSkipJump(classifier=classifier)
attack = FastGradientMethod(estimator=classifier, eps=0.1)
#attack = SaliencyMapMethod(classifier=classifier)
#attack = ZooAttack(classifier=classifier, confidence=0.5, targeted=False, learning_rate=1e-2, max_iter=100, binary_search_steps=1, initial_const=1e-3, abort_early=True, use_resize=False, use_importance=False, nb_parallel=1, batch_size=1)
#attack = BoundaryAttack(estimator=classifier, targeted=False, max_iter=10)

X_test_adv = attack.generate(x=X_test)

logits_original = classifier.predict(X_test)
logits_adv = classifier.predict(X_test_adv)

probs_original = nn.functional.softmax(torch.tensor(logits_original), dim=1).numpy()
probs_adv = nn.functional.softmax(torch.tensor(logits_adv), dim=1).numpy()
orig_predictions = np.argmax(probs_original, axis=1)
adv_predictions = np.argmax(probs_adv, axis=1)

accuracy_orig = np.mean(orig_predictions == y_test)
print(f'Accuracy on original examples: {accuracy_orig * 100:.2f}%')

accuracy_adv = np.mean(adv_predictions == y_test)
print(f'Accuracy on adversarial examples: {accuracy_adv * 100:.2f}%')

f1 = f1_score(y_test, adv_predictions, average='weighted')
print(f'Success Score (F1-score) on adverserial test examples: {f1 * 100:.2f}%')

f1_orig = f1_score(y_test, orig_predictions, average='weighted')
print(f'Success Score (F1-score) on  test examples: {f1_orig * 100:.2f}%')

successful_attacks = np.sum(orig_predictions != adv_predictions)
asr = (successful_attacks / len(y_test)) * 100
print(f'Attack Success Rate (ASR): {asr:.2f}%')

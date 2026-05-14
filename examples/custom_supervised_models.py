"""Example custom models for NLP Pilot's Prediction > Manage Models dialog.

Each class or function returns a scikit-learn compatible estimator. NLP Pilot
handles text vectorization separately, so these estimators receive numeric
feature matrices during training.
"""

from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesTextClassifier:
    def __new__(cls, random_state=None):
        return MultinomialNB(alpha=1.0)


def sgd_logistic_classifier(random_state=None):
    return SGDClassifier(
        loss="log_loss",
        max_iter=1000,
        tol=1e-3,
        random_state=random_state,
    )


def elastic_net_regressor(random_state=None):
    return ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=random_state,
        max_iter=2000,
    )

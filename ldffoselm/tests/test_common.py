from sklearn.utils.estimator_checks import check_estimator

from ldffoselm import (DffOsElm, TemplateClassifier, TemplateTransformer)

def test_estimator():
    return check_estimator(DffOsElm)


def test_classifier():
    return check_estimator(TemplateClassifier)
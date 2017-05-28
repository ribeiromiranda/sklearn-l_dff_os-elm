"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skltemplate.template.TemplateEstimator`
"""
import numpy as np
from ldffoselm import DffOsElm
from matplotlib import pyplot as plt

from scipy.io import loadmat

data = loadmat('../dSetDebutanizerOff.mat', matlab_compatible=True)

X = data['input']
y = data['output']

estimator = DffOsElm()
estimator.fit(X, y)

y_pred = []
for i in range(estimator.get_params()['slide_window'] + 1, X.shape[0]):
    y_pred.append(estimator.predict(X[i], y[i])[0])

plt.plot(y_pred)
plt.plot(y)
plt.show()

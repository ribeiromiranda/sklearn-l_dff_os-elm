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

estimator = DffOsElm(random_state=10)
estimator.fit(X, y)


y_pred = []
#X.shape[0]
for i in range(estimator.get_params()['slide_window'] + 1, len(y)):
    y_pred.append(estimator.predict(X[i], y[i][0])[0])

plt.plot(y_pred, linewidth=0.5)
plt.plot(y, linewidth=0.5)
plt.show()

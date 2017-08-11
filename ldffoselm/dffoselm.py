"""
This is a module to be used as a reference for building other modules
"""

import numpy as np
import scipy.linalg
from numpy import matmul as mm
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances, mean_squared_error
from sklearn.model_selection import KFold



class DffOsElm(BaseEstimator):
    """ A template estimator to be used as a reference implementation .

    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    kfold : int, optional
        Number of folds for the k-fold cross-validation (kfold < number of samples)
    """
    def __init__(self, kfold=10, slide_window=10, min_hidden_neurons_layer=1, max_hidden_neurons_layer=20,
                 activation_function='sig', random_state=None):
        if max_hidden_neurons_layer > slide_window:
            max_hidden_neurons_layer = slide_window

        self.kfold = kfold
        self.slide_window = slide_window
        self.min_hidden_neurons_layer = min_hidden_neurons_layer
        self.max_hidden_neurons_layer = max_hidden_neurons_layer
        self.activation_function = activation_function
        self.random_state = random_state

        self.__rng = check_random_state(random_state)

    def get_params(self, deep=True):
        return {
            "kfold": self.kfold,
            "slide_window": self.slide_window,
            "min_hidden_neurons_layer": self.min_hidden_neurons_layer,
            "max_hidden_neurons_layer": self.max_hidden_neurons_layer,
            "activation_function": self.activation_function,
            "random_state": self.random_state
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def fit(self, X, y):
        """A reference implementation of a fitting function

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """

        self.__first_phase(X[0:self.slide_window], y[0:self.slide_window])
        X, y = check_X_y(X, y)

        # Return the estimator
        return self

    def __first_phase(self, X, y):
        max_hidden_neurons_layer = self.max_hidden_neurons_layer
        kf = KFold(n_splits=self.kfold, shuffle=False, random_state=self.random_state)
        for train_index, test_index in kf.split(X):
            if train_index.size < max_hidden_neurons_layer:
                max_hidden_neurons_layer = train_index.size

        mean_mse = []
        for number_hidden_neurons_layer in range(self.min_hidden_neurons_layer, max_hidden_neurons_layer + 1):
            list_tests = []
            for train_index, test_index in kf.split(X, y):
                data_scale_train = self.__scaleZMean1Var(X[train_index], y[train_index])
                data_scale_test = self.__scale(X[test_index], data_scale_train['x_mean'], data_scale_train['x_std'])

                elm = self.__elm_fit(data_scale_train['x_scale'], data_scale_train['y_scale'], number_hidden_neurons_layer)
                y_pred = self.__elm_predict(data_scale_test, elm)
                y_pred_rescale = self.__rescale(y_pred, data_scale_train['y_mean'], data_scale_train['y_std'])

                list_tests.append(mean_squared_error(y[test_index], y_pred_rescale))

            mean_mse.append(np.mean(list_tests))

        data_scale = self.__scaleZMean1Var(X, y)
        self.__x_mean = data_scale['x_mean']
        self.__x_std = data_scale['x_std']
        self.__y_mean = data_scale['y_mean']
        self.__y_std = data_scale['y_std']

        self.__var_x = np.power(data_scale['x_std'], 2)
        self.__var_y = np.power(data_scale['y_std'], 2)

        self.__sw = self.slide_window

        X_scaled = data_scale['x_scale']
        y_scaled = data_scale['y_scale']

        number_hidden_neurons_layer = np.argmin(mean_mse)
        self._elm = self.__elm_fit(X_scaled, y_scaled, number_hidden_neurons_layer)

    def __elm_fit(self, X, y, number_hidden_neurons_layer):
        bias = self.__rng.rand(1, number_hidden_neurons_layer) * 2 - 1
        input_weight = self.__rng.rand(number_hidden_neurons_layer, X.shape[1]) * 2 - 1

        elm = ELM(number_hidden_neurons_layer, input_weight, bias)

        if self.activation_function == 'sig':
            H = self.__sig_activation_function(X, elm)

        elm.covariance_matrix = scipy.linalg.pinv(np.matmul(H.transpose(), H))

        elm.beta = np.matmul(np.linalg.pinv(H),  y)
        return elm

    def predict(self, X, y=None):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        y : array-like of shape = [n_samples, n_features]
            The real output samples.

        Returns
        -------
        y_pred : array of shape = [n_samples]
                 Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        X_scale = self.__scale(X, self.__x_mean, self.__x_std)
        y_pred = self.__elm_predict(X_scale, self._elm)
        y_pred_rescale = self.__rescale(y_pred, self.__y_mean, self.__y_std)

        if y is None:
            return y_pred_rescale

        x_y_scaled = self.__update_mean_std(X, y)
        X_scale = x_y_scaled[0]
        y_scale = x_y_scaled[1]
        self.__fit_dff_elm(X_scale, y_scale, self._elm)

        return y_pred_rescale

    def __elm_predict(self, X, elm):
        assert elm.beta is not None
        assert elm.input_weight is not None

        H = self.__sig_activation_function(X, elm)
        return np.matmul(H, elm.beta)

    def __fit_dff_elm(self, X, y, elm):
        assert elm.beta is not None
        assert elm.input_weight is not None
        assert elm.covariance_matrix is not None

        H = self.__sig_activation_function(X, elm)
        H_t = H.transpose()

        ep = y - mm(H, elm.beta)
        ks = mm(mm(H, elm.covariance_matrix), H_t)
        pp = (mm(elm.covariance_matrix, H_t) / (1 + ks)) * ep
        elm.beta += pp

        # Updating other parameters
        if ks > 0:
            eps = elm.l - (1 - elm.l) / ks
            elm.covariance_matrix -= mm(mm(mm(elm.covariance_matrix, H_t), H), elm.covariance_matrix) / (np.linalg.inv(eps) + ks)

        elm.la = elm.l * (elm.la + (ep * ep)/(1 + ks))
        elm.ny = elm.l * (elm.ny + 1)
        te = (ep * ep) / elm.la
        elm.l = 1/(1+(1+elm.ro) * (np.log(1+ks) + ((((elm.ny+1)*te)/(1+ks+te))- 1) * (ks/(1+ks))))

    def __sig_activation_function(self, X, elm):
        V = mm(X, elm.input_weight.transpose())
        V = V + elm.bias # FIXME: A quantidade de bias pode ser maior, precisando pegar somente a quantidade necessaria do bias
        H = 1 / (1 + np.exp(-V))
        return H

    def __scaleZMean1Var(self, X, y):
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0, ddof=1) + 0.00001 # If std == 0 add 0.0001
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0, ddof=1) + 0.00001  # If std == 0 add 0.0001
        x_scale = self.__scale(X, x_mean, x_std)
        y_scale = self.__scale(y, y_mean, y_std)

        return {
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std,
            'x_scale': x_scale,
            'y_scale': y_scale
        }

    def __update_mean_std(self, X, y):
        s = self.__sw

        oldmeanx = self.__x_mean
        oldmeany = self.__y_mean
        self.__x_mean = s / (s + 1) * self.__x_mean + (1 / (s + 1)) * X
        self.__y_mean = s / (s + 1) * self.__y_mean + (1 / (s + 1)) * y
        self.__var_x = ((s - 1) / s) * self.__var_x + (1 / s) * np.power(X - self.__x_mean, 2) + np.power(self.__x_mean - oldmeanx, 2)
        self.__var_y = ((s - 1) / s) * self.__var_y + (1 / s) * np.power(y - self.__y_mean, 2) + np.power(self.__y_mean - oldmeany, 2)
        self.__x_std = np.sqrt(self.__var_x)
        self.__y_std = np.sqrt(self.__var_y)
        self.__sw += 1
        X = self.__scale(X, self.__x_mean, self.__x_std)
        y = self.__scale(y, self.__y_mean, self.__y_std)

        return (X, y)

    def __scale(self, input, mean, std):
        return np.divide(np.subtract(input, mean), std)

    def __rescale(self, input, mean, std):
        return np.matmul(input, std) + mean

class ELM:
    def __init__(self, number_hidden_neurons_layer, input_weight, bias):
        self.number_hidden_neurons_layer = number_hidden_neurons_layer
        self.input_weight = input_weight
        self.bias = bias
        self.covariance_matrix = None
        self.beta = None

        self.l = 1;
        self.la = 0.001;
        self.ny = 0.000001;
        self.ro = 0.99;


class TemplateClassifier(BaseEstimator, ClassifierMixin):
    """ An example classifier which implements a 1-NN algorithm.

    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]
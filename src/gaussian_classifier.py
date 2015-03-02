import time
import numpy as np
from scipy.stats import multivariate_normal

class GaussianClassifier:

    def __init__(self, alpha = 0.001):
        self._trained = False
        self.alpha = alpha

    def fit(self, data, labels):
        start = time.time()
        self.klasses = np.unique(labels)
        self._find_prior_distributions(labels)
        self._find_post_distributions(data, labels)
        self._trained = True
        last = round(time.time() - start, 2)
        print "Finish training with data size {0} , takes {1} seconds".format(labels.shape[0], last)

    def predicate(self, data, use_log_pdf=True):
        self._check_if_trained()
        results = np.empty((data.shape[0],))
        for i in xrange(data.shape[0]):
            results[i] = self.predicate_single(data[i], use_log_pdf)
        return results

    def predicate_single(self, datum, use_log_pdf=True):
        self._check_if_trained()
        best_prop = float('-inf')
        best_klass = None
        for klass in self.klasses:
            if use_log_pdf:
                post_prop = self.post_distributions[klass].logpdf(datum)
            else:
                post_prop = self.post_distributions[klass].pdf(datum)
            prop = post_prop * self.prior_distributions[klass]
            if prop > best_prop:
                best_klass = klass
                best_prop = prop
        return best_klass

    def correct_rate(self, test_data, test_labels, use_log_pdf=True):
        self._check_if_trained()
        results = self.predicate(test_data, use_log_pdf)
        correct_count = np.count_nonzero(results == test_labels)
        return round(correct_count / float(test_labels.shape[0]), 2)

    def _find_prior_distributions(self, labels):
        self.prior_distributions = {}
        total_size = float(labels.shape[0])
        for klass in self.klasses:
            klass_size = labels[labels == klass].size
            self.prior_distributions[klass] = klass_size / total_size

    def _find_post_distributions(self, data, labels):
        self.means = {}
        self.cov_matrices = {}

        for klass in self.klasses:
            klass_data = data[labels == klass]
            self.means[klass] = np.mean(klass_data, axis=0)
            self.cov_matrices[klass] = np.cov(klass_data, rowvar=0)

        self.post_distributions = {}
        for klass in self.klasses:
            if self._is_invertible(self.cov_matrices[klass]):
                adjusted_cov = self.cov_matrices[klass]
            else:
                adjusted_cov = self.cov_matrices[klass] + \
                                self.alpha * np.identity(self.cov_matrices[klass].shape[0])
            self.post_distributions[klass] = multivariate_normal(self.means[klass], adjusted_cov)

    def _is_invertible(self, m):
        return m.shape[0] == m.shape[1] and np.linalg.matrix_rank(m) == m.shape[0]

    def _check_if_trained(self):
        if not self._trained:
            raise StandardError('Classifier has not been trained.')


class SameCovGaussianClassifier(GaussianClassifier):
    def _find_post_distributions(self, data, labels):
        self.means = {}
        self.cov_matrix = np.zeros((data[0].size, data[0].size))

        for klass in self.klasses:
            klass_data = data[labels == klass]
            self.means[klass] = np.mean(klass_data, axis=0)
            self.cov_matrix += np.cov(klass_data, rowvar=0)

        self.cov_matrix /= len(self.klasses)
        if self._is_invertible(self.cov_matrix):
            adjusted_cov = self.cov_matrix
        else:
            adjusted_cov = self.cov_matrix + \
                                self.alpha * np.identity(self.cov_matrix.shape[0])

        self.post_distributions = {}
        for klass in self.klasses:
            self.post_distributions[klass] = multivariate_normal(self.means[klass], adjusted_cov)

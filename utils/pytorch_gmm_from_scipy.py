import torch
import numpy as np

from math import pi
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
import pickle


class GaussianMixturePytorchFromScipy:
    """
    Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components). Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
    The model then extends them to (n, 1, d).
    The model parametrization (mu, sigma) is stored as (1, k, d), and probabilities are shaped (n, k, 1) if they relate to an individual sample, or (1, k, 1) if they assign membership probabilities to one of the mixture components.
    """
    
    def __init__(self, scipy_model_path):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.mu, self.precisions_cholesky, self.covariances, self.covariance_type = \
            self.load_from_scipy_model(scipy_model_path)

    
    def load_from_scipy_model(self, scipy_model_path):
        with open(scipy_model_path, 'rb') as f:
            gmm: GaussianMixture = pickle.load(f)
        means = torch.from_numpy(gmm.means_).float().to(self.device)
        precisions_cholesky = torch.from_numpy(gmm.precisions_cholesky_).float().to(self.device)
        covariances = torch.from_numpy(gmm.covariances_).float().to(self.device)
        cov_type = gmm.covariance_type
        return means, precisions_cholesky, covariances, cov_type
    
    def _estimate_log_prob(self, X):
        return self._estimate_log_gaussian_prob(
            X, self.mu, self.precisions_cholesky)
    
    def score_samples(self, X):
        X = X.to(self.device)
        return torch.logsumexp(self._estimate_log_prob(X), dim=1)
    
    def _compute_log_det_cholesky(self, matrix_chol, n_features):
        """
        Compute the log-det of the cholesky decomposition of matrices.
        """
        if self.covariance_type == 'full':
            n_components, _, _ = matrix_chol.shape
            log_det_chol = (torch.sum(torch.log(matrix_chol.view(n_components, -1)[:, ::n_features + 1]), 1))
        elif self.covariance_type == 'diag':
            log_det_chol = (torch.sum(torch.log(matrix_chol), axis=1))
        
        return log_det_chol
    
    def _estimate_log_gaussian_prob(self, X, means, precisions_chol):
        n_samples, n_features = X.shape
        n_components, _ = means.shape
        # det(precision_chol) is half of det(precision)
        log_det = self._compute_log_det_cholesky(
            precisions_chol, n_features)
        
        if self.covariance_type == 'full':
            log_prob = torch.empty((n_samples, n_components)).to(self.device)
            for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
                y = torch.matmul(X, prec_chol) - torch.matmul(mu, prec_chol)
                log_prob[:, k] = torch.sum(torch.square(y), axis=1)
        elif self.covariance_type == 'diag':
            precisions = precisions_chol ** 2
            log_prob = (torch.sum((means ** 2 * precisions), 1) -
                        2. * torch.matmul(X, (means * precisions).T) +
                        torch.matmul(X ** 2, precisions.T))
        else:
            raise Exception('wrong covariance type!')
        
        return -.5 * (n_features * torch.log(torch.Tensor([2 * np.pi])).to(self.device) + log_prob) + log_det


if __name__ == '__main__':
    gmm = GaussianMixturePytorchFromScipy('../AMASSDataConverter/model_global_ncomp_1_nframe_5_slidewindow_True_covtype_diag.pkl')
    X_pose = torch.randn(5, 5, 15, 3)
    X = X_pose.view(-1, 225)
    res = gmm.score_samples(X)
    conv = gmm.covariances.detach().cpu().numpy()
    print(res)

    with open('../AMASSDataConverter/model_global_ncomp_1_nframe_5_slidewindow_True_covtype_diag.pkl', 'rb') as f:
        gmm_scipy: GaussianMixture = pickle.load(f)
    X = X.cpu().numpy()
    print(gmm_scipy.score_samples(X))

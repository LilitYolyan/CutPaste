import torch
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution

# Code from https://discuss.pytorch.org/t/kernel-density-estimation-as-loss-function/62261/8
# class GaussianKDE(Distribution):
#     def __init__(self, X, bw):
#         """
#         X : tensor (n, d)
#           `n` points with `d` dimensions to which KDE will be fit
#         bw : numeric
#           bandwidth for Gaussian kernel
#         """
#         self.X = X
#         self.bw = bw
#         self.dims = X.shape[-1]
#         self.n = X.shape[0]
#         self.mvn = MultivariateNormal(loc=torch.zeros(self.dims),
#                                       covariance_matrix=torch.eye(self.dims))
#
#     def sample(self, num_samples):
#         idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
#         norm = Normal(loc=self.X[idxs], scale=self.bw)
#         return norm.sample()
#
#     def score_samples(self, Y, X=None):
#         """Returns the kernel density estimates of each point in `Y`.
#
#         Parameters
#         ----------
#         Y : tensor (m, d)
#           `m` points with `d` dimensions for which the probability density will
#           be calculated
#         X : tensor (n, d), optional
#           `n` points with `d` dimensions to which KDE will be fit. Provided to
#           allow batch calculations in `log_prob`. By default, `X` is None and
#           all points used to initialize KernelDensityEstimator are included.
#
#
#         Returns
#         -------
#         log_probs : tensor (m)
#           log probability densities for each of the queried points in `Y`
#         """
#         if X == None:
#             X = self.X
#         log_probs = torch.log(
#             (self.bw**(-self.dims) *
#              torch.exp(self.mvn.log_prob(
#                  (X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n)
#         return log_probs
#
#     def log_prob(self, Y):
#         """Returns the total log probability of one or more points, `Y`, using
#         a Multivariate Normal kernel fit to `X` and scaled using `bw`.
#
#         Parameters
#         ----------
#         Y : tensor (m, d)
#           `m` points with `d` dimensions for which the probability density will
#           be calculated
#
#         Returns
#         -------
#         log_prob : numeric
#           total log probability density for the queried points, `Y`
#         """
#
#         X_chunks = self.X.split(1000)
#         Y_chunks = Y.split(1000)
#
#         log_prob = 0
#
#         for x in X_chunks:
#             for y in Y_chunks:
#                 log_prob += self.score_samples(y, x).sum(dim=0)
#
#         return log_prob

train = torch.rand(100, 200)
test = torch.rand(1, 200)

# GDE = GaussianKDE(train, 1)
# print(GDE.score_samples(test))

from sklearn.neighbors import KernelDensity
import numpy as np

kde = KernelDensity(kernel='gaussian').fit(train.numpy())
scores = kde.score_samples(test)
#norm = np.linalg.norm(-scores)
print(scores)


#####################
# import abc
#
# import numpy as np
# import torch
# from torch import nn
#
#
# class GenerativeModel(abc.ABC, nn.Module):
#     """Base class inherited by all generative models in pytorch-generative.
#     Provides:
#         * An abstract `sample()` method which is implemented by subclasses that support
#           generating samples.
#         * Variables `self._c, self._h, self._w` which store the shape of the (first)
#           image Tensor the model was trained with. Note that `forward()` must have been
#           called at least once and the input must be an image for these variables to be
#           available.
#         * A `device` property which returns the device of the model's parameters.
#     """
#
#     def __call__(self, *args, **kwargs):
#         if getattr(self, "_c", None) is None and len(args[0].shape) == 4:
#             _, self._c, self._h, self._w = args[0].shape
#         return super().__call__(*args, **kwargs)
#
#     @property
#     def device(self):
#         return next(self.parameters()).device
#
#     @abc.abstractmethod
#     def sample(self, n_samples):
#         ...
#
#
# class Kernel(abc.ABC, nn.Module):
#     """Base class which defines the interface for all kernels."""
#
#     def __init__(self, bandwidth=0.05):
#         """Initializes a new Kernel.
#         Args:
#             bandwidth: The kernel's (band)width.
#         """
#         super().__init__()
#         self.bandwidth = bandwidth
#
#     def _diffs(self, test_Xs, train_Xs):
#         """Computes difference between each x in test_Xs with all train_Xs."""
#         test_Xs = test_Xs.view(test_Xs.shape[0], 1, *test_Xs.shape[1:])
#         train_Xs = train_Xs.view(1, train_Xs.shape[0], *train_Xs.shape[1:])
#         return torch.cdist(train_Xs, test_Xs)
#
#     @abc.abstractmethod
#     def forward(self, test_Xs, train_Xs):
#         """Computes p(x) for each x in test_Xs given train_Xs."""
#
#     @abc.abstractmethod
#     def sample(self, train_Xs):
#         """Generates samples from the kernel distribution."""
#
#
#
# class GaussianKernel(Kernel):
#     """Implementation of the Gaussian kernel."""
#
#     def forward(self, test_Xs, train_Xs):
#         diffs = self._diffs(test_Xs, train_Xs)
#
#         dims = tuple(range(len(diffs.shape))[2:])
#
#         var = self.bandwidth ** 2
#
#         # exp = torch.exp(-torch.linalg.norm(diffs,  dim=dims) ** 2 / (2 * var))
#         # print(exp.shape)
#         exp = -torch.linalg.norm(diffs,  dim=dims) ** 2 / (2 * var)
#         print('here', exp)
#         #print(exp)
#         return exp.mean(dim =1 )
#         # coef = 1 / torch.sqrt(torch.tensor(2 * np.pi * var))
#         # return (coef * exp).mean(dim=1)
#
#     def sample(self, train_Xs):
#         device = train_Xs.device
#         noise = self.bandwidth #torch.randn(train_Xs.shape) * self.bandwidth
#         return train_Xs + noise
#
#
# class KernelDensityEstimator(GenerativeModel):
#     """The KernelDensityEstimator model."""
#
#     def __init__(self, train_Xs, kernel=None):
#         """Initializes a new KernelDensityEstimator.
#         Args:
#             train_Xs: The "training" data to use when estimating probabilities.
#             kernel: The kernel to place on each of the train_Xs.
#         """
#         super().__init__()
#         self.kernel = kernel or GaussianKernel()
#         self.train_Xs = train_Xs
#
#     @property
#     def device(self):
#         return self.train_Xs.device
#
#     # TODO(eugenhotaj): This method consumes O(train_Xs * x) memory. Implement an
#     # iterative version instead.
#     def forward(self, x):
#         return self.kernel(x, self.train_Xs)
#
#     def sample(self, n_samples):
#         idxs = np.random.choice(range(len(self.train_Xs)), size=n_samples)
#         return self.kernel.sample(self.train_Xs[idxs])
#
# kde = KernelDensityEstimator(train)
# print(kde.forward(test))







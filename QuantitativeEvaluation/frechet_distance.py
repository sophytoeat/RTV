from warnings import warn

import numpy as np
from scipy import linalg
from func_timeout import func_set_timeout
from func_timeout.exceptions import FunctionTimedOut


@func_set_timeout(60)
def sqrtm(*args, **kwargs):
    """Computes the square root of a matrix with a timeout of 60 seconds to avoid hanging."""
    return linalg.sqrtm(*args, **kwargs)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, atol=1e-3):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland. Timeout version by Ryan Szeto.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    try:
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    except FunctionTimedOut:
        raise RuntimeError('sqrtm timed out')

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=atol):
            m = np.max(np.abs(covmean.imag))
            warn('Maximum imaginary component {} is greater than tolerance {}'.format(m, atol))
        covmean = covmean.real
        warn('Difference between covmean*covmean and sigma1*sigma2: {}'.format(
            np.max(np.abs(covmean.dot(covmean), sigma1.dot(sigma2)))
        ))

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
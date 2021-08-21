import numpy as np
from scipy.stats import median_abs_deviation


def clip_quantile(feat, q=(0.01, 0.95)):
    qmin, qmax = (np.quantile(feat, q=q[0]),
                  np.quantile(feat, q=q[1]))
    return np.clip(feat, qmin, qmax)


def quantile_zscore(feat, q=(0.01, 0.95), std=None):
    if q is not None:
        feat = clip_quantile(feat, q)

    std = np.std(feat) if std is None else 1
    feat = (feat - np.mean(feat)) / std
    return feat


def quantile_robust_zscore(feat, q=(0.01, 0.95), mad=1):
    if q is not None:
        feat = clip_quantile(feat, q)

    mad = median_abs_deviation(feat) if mad is None else 1
    feat = (feat - np.median(feat)) / mad
    return feat


def quantile_norm(feat, data_range=(0, 1), q=(0.01, 0.95)):
    if q is not None:
        feat = clip_quantile(feat, q)
    feat = (feat - np.min(feat)) / (np.max(feat) - np.min(feat))
    feat = feat * (data_range[1] - data_range[0]) + data_range[0]
    return feat


def feat_to_bg_onehot(feat, max_channel=None, extreme=(0, 1)):
    # scale data between 0-max
    feat = feat - np.min(feat)

    # clip value larger than max channel number
    if max_channel is not None:
        feat[feat >= max_channel] = max_channel - 1
    else:
        max_channel = np.max(feat) + 1

    # create onehot encoding
    feat_onehot = np.zeros((feat.shape[0], max_channel))
    feat_onehot += extreme[0]
    min_feat = np.min(feat)

    for i, f in enumerate(feat):
        feat_onehot[i, f - min_feat] = extreme[1]

    return feat_onehot

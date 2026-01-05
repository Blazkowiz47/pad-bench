from typing import Any, List, Tuple, Union

import numpy as np


def calculate_eer(
    genuine: List[Union[float, int]],
    imposter: List[Union[float, int]],
    bins: int = 10_001,
    reverse: bool = False,
) -> Tuple[float, Any, Any, Any]:
    """

    Calculates Equal Error Rate (eer).

    Remember: Genuine scores provided must be greater in value compared to

    imposter.

    Can be used to calculate D-EER, by replacing imposter scores to morph

    scores.

    Parameters

    ----------------------------------------------------------------------

    genuine : List[float] | NDArray

        The list of genuine scores.

    imposter : List[float] | NDArray

        The list of imposter scores.

    Returns

    ----------------------------------------------------------------------

    eer : float

        Equal Error Rate (eer) calculated from given genuine and imposter

        scores.
    far: array
    frr: array
    thresholds: array

    Example

    ----------------------------------------------------------------------

    import common_metrics

    genuine_scores = ... # genuine is a 1D numpy array or List of float

    imposter_scores = ... # imposter is a 1D numpy array or List of float

    eer = common_metrics.eer(

        genuine_scores,

        imposter_scores,

        bins=10_001,

    )

    ----------------------------------------------------------------------

    """

    genuine = np.squeeze(np.array(genuine))

    imposter = np.squeeze(np.array(imposter))

    far = np.ones(bins)

    frr = np.ones(bins)

    mi = min(np.min(imposter), np.min(genuine))

    mx = max(np.max(genuine), np.max(imposter))

    thresholds = np.linspace(mi, mx, bins)

    for id, threshold in enumerate(thresholds):
        if reverse:
            fr = np.where(genuine >= threshold)[0].shape[0]
            fa = np.where(imposter <= threshold)[0].shape[0]
        else:
            fr = np.where(genuine <= threshold)[0].shape[0]
            fa = np.where(imposter >= threshold)[0].shape[0]

        frr[id] = fr * 100 / genuine.shape[0]

        far[id] = fa * 100 / imposter.shape[0]

    di = np.argmin(np.abs(far - frr))

    eer = (far[di] + frr[di]) / 2

    return eer, far, frr, thresholds

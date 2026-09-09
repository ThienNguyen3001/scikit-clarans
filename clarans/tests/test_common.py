from sklearn.utils.estimator_checks import check_estimator

from clarans import CLARANS, FastCLARANS


def test_clarans_estimator():
    check_estimator(CLARANS(n_clusters=2))


def test_fast_clarans_estimator():
    check_estimator(FastCLARANS(n_clusters=2))

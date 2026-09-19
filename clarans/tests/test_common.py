from sklearn.utils.estimator_checks import parametrize_with_checks

from clarans import CLARANS, FastCLARANS


@parametrize_with_checks([CLARANS(n_clusters=2), FastCLARANS(n_clusters=2)])
def test_all_estimators(estimator, check):
    check(estimator)

from sklearn.utils.estimator_checks import check_estimator

from clarans import CLARANS


def test_all_estimators():
    check_estimator(CLARANS(n_clusters=2))


def test_dummy_fail():
    assert 1 == 1


from src.data import split_dataset


def test_split_has_rows_and_matching_lengths():
    X_train, X_test, y_train, y_test = split_dataset()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert list(X_train.columns) == list(X_test.columns)

import numpy as np
from sklearn.datasets import make_blobs
from clarans import CLARANS
import sys


def test_clarans_transform():
    print("=== Testing CLARANS transform method ===")

    # 1. Generate sample data
    print("1. Generating sample data...")
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    # 2. Train CLARANS
    n_clusters = 3
    print(f"2. Training CLARANS (n_clusters={n_clusters})...")
    clarans = CLARANS(n_clusters=n_clusters, random_state=42)
    clarans.fit(X)

    # 3. Test transform
    print("3. Calling transform(X)...")
    try:
        X_trans = clarans.transform(X)
    except AttributeError:
        print("ERROR: 'transform' method not found.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR executing transform: {e}")
        sys.exit(1)

    # 4. Check shape
    print(f"   -> Output shape: {X_trans.shape}")
    if X_trans.shape == (100, n_clusters):
        print("   [OK] Shape is correct.")
    else:
        print(f"   [FAIL] Shape mismatch. Expected (100, {n_clusters}), got {X_trans.shape}")

    # 5. Consistency check with predict
    print("4. Verifying consistency with predict()...")
    labels_predict = clarans.predict(X)
    labels_from_transform = np.argmin(X_trans, axis=1)

    if np.array_equal(labels_predict, labels_from_transform):
        print("   [OK] transform results match predict results.")
    else:
        mismatch_count = np.sum(labels_predict != labels_from_transform)
        print(f"   [FAIL] Mismatch found in {mismatch_count} points!")

    # 6. Test with unseen data
    print("5. Testing with unseen data...")
    X_new, _ = make_blobs(n_samples=5, centers=3, n_features=2, random_state=99)
    X_new_trans = clarans.transform(X_new)
    print(f"   -> Unseen data output shape: {X_new_trans.shape}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_clarans_transform()

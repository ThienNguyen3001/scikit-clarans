Algorithm comparison
====================

.. figure:: /_static/comparison_clustering.png
   :alt: Comparison CLARANS vs FastCLARANS vs KMeans
   :figwidth: 95%
   :align: center

   Side-by-side comparison of CLARANS, FastCLARANS and scikit-learn's
   `KMeans` on the same dataset.

.. _code:

.. code-block:: python
   :linenos:

   """Generate `comparison_clustering.png`.

   Side-by-side comparison of CLARANS, FastCLARANS and scikit-learn's KMeans.
   """
   from pathlib import Path
   import time
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_blobs
   from clarans import CLARANS, FastCLARANS
   from sklearn.cluster import KMeans


   def main():
       X, _ = make_blobs(n_samples=500, centers=3, random_state=42)
       models = [
           ("CLARANS", CLARANS(n_clusters=3, numlocal=5, random_state=42)),
           ("FastCLARANS", FastCLARANS(n_clusters=3, numlocal=5, random_state=42)),
           ("KMeans", KMeans(n_clusters=3, random_state=42)),
       ]

       fig, axes = plt.subplots(1, 3, figsize=(15, 4))

       for ax, (name, model) in zip(axes, models):
           t0 = time.perf_counter()
           model.fit(X)
           t1 = time.perf_counter()
           labels = model.labels_
           centers = getattr(model, "cluster_centers_", None)
           ax.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap="tab10", alpha=0.8)
           if centers is not None:
               ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="x", s=100, linewidths=2)
           ax.set_title(f"{name}\n{(t1-t0):.3f}s")
           ax.set_xticks([])
           ax.set_yticks([])

       out = "comparison_clustering.png"
       fig.savefig(out, bbox_inches="tight", dpi=150)
       print(f"Saved {out}")


   if __name__ == "__main__":
       main()

Robustness to Outliers
======================

One of the fundamental advantages of **k-medoids** algorithms like **CLARANS**
over **k-means** is their robustness to extreme noise and outliers.

Why k-medoids is robust
-----------------------

In standard **k-means**, each cluster center is defined as the arithmetic mean (centroid):

.. math::

   \mu_j = \frac{1}{|C_j|} \sum_{x \in C_j} x

Because k-means minimizes squared Euclidean distances, a single extreme outlier exerts
a strong pulling force on the centroid, dragging it away from the true cluster core
and distorting cluster boundaries.

In **CLARANS (k-medoids)**, each cluster center is constrained to be an actual sample
from the dataset (a *medoid*):

.. math::

   m_j = \arg\min_{m \in C_j} \sum_{x \in C_j} d(x, m)

Because medoids are actual observed data points and the objective function minimizes
unsquared pairwise distances, extreme outliers have minimal influence on the optimal
medoid location. The medoid remains safely anchored at the center of the dense point cloud.

.. figure:: /_static/outlier_robustness.png
   :alt: Robustness to Outliers: CLARANS vs KMeans
   :figwidth: 95%
   :align: center

   Comparison between K-Means and CLARANS on synthetic clusters contaminated with
   extreme outlier points. While the K-Means centroid is dragged towards the noise,
   the CLARANS medoid remains firmly located in the dense cluster core.

.. _gallery-outliers-code:

.. code-block:: python
   :linenos:

   """Generate `outlier_robustness.png`.

   Demonstrates CLARANS robustness to extreme outliers compared to KMeans.
   """
   import numpy as np
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.cluster import KMeans
   from clarans import CLARANS

   COLORS = ["#2b5c8f", "#d95f02"]


   def main():
       plt.style.use("default")
       np.random.seed(42)

       # Two dense Gaussian clusters
       c1 = np.random.normal(loc=[-3, 0], scale=0.6, size=(150, 2))
       c2 = np.random.normal(loc=[3, 0], scale=0.6, size=(150, 2))
       # Extreme outliers placed far in the upper-right corner
       outliers = np.random.uniform(low=[8, 6], high=[14, 10], size=(12, 2))
       X = np.vstack([c1, c2, outliers])

       km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
       cl = CLARANS(n_clusters=2, num_local=5, random_state=42).fit(X)

       fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=200)

       # (a) K-Means
       for k in range(2):
           mask = km.labels_[:-12] == k
           axes[0].scatter(X[:-12][mask, 0], X[:-12][mask, 1], c=COLORS[k], s=18, alpha=0.65, edgecolors="none")
       axes[0].scatter(X[-12:, 0], X[-12:, 1], c="#7f8c8d", marker="^", s=35, label="Outliers", zorder=3)
       axes[0].scatter(
           km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           marker="+", s=110, c="black", linewidths=2.0, label="Centroid", zorder=5
       )
       axes[0].set_title("(a) K-Means (Centroid displaced by outliers)", fontsize=10.5, pad=8)
       axes[0].set_xticks([])
       axes[0].set_yticks([])
       for spine in axes[0].spines.values():
           spine.set_color("#cccccc")
           spine.set_linewidth(0.8)
       axes[0].legend(loc="lower right", frameon=False, fontsize=9)

       # (b) CLARANS
       for k in range(2):
           mask = cl.labels_[:-12] == k
           axes[1].scatter(X[:-12][mask, 0], X[:-12][mask, 1], c=COLORS[k], s=18, alpha=0.65, edgecolors="none")
       axes[1].scatter(X[-12:, 0], X[-12:, 1], c="#7f8c8d", marker="^", s=35, label="Outliers", zorder=3)
       axes[1].scatter(
           cl.cluster_centers_[:, 0], cl.cluster_centers_[:, 1],
           marker="x", s=90, c="black", linewidths=1.8, label="Medoid", zorder=5
       )
       axes[1].set_title("(b) CLARANS (Medoid robust in dense core)", fontsize=10.5, pad=8)
       axes[1].set_xticks([])
       axes[1].set_yticks([])
       for spine in axes[1].spines.values():
           spine.set_color("#cccccc")
           spine.set_linewidth(0.8)
       axes[1].legend(loc="lower right", frameon=False, fontsize=9)

       plt.tight_layout()
       out = "outlier_robustness.png"
       fig.savefig(out, dpi=200)
       print(f"Saved {out}")


   if __name__ == "__main__":
       main()


2D clustering examples
======================

.. figure:: /_static/gallery_clusters_blobs.png
   :alt: CLARANS on blobs
   :figwidth: 80%
   :align: center

   CLARANS clustering on a synthetic blobs dataset ($k=4$).

.. _gallery-2d-code:

.. code-block:: python
   :linenos:

   """Generate `gallery_clusters_blobs.png`.

   Minimalist example demonstrating CLARANS on synthetic blobs.
   """
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_blobs
   from clarans import CLARANS

   COLORS = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]


   def main():
       plt.style.use("default")
       X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
       model = CLARANS(n_clusters=4, num_local=5, random_state=42)
       model.fit(X)

       fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=200)
       for k in range(4):
           mask = model.labels_ == k
           ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[k], s=20, alpha=0.65, edgecolors="none")

       centers = model.cluster_centers_
       ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=90, c="black", linewidths=1.8, label="Medoids")

       ax.set_title("CLARANS on Synthetic Blobs ($k=4$)", fontsize=11, pad=8)
       ax.set_xticks([])
       ax.set_yticks([])
       for spine in ax.spines.values():
           spine.set_color("#cccccc")
           spine.set_linewidth(0.8)

       ax.legend(loc="upper right", frameon=False, fontsize=9)
       plt.tight_layout()

       out = "gallery_clusters_blobs.png"
       fig.savefig(out, dpi=200)
       print(f"Saved {out}")


   if __name__ == "__main__":
       main()


.. figure:: /_static/gallery_clusters_moons.png
   :alt: CLARANS on moons
   :figwidth: 80%
   :align: center

   CLARANS on two interleaving moons ($k=2$).

.. code-block:: python
   :linenos:

   """Generate `gallery_clusters_moons.png`.

   Minimalist example for moons dataset.
   """
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_moons
   from clarans import CLARANS

   COLORS = ["#2b5c8f", "#d95f02"]


   def main():
       plt.style.use("default")
       X, _ = make_moons(n_samples=500, noise=0.06, random_state=42)
       model = CLARANS(n_clusters=2, num_local=5, random_state=42)
       model.fit(X)

       fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=200)
       for k in range(2):
           mask = model.labels_ == k
           ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[k], s=20, alpha=0.65, edgecolors="none")

       centers = model.cluster_centers_
       ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=90, c="black", linewidths=1.8, label="Medoids")

       ax.set_title("CLARANS on Interleaving Moons ($k=2$)", fontsize=11, pad=8)
       ax.set_xticks([])
       ax.set_yticks([])
       for spine in ax.spines.values():
           spine.set_color("#cccccc")
           spine.set_linewidth(0.8)

       ax.legend(loc="upper right", frameon=False, fontsize=9)
       plt.tight_layout()

       out = "gallery_clusters_moons.png"
       fig.savefig(out, dpi=200)
       print(f"Saved {out}")


   if __name__ == "__main__":
       main()


.. figure:: /_static/gallery_clusters_anisotropic.png
   :alt: CLARANS on anisotropic data
   :figwidth: 80%
   :align: center

   CLARANS on an anisotropic dataset with non-spherical clusters ($k=3$).

.. code-block:: python
   :linenos:

   """Generate `gallery_clusters_anisotropic.png`.

   Minimalist example for anisotropic dataset.
   """
   import numpy as np
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_blobs
   from clarans import CLARANS

   COLORS = ["#2b5c8f", "#d95f02", "#7570b3"]


   def main():
       plt.style.use("default")
       X, _ = make_blobs(n_samples=500, centers=3, random_state=170)
       transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
       X = X.dot(transformation)

       model = CLARANS(n_clusters=3, num_local=5, random_state=42)
       model.fit(X)

       fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=200)
       for k in range(3):
           mask = model.labels_ == k
           ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[k], s=20, alpha=0.65, edgecolors="none")

       centers = model.cluster_centers_
       ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=90, c="black", linewidths=1.8, label="Medoids")

       ax.set_title("CLARANS on Anisotropic Clusters ($k=3$)", fontsize=11, pad=8)
       ax.set_xticks([])
       ax.set_yticks([])
       for spine in ax.spines.values():
           spine.set_color("#cccccc")
           spine.set_linewidth(0.8)

       ax.legend(loc="upper left", frameon=False, fontsize=9)
       plt.tight_layout()

       out = "gallery_clusters_anisotropic.png"
       fig.savefig(out, dpi=200)
       print(f"Saved {out}")


   if __name__ == "__main__":
       main()




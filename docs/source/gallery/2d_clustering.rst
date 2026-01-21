2D clustering examples
======================

.. figure:: /_static/gallery_clusters_blobs.png
   :alt: CLARANS on blobs
   :figwidth: 75%
   :align: center

   CLARANS clustering on a synthetic blobs dataset.

.. _code:

.. code-block:: python
   :linenos:

   """Generate `gallery_clusters_blobs.png`.

   Standalone example that demonstrates CLARANS on synthetic blobs.
   """
   from pathlib import Path
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_blobs
   from clarans import CLARANS


   def main():
       X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
       model = CLARANS(n_clusters=4, numlocal=5, random_state=42)
       model.fit(X)

       fig, ax = plt.subplots(figsize=(6, 4))
       ax.scatter(X[:, 0], X[:, 1], c=model.labels_, s=20, cmap="tab10", alpha=0.8)
       ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", marker="x", s=100)
       ax.set_title("CLARANS: blobs")
       ax.set_xticks([])
       ax.set_yticks([])

       out = "gallery_clusters_blobs.png"
       fig.savefig(out, bbox_inches="tight", dpi=150)
       print(f"Saved {out}")


   if __name__ == "__main__":
       main()


.. figure:: /_static/gallery_clusters_moons.png
   :alt: CLARANS on moons
   :figwidth: 75%
   :align: center

   CLARANS on two interleaving moons.

.. code-block:: python
   :linenos:

   """Generate `gallery_clusters_moons.png`.

   Standalone example for moons dataset.
   """
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_moons
   from clarans import CLARANS


   def main():
       X, _ = make_moons(n_samples=500, noise=0.05, random_state=42)
       model = CLARANS(n_clusters=2, numlocal=5, random_state=42)
       model.fit(X)

       fig, ax = plt.subplots(figsize=(6, 4))
       ax.scatter(X[:, 0], X[:, 1], c=model.labels_, s=20, cmap="tab10", alpha=0.8)
       ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", marker="x", s=100)
       ax.set_title("CLARANS: moons")
       ax.set_xticks([])
       ax.set_yticks([])

       out = "gallery_clusters_moons.png"
       fig.savefig(out, bbox_inches="tight", dpi=150)
       print(f"Saved {out}")


   if __name__ == "__main__":
       main()


.. figure:: /_static/gallery_clusters_anisotropic.png
   :alt: CLARANS on anisotropic data
   :figwidth: 75%
   :align: center

   CLARANS on an anisotropic dataset.

.. code-block:: python
   :linenos:

   """Generate `gallery_clusters_anisotropic.png`.

   Standalone example for anisotropic dataset.
   """
   import numpy as np
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_blobs
   from clarans import CLARANS


   def main():
       X, _ = make_blobs(n_samples=500, centers=3, random_state=170)
       transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
       X = X.dot(transformation)

       model = CLARANS(n_clusters=3, numlocal=5, random_state=42)
       model.fit(X)

       fig, ax = plt.subplots(figsize=(6, 4))
       ax.scatter(X[:, 0], X[:, 1], c=model.labels_, s=20, cmap="tab10", alpha=0.8)
       ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", marker="x", s=100)
       ax.set_title("CLARANS: anisotropic")
       ax.set_xticks([])
       ax.set_yticks([])

       out = "gallery_clusters_anisotropic.png"
       fig.savefig(out, bbox_inches="tight", dpi=150)
       print(f"Saved {out}")


   if __name__ == "__main__":
       main()

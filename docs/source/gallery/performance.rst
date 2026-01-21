Performance tuning
===================

.. figure:: /_static/runtime_scaling.png
   :alt: Runtime scaling
   :figwidth: 75%
   :align: center

   Runtime scaling across dataset sizes (CLARANS, FastCLARANS, KMeans).

.. _code:

.. code-block:: python
   :linenos:

   """Generate `runtime_scaling.png` comparing runtimes for different data sizes.
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
       Ns = [200, 500, 1000, 2000]
       clarans_times = []
       fast_times = []
       kmeans_times = []

       for N in Ns:
           X, _ = make_blobs(n_samples=N, centers=4, cluster_std=0.60, random_state=42)

           t0 = time.perf_counter()
           CLARANS(n_clusters=4, numlocal=1, random_state=42).fit(X)
           clarans_times.append(time.perf_counter() - t0)

           t0 = time.perf_counter()
           FastCLARANS(n_clusters=4, numlocal=1, random_state=42).fit(X)
           fast_times.append(time.perf_counter() - t0)

           t0 = time.perf_counter()
           KMeans(n_clusters=4, random_state=42).fit(X)
           kmeans_times.append(time.perf_counter() - t0)

       fig, ax = plt.subplots(figsize=(6, 4))
       ax.plot(Ns, clarans_times, marker="o", label="CLARANS")
       ax.plot(Ns, fast_times, marker="o", label="FastCLARANS")
       ax.plot(Ns, kmeans_times, marker="o", label="KMeans")
       ax.set_xlabel("n samples")
       ax.set_ylabel("time (s)")
       ax.set_title("Runtime scaling (numlocal=1)")
       ax.legend()

       out = "runtime_scaling.png"
       fig.savefig(out, bbox_inches="tight", dpi=150)
       print(f"Saved {out}")


.. figure:: /_static/parameter_sensitivity.png
   :alt: Parameter sensitivity
   :figwidth: 80%
   :align: center

   Sensitivity of final cost / runtime to `numlocal` and `maxneighbor`.

.. code-block:: python
   :linenos:

   """Generate `parameter_sensitivity.png` showing cost/runtime for parameter grid.
   """
   from pathlib import Path
   import time
   import numpy as np
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_blobs
   from clarans import CLARANS
   from clarans.utils import calculate_cost


   def main():
       X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
       numlocals = [1, 2, 5]
       maxneighbors = [50, 200, 500]

       cost_grid = np.zeros((len(numlocals), len(maxneighbors)))
       time_grid = np.zeros_like(cost_grid)

       for i, nl in enumerate(numlocals):
           for j, mn in enumerate(maxneighbors):
               t0 = time.perf_counter()
               model = CLARANS(n_clusters=4, numlocal=nl, maxneighbor=mn, random_state=42)
               model.fit(X)
               time_grid[i, j] = time.perf_counter() - t0
               cost_grid[i, j] = calculate_cost(X, model.medoid_indices_)

       fig, axes = plt.subplots(1, 2, figsize=(12, 4))
       im0 = axes[0].imshow(cost_grid, cmap="viridis", origin="lower")
       axes[0].set_xticks(range(len(maxneighbors)))
       axes[0].set_xticklabels([str(m) for m in maxneighbors])
       axes[0].set_yticks(range(len(numlocals)))
       axes[0].set_yticklabels([str(n) for n in numlocals])
       axes[0].set_xlabel("maxneighbor")
       axes[0].set_ylabel("numlocal")
       axes[0].set_title("Final cost")
       fig.colorbar(im0, ax=axes[0])

       im1 = axes[1].imshow(time_grid, cmap="magma", origin="lower")
       axes[1].set_xticks(range(len(maxneighbors)))
       axes[1].set_xticklabels([str(m) for m in maxneighbors])
       axes[1].set_yticks(range(len(numlocals)))
       axes[1].set_yticklabels([str(n) for n in numlocals])
       axes[1].set_xlabel("maxneighbor")
       axes[1].set_ylabel("numlocal")
       axes[1].set_title("Runtime (s)")
       fig.colorbar(im1, ax=axes[1])

       out = "parameter_sensitivity.png"
       fig.savefig(out, bbox_inches="tight", dpi=150)
       print(f"Saved {out}")

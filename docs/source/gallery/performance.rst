Performance tuning
===================

.. figure:: /_static/runtime_scaling.png
   :alt: Runtime scaling
   :figwidth: 80%
   :align: center

   Runtime scaling across dataset sizes (:math:`N` from 500 to 8,000 samples)
   for CLARANS, FastCLARANS, and KMeans.

.. _gallery-performance-code:

.. code-block:: python
   :linenos:

   """Generate `runtime_scaling.png` comparing runtimes for different data sizes.
   """
   import time
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_blobs
   from clarans import CLARANS, FastCLARANS
   from sklearn.cluster import KMeans


   def main():
       plt.style.use("default")
       plt.rcParams.update({
           "axes.spines.top": False,
           "axes.spines.right": False,
           "axes.edgecolor": "#2c3e50",
       })
       Ns = [500, 1000, 2000, 4000, 8000]
       clarans_times = []
       fast_times = []
       kmeans_times = []

       for N in Ns:
           X, _ = make_blobs(n_samples=N, centers=4, cluster_std=0.60, random_state=42)

           tc, tf, tk = 0.0, 0.0, 0.0
           n_repeats = 2
           for seed in range(42, 42 + n_repeats):
               t0 = time.perf_counter()
               CLARANS(n_clusters=4, num_local=2, random_state=seed).fit(X)
               tc += time.perf_counter() - t0

               t0 = time.perf_counter()
               FastCLARANS(n_clusters=4, num_local=2, random_state=seed).fit(X)
               tf += time.perf_counter() - t0

               t0 = time.perf_counter()
               KMeans(n_clusters=4, random_state=seed, n_init=10).fit(X)
               tk += time.perf_counter() - t0

           clarans_times.append(tc / n_repeats)
           fast_times.append(tf / n_repeats)
           kmeans_times.append(tk / n_repeats)

       fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
       ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#e5e5e5")
       ax.xaxis.grid(False)

       ax.plot(Ns, clarans_times, marker="o", markersize=5, linewidth=1.6, color="#2b5c8f", label="CLARANS")
       ax.plot(Ns, fast_times, marker="s", markersize=5, linewidth=1.6, color="#1b9e77", label="FastCLARANS")
       ax.plot(Ns, kmeans_times, marker="^", markersize=5, linewidth=1.6, color="#d95f02", label="K-Means")

       ax.set_xlabel("Number of samples ($N$)")
       ax.set_ylabel("Runtime (seconds)")
       ax.set_title("Runtime Scaling vs. Dataset Size (num_local=2)", pad=10)
       ax.legend(loc="upper left", frameon=False)
       plt.tight_layout()

       out = "runtime_scaling.png"
       fig.savefig(out, dpi=200)
       print(f"Saved {out}")


   if __name__ == "__main__":
       main()

.. figure:: /_static/parameter_sensitivity.png
   :alt: Parameter sensitivity
   :figwidth: 90%
   :align: center

   Sensitivity of final cost and runtime to ``num_local`` and ``max_neighbors``,
   with values annotated directly within each grid cell.

.. code-block:: python
   :linenos:

   """Generate `parameter_sensitivity.png` showing cost/runtime for parameter grid.
   """
   import time
   import numpy as np
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_blobs
   from clarans import CLARANS
   from clarans.utils import calculate_cost


   def main():
       plt.style.use("default")
       plt.rcParams.update({
           "axes.spines.top": False,
           "axes.spines.right": False,
           "axes.edgecolor": "#2c3e50",
       })
       X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
       num_locals = [1, 2, 5]
       max_neighbors = [50, 150, 300, 500]

       cost_grid = np.zeros((len(num_locals), len(max_neighbors)))
       time_grid = np.zeros_like(cost_grid)

       for i, nl in enumerate(num_locals):
           for j, mn in enumerate(max_neighbors):
               t0 = time.perf_counter()
               model = CLARANS(n_clusters=4, num_local=nl, max_neighbors=mn, random_state=42)
               model.fit(X)
               time_grid[i, j] = time.perf_counter() - t0
               cost_grid[i, j] = calculate_cost(X, model.medoid_indices_)

       fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=200)

       # Cost Heatmap: Blues
       im0 = axes[0].imshow(cost_grid, cmap="Blues", origin="lower", aspect="auto")
       axes[0].set_xticks(range(len(max_neighbors)))
       axes[0].set_xticklabels([str(m) for m in max_neighbors])
       axes[0].set_yticks(range(len(num_locals)))
       axes[0].set_yticklabels([str(n) for n in num_locals])
       axes[0].set_xlabel("max_neighbors")
       axes[0].set_ylabel("num_local")
       axes[0].set_title("(a) Solution Cost (lower is better)", pad=8)
       cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.85)
       cbar0.ax.tick_params(labelsize=8)

       cost_min, cost_max = cost_grid.min(), cost_grid.max()
       for i in range(len(num_locals)):
           for j in range(len(max_neighbors)):
               val = cost_grid[i, j]
               norm = (val - cost_min) / (cost_max - cost_min + 1e-8)
               text_color = "white" if norm > 0.65 else "#222222"
               axes[0].text(
                   j, i, f"{val:.1f}",
                   ha="center", va="center",
                   color=text_color, fontsize=9
               )

       # Runtime Heatmap: YlOrRd
       im1 = axes[1].imshow(time_grid, cmap="YlOrRd", origin="lower", aspect="auto")
       axes[1].set_xticks(range(len(max_neighbors)))
       axes[1].set_xticklabels([str(m) for m in max_neighbors])
       axes[1].set_yticks(range(len(num_locals)))
       axes[1].set_yticklabels([str(n) for n in num_locals])
       axes[1].set_xlabel("max_neighbors")
       axes[1].set_ylabel("num_local")
       axes[1].set_title("(b) Runtime in seconds", pad=8)
       cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.85)
       cbar1.ax.tick_params(labelsize=8)

       time_min, time_max = time_grid.min(), time_grid.max()
       for i in range(len(num_locals)):
           for j in range(len(max_neighbors)):
               val = time_grid[i, j]
               norm = (val - time_min) / (time_max - time_min + 1e-8)
               text_color = "white" if norm > 0.65 else "#222222"
               axes[1].text(
                   j, i, f"{val:.2f}s",
                   ha="center", va="center",
                   color=text_color, fontsize=9
               )

       plt.tight_layout()
       out = "parameter_sensitivity.png"
       fig.savefig(out, dpi=200)
       print(f"Saved {out}")


   if __name__ == "__main__":
       main()

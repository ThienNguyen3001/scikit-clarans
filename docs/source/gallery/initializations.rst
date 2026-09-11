Initialization Strategies
=========================

The choice of initial medoids significantly influences both the convergence speed
and the final solution quality of CLARANS. `scikit-clarans` supports 4 built-in
initialization methods via the ``init`` parameter, adapted from the
`scikit-learn-extra <https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html>`_
``KMedoids`` implementation.

Supported Methods
-----------------

* ``init="random"``: Selects $k$ unique samples uniformly at random. Fast and simple,
  but can place multiple initial medoids in the same cluster.
* ``init="k-medoids++"``: Adapts the $k$-means++ seeding heuristic to $k$-medoids,
  probabilistically choosing subsequent medoids proportional to squared distance
  from the nearest already selected medoid. Guarantees well-dispersed starting points.
* ``init="heuristic"``: Computes pairwise distances and selects the $k$ points with
  the smallest aggregate distance to all other samples (centrality).
* ``init="build"``: Implements the classic PAM BUILD greedy step, iteratively selecting
  medoids that maximize the reduction in total clustering cost.

.. figure:: /_static/initialization_comparison.png
   :alt: Comparison of Initialization Strategies
   :figwidth: 95%
   :align: center

   Initial medoids selected by each strategy before CLARANS randomized search begins.
   Note how ``k-medoids++`` and ``BUILD`` pick exactly one well-placed seed per cluster,
   yielding much lower starting costs.

.. _gallery-initializations-code:

.. code-block:: python
   :linenos:

   """Generate `initialization_comparison.png`.

   Visual comparison of the 4 initial medoid seeding strategies in scikit-clarans.
   """
   import numpy as np
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_blobs
   from clarans.initialization import (
       initialize_k_medoids_plus_plus,
       initialize_heuristic,
       initialize_build,
   )
   from clarans.utils import calculate_cost


   def main():
       plt.style.use("default")
       X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.70, random_state=42)
       k = 4
       rng = np.random.default_rng(42)

       inits = [
           ("Random", rng.choice(X.shape[0], size=k, replace=False)),
           ("k-medoids++", initialize_k_medoids_plus_plus(X, k, metric="euclidean", random_state=42)),
           ("Heuristic", initialize_heuristic(X, k, metric="euclidean")),
           ("BUILD", initialize_build(X, k, metric="euclidean")),
       ]

       letters = ["(a)", "(b)", "(c)", "(d)"]
       fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0), dpi=200)
       axes = axes.flatten()

       for ax, letter, (name, medoid_idx) in zip(axes, letters, inits):
           initial_cost = calculate_cost(X, medoid_idx)
           ax.scatter(X[:, 0], X[:, 1], c="#bdc3c7", s=15, alpha=0.55, edgecolors="none")
           chosen_points = X[medoid_idx]
           ax.scatter(
               chosen_points[:, 0], chosen_points[:, 1],
               marker="D", s=55, facecolors="#c0392b", edgecolors="black", linewidths=1.0,
               label="Initial Medoid", zorder=5
           )
           ax.set_title(f"{letter} {name} (Cost: {initial_cost:.1f})", pad=6, fontsize=10)
           ax.set_xticks([])
           ax.set_yticks([])
           for spine in ax.spines.values():
               spine.set_color("#cccccc")
               spine.set_linewidth(0.8)
           ax.legend(loc="upper right", frameon=False, fontsize=8.5)

       plt.tight_layout()
       out = "initialization_comparison.png"
       fig.savefig(out, dpi=200)
       print(f"Saved {out}")


   if __name__ == "__main__":
       main()


Quality vs k
============

.. figure:: /_static/silhouette_vs_k.png
   :alt: Silhouette score vs k
   :figwidth: 80%
   :align: center

   Silhouette score versus number of clusters (:math:`k`) for CLARANS, FastCLARANS,
   and KMeans on a dataset with 4 true clusters.

.. _gallery-quality-code:

.. code-block:: python
   :linenos:

   """Generate `silhouette_vs_k.png` comparing CLARANS, FastCLARANS, and KMeans.
   """
   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_blobs
   from sklearn.cluster import KMeans
   from sklearn.metrics import silhouette_score
   from clarans import CLARANS, FastCLARANS


   def main():
       plt.style.use("default")
       plt.rcParams.update({
           "axes.spines.top": False,
           "axes.spines.right": False,
           "axes.edgecolor": "#2c3e50",
       })

       X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
       ks = list(range(2, 9))
       methods = {
           "CLARANS": lambda k: CLARANS(n_clusters=k, num_local=3, random_state=42),
           "FastCLARANS": lambda k: FastCLARANS(n_clusters=k, num_local=3, random_state=42),
           "K-Means": lambda k: KMeans(n_clusters=k, random_state=42, n_init=10),
       }

       results = {name: [] for name in methods}

       for k in ks:
           for name, factory in methods.items():
               model = factory(k)
               model.fit(X)
               labels = model.labels_
               if len(set(labels)) > 1:
                   score = silhouette_score(X, labels)
               else:
                   score = float("nan")
               results[name].append(score)

       fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
       ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#e5e5e5")
       ax.xaxis.grid(False)

       colors = {"CLARANS": "#2b5c8f", "FastCLARANS": "#1b9e77", "K-Means": "#d95f02"}
       markers = {"CLARANS": "o", "FastCLARANS": "s", "K-Means": "^"}

       for name, scores in results.items():
           ax.plot(
               ks, scores,
               marker=markers[name], markersize=5, linewidth=1.6,
               color=colors[name], label=name
           )

       ax.axvline(x=4, color="#7f8c8d", linestyle=":", linewidth=1.2, label="Optimal ($k=4$)")

       ax.set_xlabel("Number of clusters ($k$)")
       ax.set_ylabel("Silhouette score")
       ax.set_title("Cluster Quality vs. Number of Clusters ($k$)", pad=10)
       ax.set_xticks(ks)
       ax.legend(loc="upper right", frameon=False)
       plt.tight_layout()

       out = "silhouette_vs_k.png"
       fig.savefig(out, dpi=200)
       print(f"Saved {out}")


   if __name__ == "__main__":
       main()




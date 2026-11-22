# Visual ML Lab

A dependency-free interactive playground for exploring clustering, principal
component analysis (PCA), and a nearest-centroid classifier on synthetic data.
Open `index.html` in a modern browser or serve the folder with
`python -m http.server`.

To publish on GitHub Pages, choose **Settings → Pages → Deploy from a branch**,
then select `main` and `/ (root)`.

- **PCA:** project three features into two principal components. PCA is fit on
  the training split, and explained-variance percentages describe that split.
- **Clustering:** run deterministic k-means on the projected data. Cluster
  colors are arbitrary and do not represent ground-truth labels.
- **Classification:** train a nearest-centroid classifier on the projected
  training split and display accuracy on the held-out split. Shading shows its
  decision regions in the displayed PCA plane.

Change the seed and noise to explore how the results shift. This educational
demo uses generated data and is not a general-purpose modeling or validation
tool. Run the core checks with `node --test lab.test.js`.

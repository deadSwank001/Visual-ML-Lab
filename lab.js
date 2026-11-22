(function (root) {
  "use strict";
  function generate(seed, noise, count = 120) {
    let state = seed >>> 0;
    const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
    const normal = () => Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12))) * Math.cos(2 * Math.PI * random());
    return Array.from({ length: count }, (_, i) => {
      const label = i % 2, sign = label ? 1 : -1;
      return { values: [sign * 1.3 + noise * normal(), sign * .7 + noise * normal(),
        sign * .4 + noise * normal()], label, test: i % 10 < 2 };
    });
  }
  function fitPCA(samples) {
    if (samples.length < 2 || samples.some(s => s.values.length !== 3 || s.values.some(v => !Number.isFinite(v)))) {
      throw new Error("PCA requires at least two finite three-feature samples.");
    }
    const mean = [0, 1, 2].map(j => samples.reduce((sum, s) => sum + s.values[j], 0) / samples.length);
    const cov = [0, 1, 2].map(i => [0, 1, 2].map(j =>
      samples.reduce((sum, s) => sum + (s.values[i] - mean[i]) * (s.values[j] - mean[j]), 0) / (samples.length - 1)));
    const vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (let iteration = 0; iteration < 50; iteration++) {
      let p = 0, q = 1;
      for (const [i, j] of [[0, 1], [0, 2], [1, 2]]) if (Math.abs(cov[i][j]) > Math.abs(cov[p][q])) [p, q] = [i, j];
      if (Math.abs(cov[p][q]) < 1e-12) break;
      const angle = .5 * Math.atan2(2 * cov[p][q], cov[q][q] - cov[p][p]);
      const c = Math.cos(angle), s = Math.sin(angle);
      const app = cov[p][p], aqq = cov[q][q], apq = cov[p][q];
      cov[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
      cov[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
      cov[p][q] = cov[q][p] = 0;
      for (let k = 0; k < 3; k++) {
        if (k !== p && k !== q) {
          const kp = cov[k][p], kq = cov[k][q];
          cov[k][p] = cov[p][k] = c * kp - s * kq;
          cov[k][q] = cov[q][k] = s * kp + c * kq;
        }
        const vp = vectors[k][p], vq = vectors[k][q];
        vectors[k][p] = c * vp - s * vq;
        vectors[k][q] = s * vp + c * vq;
      }
    }
    const order = [0, 1, 2].sort((a, b) => cov[b][b] - cov[a][a]);
    const eigenvalues = order.map(i => Math.max(0, cov[i][i]));
    const total = eigenvalues.reduce((a, b) => a + b, 0);
    return { mean, axes: order.slice(0, 2).map(i => vectors.map(row => row[i])),
      explained: eigenvalues.slice(0, 2).map(value => total ? value / total : 0) };
  }
  function project(pca, values) {
    return pca.axes.map(axis => axis.reduce((sum, v, i) => sum + v * (values[i] - pca.mean[i]), 0));
  }
  const distance = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  function kmeans(points, k) {
    if (!points.length || !Number.isInteger(k) || k < 1) throw new Error("Invalid k-means input.");
    const distinct = [...new Map(points.map(point => [point.join(","), point])).values()];
    const centers = [distinct[0].slice()];
    while (centers.length < Math.min(k, distinct.length)) {
      const next = distinct.reduce((best, point) => {
        const d = Math.min(...centers.map(center => distance(point, center)));
        return d > best.d ? { point, d } : best;
      }, { point: null, d: -1 });
      centers.push(next.point.slice());
    }
    let labels = [];
    for (let iteration = 0; iteration < 100; iteration++) {
      const next = points.map(point => centers.reduce((best, center, i) =>
        distance(point, center) < best.d ? { i, d: distance(point, center) } : best, { i: 0, d: Infinity }).i);
      if (labels.length === next.length && next.every((value, i) => value === labels[i])) break;
      labels = next;
      centers.forEach((_, i) => {
        const members = points.filter((point, index) => labels[index] === i);
        if (members.length) centers[i] = [0, 1].map(axis => members.reduce((sum, point) => sum + point[axis], 0) / members.length);
      });
    }
    return { labels, centers };
  }
  function fitClassifier(points) {
    return [0, 1].map(label => {
      const group = points.filter(point => point.label === label);
      if (!group.length) throw new Error("Training split must contain both classes.");
      return [0, 1].map(axis => group.reduce((sum, point) => sum + point.xy[axis], 0) / group.length);
    });
  }
  function predict(centers, xy) { return distance(xy, centers[0]) <= distance(xy, centers[1]) ? 0 : 1; }
  const api = { generate, fitPCA, project, kmeans, fitClassifier, predict };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Lab = api;
})(typeof globalThis !== "undefined" ? globalThis : this);

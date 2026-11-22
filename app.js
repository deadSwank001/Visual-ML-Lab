"use strict";
const colors = ["#8be2d3", "#ffae85", "#bdacff", "#ff8ab4"];
const plot = document.getElementById("plot");
let samples = [];
function node(tag, attrs, text) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  if (text !== undefined) el.textContent = text;
  plot.appendChild(el);
  return el;
}
function redraw() {
  const mode = document.getElementById("mode").value;
  document.getElementById("kLabel").hidden = mode !== "cluster";
  const train = samples.filter(s => !s.test);
  const pca = Lab.fitPCA(train);
  const points = samples.map(s => ({ ...s, xy: Lab.project(pca, s.values) }));
  const xmin = Math.min(...points.map(p => p.xy[0])) - .5, xmax = Math.max(...points.map(p => p.xy[0])) + .5;
  const ymin = Math.min(...points.map(p => p.xy[1])) - .5, ymax = Math.max(...points.map(p => p.xy[1])) + .5;
  const x = value => 60 + (value - xmin) / (xmax - xmin) * 650;
  const y = value => 450 - (value - ymin) / (ymax - ymin) * 410;
  const title = document.getElementById("title"), metric = document.getElementById("metric");
  const explanation = document.getElementById("explanation"), legend = document.getElementById("legend");
  const cluster = mode === "cluster" ? Lab.kmeans(points.map(p => p.xy), Number(document.getElementById("k").value)) : null;
  const centers = mode === "classify" ? Lab.fitClassifier(points.filter(p => !p.test)) : null;
  plot.replaceChildren();
  if (centers) {
    for (let row = 0; row < 20; row++) for (let col = 0; col < 30; col++) {
      const px = xmin + (col + .5) / 30 * (xmax - xmin), py = ymin + (row + .5) / 20 * (ymax - ymin);
      node("rect", { x: 60 + col * 650 / 30, y: 450 - (row + 1) * 410 / 20,
        width: 650 / 30 + .2, height: 410 / 20 + .2, fill: colors[Lab.predict(centers, [px, py])], opacity: .09 });
    }
    const test = points.filter(p => p.test);
    const correct = test.filter(p => Lab.predict(centers, p.xy) === p.label).length;
    title.textContent = "Nearest-centroid classifier";
    metric.textContent = `Held-out accuracy: ${correct}/${test.length} (${(100 * correct / test.length).toFixed(1)}%)`;
    explanation.textContent = "Shading shows predicted classes. Hollow points are held-out samples; their colors show true labels.";
    legend.textContent = "Teal: class 0  •  Orange: class 1  •  Hollow: held out";
  } else if (cluster) {
    title.textContent = "K-means clustering";
    metric.textContent = `${cluster.centers.length} clusters`;
    explanation.textContent = "K-means groups all projected points by proximity. Colors are cluster IDs, not class labels.";
    legend.textContent = "Cluster colors are arbitrary; hollow points are the held-out split.";
  } else {
    title.textContent = "Principal components";
    metric.textContent = `Training variance: PC1 ${(pca.explained[0] * 100).toFixed(1)}% · PC2 ${(pca.explained[1] * 100).toFixed(1)}%`;
    explanation.textContent = "PCA is fitted only on training data. Both train and held-out samples are projected onto its first two axes.";
    legend.textContent = "Teal: class 0  •  Orange: class 1  •  Hollow: held out";
  }
  node("line", { x1: 60, y1: 450, x2: 710, y2: 450, stroke: "#7c7b9b" });
  node("line", { x1: 60, y1: 40, x2: 60, y2: 450, stroke: "#7c7b9b" });
  node("text", { x: 360, y: 490, fill: "#bab9d1" }, "Principal component 1");
  node("text", { x: 65, y: 25, fill: "#bab9d1" }, "Principal component 2");
  points.forEach((point, i) => node("circle", { cx: x(point.xy[0]), cy: y(point.xy[1]), r: point.test ? 5 : 4,
    fill: point.test ? "#101020" : colors[cluster ? cluster.labels[i] : point.label],
    stroke: colors[cluster ? cluster.labels[i] : point.label], "stroke-width": 1.5, opacity: .9 }));
}
function generate() {
  const seed = document.getElementById("seed");
  if (!seed.reportValidity()) return;
  samples = Lab.generate(Number(seed.value), Number(document.getElementById("noise").value));
  document.getElementById("noiseValue").value = document.getElementById("noise").value;
  redraw();
}
document.getElementById("generate").addEventListener("click", generate);
document.getElementById("noise").addEventListener("input", generate);
document.getElementById("mode").addEventListener("change", redraw);
document.getElementById("k").addEventListener("change", redraw);
generate();

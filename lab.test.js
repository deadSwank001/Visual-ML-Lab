const test = require("node:test");
const assert = require("node:assert/strict");
const lab = require("./lab.js");
test("generation is reproducible and split contains both classes", () => {
  const data = lab.generate(42, .8);
  assert.deepEqual(data, lab.generate(42, .8));
  for (const split of [true, false]) assert.deepEqual([...new Set(data.filter(x => x.test === split).map(x => x.label))].sort(), [0, 1]);
});
test("PCA finds the highest variance direction", () => {
  const pca = lab.fitPCA([{ values: [-2, 0, 0] }, { values: [0, 0, 0] }, { values: [2, 0, 0] }]);
  assert.ok(Math.abs(Math.abs(pca.axes[0][0]) - 1) < 1e-9);
  assert.ok(Math.abs(pca.explained[0] - 1) < 1e-9);
});
test("k-means groups separated points and caps k at distinct points", () => {
  const result = lab.kmeans([[0, 0], [0, 0], [10, 10]], 4);
  assert.equal(result.centers.length, 2);
  assert.equal(result.labels[0], result.labels[1]);
  assert.notEqual(result.labels[0], result.labels[2]);
});
test("classifier predicts nearest class centroid", () => {
  const centers = lab.fitClassifier([{ label: 0, xy: [0, 0] }, { label: 1, xy: [4, 4] }]);
  assert.equal(lab.predict(centers, [1, 1]), 0);
  assert.equal(lab.predict(centers, [3, 3]), 1);
});

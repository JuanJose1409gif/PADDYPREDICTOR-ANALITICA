// ===== INFERENCE ENGINE =====

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function relu(x) { return Math.max(0, x); }

function logisticPredict(features) {
  const coef = MODEL_DATA.lr_coef[0];
  const intercept = MODEL_DATA.lr_intercept[0];
  let z = intercept;
  for (let i = 0; i < features.length; i++) z += coef[i] * features[i];
  const prob = sigmoid(z);
  return { pred: prob >= 0.5 ? 1 : 0, prob };
}

function nnPredict(features) {
  let a = [...features];
  const nLayers = MODEL_DATA.nn_coefs.length;
  for (let l = 0; l < nLayers; l++) {
    const W = MODEL_DATA.nn_coefs[l];    // shape: [in, out]
    const b = MODEL_DATA.nn_intercepts[l]; // shape: [out]
    const out = [];
    for (let j = 0; j < b.length; j++) {
      let s = b[j];
      for (let i = 0; i < a.length; i++) s += a[i] * W[i][j];
      // Last layer: sigmoid; hidden: relu
      out.push(l === nLayers - 1 ? sigmoid(s) : relu(s));
    }
    a = out;
  }
  const prob = a[0];
  return { pred: prob >= 0.5 ? 1 : 0, prob };
}

function scaleFeatures(rawFeatures) {
  const mean = MODEL_DATA.scaler_mean;
  const scale = MODEL_DATA.scaler_scale;
  return rawFeatures.map((v, i) => (v - mean[i]) / scale[i]);
}

function encodeRow(row) {
  const cols = MODEL_DATA.feature_cols;
  const encoders = MODEL_DATA.cat_encoders;
  return cols.map(col => {
    const val = row[col.trim()];
    if (encoders[col] !== undefined) {
      const enc = encoders[col];
      return enc[val] !== undefined ? enc[val] : 0;
    }
    return parseFloat(val) || 0;
  });
}

function predict(row, model = 'lr') {
  const raw = encodeRow(row);
  const scaled = scaleFeatures(raw);
  return model === 'lr' ? logisticPredict(scaled) : nnPredict(scaled);
}

function confusionMatrix(rows, model = 'lr') {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const row of rows) {
    const actual = parseInt(row['target'] ?? row['high_yield'] ?? -1);
    if (actual === -1) continue;
    const { pred } = predict(row, model);
    if (actual === 1 && pred === 1) tp++;
    else if (actual === 0 && pred === 0) tn++;
    else if (actual === 0 && pred === 1) fp++;
    else fn++;
  }
  const total = tp + tn + fp + fn;
  const accuracy = total > 0 ? ((tp + tn) / total * 100).toFixed(2) : 0;
  const precision = (tp + fp) > 0 ? (tp / (tp + fp) * 100).toFixed(2) : 0;
  const recall = (tp + fn) > 0 ? (tp / (tp + fn) * 100).toFixed(2) : 0;
  const f1p = parseFloat(precision), f1r = parseFloat(recall);
  const f1 = (f1p + f1r) > 0 ? (2 * f1p * f1r / (f1p + f1r)).toFixed(2) : 0;
  return { tp, tn, fp, fn, accuracy, precision, recall, f1 };
}

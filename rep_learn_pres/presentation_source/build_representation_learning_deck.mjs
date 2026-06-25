import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const TMP_DIR = path.dirname(fileURLToPath(import.meta.url));
const FINAL_PPTX = path.resolve(TMP_DIR, "..", "representation_learning_high_school_talk.pptx");
const PREVIEW_DIR = `${TMP_DIR}\\preview`;
const LAYOUT_DIR = `${TMP_DIR}\\layout`;
const QA_DIR = `${TMP_DIR}\\qa`;
const CAR_JSON = `${TMP_DIR}\\data\\car_experiment_metrics.json`;
const BERT_JSON = `${TMP_DIR}\\models\\modernbert_examples.json`;

const W = 1280;
const H = 720;
const BLACK = "#111111";
const MUTED = "#555555";
const RULE = "#B8BCC4";
const PANEL = "#EDEDED";
const PALE = "#F7F7F7";
const HIGHLIGHT = "#FF6B35";
const BLUE = "#3266CC";
const GREEN = "#207A4A";
const FONT = "Helvetica Neue";

function noLine() {
  return { style: "solid", fill: "none", width: 0 };
}

function line(fill = RULE, width = 1) {
  return { style: "solid", fill, width };
}

function addText(slide, name, value, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: noLine(),
  });
  shape.text = value;
  const textStyle = {
    fontSize: style.fontSize ?? 24,
    bold: Boolean(style.bold),
    color: style.color ?? BLACK,
    alignment: style.alignment ?? "left",
    typeface: FONT,
  };
  shape.text.style = textStyle;
  return shape;
}

function addRect(slide, name, position, fill = PANEL, stroke = "none", width = 1) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position,
    fill,
    line: stroke === "none" ? noLine() : line(stroke, width),
  });
}

function addCircle(slide, name, position, fill = BLACK, stroke = "none") {
  return slide.shapes.add({
    geometry: "ellipse",
    name,
    position,
    fill,
    line: stroke === "none" ? noLine() : line(stroke, 1),
  });
}

function addRule(slide, left, top, width) {
  slide.shapes.add({
    geometry: "line",
    name: "quiet-rule",
    position: { left, top, width, height: 0 },
    fill: "none",
    line: line(RULE, 1),
  });
}

function addFooter(slide, slideNumber, source = "") {
  addText(slide, `footer-source-${slideNumber}`, source, { left: 42, top: 660, width: 850, height: 28 }, { fontSize: 13, color: MUTED });
  addText(slide, `footer-page-${slideNumber}`, String(slideNumber).padStart(2, "0"), { left: 1180, top: 660, width: 58, height: 28 }, { fontSize: 15, color: MUTED, alignment: "right" });
}

function titleSlide(slide, title, kicker, slideNumber) {
  addText(slide, `title-${slideNumber}`, title, { left: 42, top: 36, width: 900, height: 112 }, { fontSize: 43, bold: true });
  addText(slide, `kicker-${slideNumber}`, kicker, { left: 42, top: 128, width: 760, height: 60 }, { fontSize: 24, color: MUTED });
  addRule(slide, 42, 194, 1196);
}

function metricBox(slide, name, label, value, left, top, fill = PALE) {
  addRect(slide, `${name}-box`, { left, top, width: 250, height: 150 }, fill);
  addText(slide, `${name}-value`, value, { left: left + 22, top: top + 25, width: 206, height: 60 }, { fontSize: 42, bold: true });
  addText(slide, `${name}-label`, label, { left: left + 22, top: top + 93, width: 206, height: 42 }, { fontSize: 18, color: MUTED });
}

function bulletList(slide, name, items, left, top, width, size = 24) {
  const text = items.map((item) => `- ${item}`).join("\n");
  addText(slide, name, text, { left, top, width, height: 360 }, { fontSize: size });
}

function sourceMetric(summary, encoding, field) {
  const row = summary.find((item) => item.encoding === encoding);
  if (!row) {
    throw new Error(`Missing summary row for ${encoding}`);
  }
  return row[field];
}

function currency(value) {
  return `$${Math.round(value).toLocaleString("en-US")}`;
}

function fixed(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function addThreeColumns(slide, cols, top = 240) {
  const lefts = [42, 452, 862];
  for (let idx = 0; idx < cols.length; idx += 1) {
    const col = cols[idx];
    addText(slide, `col-title-${idx}`, col.title, { left: lefts[idx], top, width: 330, height: 44 }, { fontSize: 28, bold: true });
    addText(slide, `col-body-${idx}`, col.body, { left: lefts[idx], top: top + 62, width: 330, height: 210 }, { fontSize: 21, color: MUTED });
    addRect(slide, `col-rule-${idx}`, { left: lefts[idx], top: top - 26, width: 28, height: 28 }, idx === 1 ? HIGHLIGHT : BLACK);
  }
}

function addProcessRow(slide, steps, top = 330) {
  const cellW = 220;
  const gap = 26;
  let left = 58;
  const boxes = [];
  for (let idx = 0; idx < steps.length; idx += 1) {
    const box = addRect(slide, `step-${idx}`, { left, top, width: cellW, height: 120 }, idx === steps.length - 1 ? "#FFF0E8" : PALE, idx === steps.length - 1 ? HIGHLIGHT : "none");
    boxes.push(box);
    addText(slide, `step-label-${idx}`, steps[idx], { left: left + 18, top: top + 28, width: cellW - 36, height: 64 }, { fontSize: 24, bold: true, alignment: "center" });
    left += cellW + gap;
  }
  for (let idx = 0; idx < boxes.length - 1; idx += 1) {
    slide.shapes.connect(boxes[idx], boxes[idx + 1], {
      kind: "straight",
      fromSide: "right",
      toSide: "left",
      line: line(MUTED, 2),
      tail: { type: "arrow", width: "med", length: "med" },
    });
  }
}

function applySpeakerNotes(presentation) {
  const notes = [
    "Open with the one-sentence promise: representation learning means turning messy things into useful numbers. Keep this welcoming and say there will be math, but all of it will connect to intuition.",
    "Frame the talk as a path. Tell them the car-price example is the concrete middle, then language models and proteins are the payoff.",
    "Slow down here. Connect f(x)=y to functions they already know, then say machine learning often means we do not know f, but we have examples of x and y.",
    "Use quick audience examples. Ask for one prediction problem from everyday life, then show language and biology as the same pattern.",
    "Define parameters as knobs. The loop is predict, measure, adjust, repeat. Avoid jargon beyond that unless someone asks.",
    "Make error concrete with dollars. The important idea is not the exact loss formula yet, just that wrongness becomes a number.",
    "Use this as the two-parameter mental picture from the outline. A gradient tells you which direction on the loss map moves downhill.",
    "Say a vector is a location in number-space. Once something is a vector, we can compare, move, and transform it with math.",
    "Dot products are the attention hook. They ask how much two vectors point in the same direction.",
    "Demystify neural nets. Matrix multiply, small nonlinear twist, repeat. Each layer makes a new representation.",
    "Introduce the real experiment. Be transparent that the 25k rows are a laptop-friendly deterministic subset of the public dataset.",
    "Emphasize feature choice. Wheels are almost always four, so they barely vary; mileage and horsepower vary and can carry signal.",
    "Use this as the representation-learning hinge. Integer IDs are fake order, one-hot is honest but sparse, embeddings are learned geometry.",
    "Stress fairness without overstating it: same dataset, splits, numeric features, optimizer, and test protocol, but category representation and parameter counts differ.",
    "Tell the honest result. Integer IDs failed; embeddings were slightly better on RMSE and R2 in this run, while one-hot had better MAE and overlapping variation.",
    "Explain the plot carefully. This is a 2D projection from 16 dimensions and shows learned geometry for prediction, not real-world truth about brands.",
    "Transition from cars to words. A word is a category before it becomes a vector.",
    "ModernBERT is a masked-language model. Read one prompt aloud and ask the audience what they would fill in before revealing the model answer.",
    "This is the caveat slide. The strange strings are subword token fragments, not proof that the model is broken. ModernBERT is contextual and tokenization-sensitive.",
    "Use bank as the memorable example. Same spelling, different surroundings, different representation.",
    "Attention is the mechanism that lets tokens compare context. Keep it high-level: learned dot-product questions between tokens.",
    "Proteins are sequences over an alphabet. The same idea that embeds words can embed amino acid sequences. Mention ProtTrans and ESM as well-known examples.",
    "Pause here for the live demo. Say: input a protein sequence, compute an embedding, then use that vector for annotation, similarity search, or design triage.",
    "Close with the unifying idea. Sources and methods: outline.md, gsv24/car-price, PyTorch experiment in scratch, answerdotai/ModernBERT-base, ProtTrans arXiv:2007.06225, and ESM protein-language-model literature.",
  ];
  if (notes.length !== presentation.slides.items.length) {
    throw new Error(`Expected ${presentation.slides.items.length} notes, got ${notes.length}`);
  }
  for (const [index, slide] of presentation.slides.items.entries()) {
    slide.speakerNotes.textFrame.setText(notes[index]);
    slide.speakerNotes.setVisible(true);
  }
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

function makeDeck(car, bert) {
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  const summary = car.summary;
  const rmseInt = sourceMetric(summary, "integer IDs", "test_rmse_mean");
  const rmseHot = sourceMetric(summary, "one-hot", "test_rmse_mean");
  const rmseEmb = sourceMetric(summary, "learned embeddings", "test_rmse_mean");
  const maeHot = sourceMetric(summary, "one-hot", "test_mae_mean");
  const maeEmb = sourceMetric(summary, "learned embeddings", "test_mae_mean");
  const r2Hot = sourceMetric(summary, "one-hot", "test_r2_mean");
  const r2Emb = sourceMetric(summary, "learned embeddings", "test_r2_mean");
  const projectionPoints = car.embedding_projection.points.slice(0, 25);

  let slideNumber = 1;
  function addSlide() {
    const slide = presentation.slides.add();
    slide.background.fill = "#FFFFFF";
    return slide;
  }

  let slide = addSlide();
  addText(slide, "cover-title", "Representation Learning", { left: 42, top: 62, width: 900, height: 110 }, { fontSize: 72, bold: true });
  addText(slide, "cover-subtitle", "Turning cars, words, and proteins into useful numbers", { left: 42, top: 178, width: 760, height: 70 }, { fontSize: 28, color: MUTED });
  addText(slide, "cover-audience", "45 minute high school math camp talk", { left: 42, top: 560, width: 520, height: 40 }, { fontSize: 22, color: MUTED });
  addRect(slide, "cover-field", { left: 864, top: 64, width: 374, height: 530 }, PALE);
  addText(slide, "cover-equation", "x -> vector -> y", { left: 900, top: 256, width: 300, height: 64 }, { fontSize: 38, bold: true, alignment: "center" });
  addText(slide, "cover-mini", "same math,\nmany objects", { left: 920, top: 346, width: 260, height: 84 }, { fontSize: 25, color: MUTED, alignment: "center" });
  addFooter(slide, slideNumber, "Outline: rep_learn_pres/outline.md");

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Where we are going", "A short path from functions to protein language models.", slideNumber);
  addText(slide, "agenda-list", "01  Functions that predict\n02  Neural nets as matrix math\n03  Car prices and categorical embeddings\n04  Words, ModernBERT, and transformers\n05  Proteins and the demo", { left: 100, top: 236, width: 780, height: 330 }, { fontSize: 36, bold: true });
  addText(slide, "agenda-time", "Rough timing\n10 min basics\n15 min car experiment\n12 min language models\n8 min proteins/demo", { left: 900, top: 250, width: 280, height: 260 }, { fontSize: 22, color: MUTED });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "The core pattern is a function", "Machine learning starts from a familiar math idea.", slideNumber);
  addText(slide, "function-eq", "f(x) = y", { left: 168, top: 260, width: 390, height: 90 }, { fontSize: 68, bold: true, alignment: "center" });
  addText(slide, "function-copy", "We want a useful function: feed in information, get out a valuable answer.", { left: 660, top: 250, width: 460, height: 130 }, { fontSize: 30 });
  addText(slide, "function-examples", "car -> price\nsentence -> missing word\nprotein sequence -> annotation", { left: 690, top: 410, width: 440, height: 140 }, { fontSize: 25, color: MUTED });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Prediction shows up everywhere", "The same pattern can describe very different problems.", slideNumber);
  addThreeColumns(slide, [
    { title: "Everyday", body: "Price, weather, traffic, medical risk, recommendations." },
    { title: "Language", body: "Translate, summarize, answer, classify, or fill in a missing word." },
    { title: "Biology", body: "Annotate proteins, predict structure, suggest useful designs." },
  ], 255);
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Learning means tuning the function", "The machine is a function; the learning is adjusting its parameters.", slideNumber);
  addProcessRow(slide, ["predict", "measure error", "adjust knobs", "try again"], 330);
  addText(slide, "learning-note", "A parameter is just a number the model is allowed to change.", { left: 130, top: 518, width: 1020, height: 44 }, { fontSize: 26, alignment: "center" });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Error gives training a target", "If the prediction is wrong, the loss turns wrongness into a number.", slideNumber);
  addRect(slide, "loss-panel", { left: 86, top: 250, width: 520, height: 210 }, PALE);
  addText(slide, "loss-example", "true price:      $20,000\nmodel guess: $18,000\nerror signal:   $2,000", { left: 122, top: 288, width: 450, height: 132 }, { fontSize: 30, bold: true });
  addText(slide, "loss-copy", "Training tries to make this error smaller across many examples, not just one.", { left: 700, top: 282, width: 440, height: 150 }, { fontSize: 30 });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Gradient descent is downhill math", "A two-parameter picture: move toward lower error.", slideNumber);
  addRect(slide, "loss-map-frame", { left: 106, top: 230, width: 640, height: 360 }, PALE, RULE);
  const rings = [
    { left: 214, top: 260, width: 430, height: 270, stroke: "#BBBBBB" },
    { left: 260, top: 292, width: 338, height: 210, stroke: "#999999" },
    { left: 313, top: 326, width: 232, height: 145, stroke: "#777777" },
    { left: 374, top: 362, width: 112, height: 72, stroke: HIGHLIGHT },
  ];
  for (const [idx, ring] of rings.entries()) {
    addCircle(slide, `loss-contour-${idx}`, { left: ring.left, top: ring.top, width: ring.width, height: ring.height }, "none", ring.stroke);
  }
  const path = [
    [606, 292],
    [552, 322],
    [505, 350],
    [465, 376],
    [430, 396],
  ];
  for (let idx = 0; idx < path.length; idx += 1) {
    addCircle(slide, `descent-point-${idx}`, { left: path[idx][0], top: path[idx][1], width: 14, height: 14 }, idx === path.length - 1 ? HIGHLIGHT : BLACK);
    if (idx < path.length - 1) {
      const a = path[idx];
      const b = path[idx + 1];
      slide.shapes.add({ geometry: "line", name: `descent-step-${idx}`, position: { left: b[0] + 7, top: b[1] + 7, width: a[0] - b[0], height: a[1] - b[1] }, fill: "none", line: line(HIGHLIGHT, 2) });
    }
  }
  addText(slide, "theta-x", "parameter 1", { left: 330, top: 545, width: 210, height: 28 }, { fontSize: 18, color: MUTED, alignment: "center" });
  addText(slide, "theta-y", "parameter 2", { left: 118, top: 378, width: 110, height: 28 }, { fontSize: 18, color: MUTED, alignment: "center" });
  addText(slide, "gradient-note", "Each contour is a loss level. The gradient points toward the downhill direction.", { left: 835, top: 285, width: 340, height: 140 }, { fontSize: 27 });
  addText(slide, "gradient-note-2", "Most real models have many more than two parameters, but this picture carries the intuition.", { left: 835, top: 465, width: 340, height: 100 }, { fontSize: 23, color: MUTED });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Vectors are objects as number lists", "Once an object is a vector, geometry becomes useful.", slideNumber);
  addText(slide, "vector-car", "car = [year, mileage, horsepower, ...]", { left: 92, top: 250, width: 780, height: 60 }, { fontSize: 34, bold: true });
  addText(slide, "vector-word", "word = [0.12, -0.44, 1.80, ...]", { left: 92, top: 344, width: 780, height: 60 }, { fontSize: 34, bold: true });
  addText(slide, "vector-copy", "A vector is a location. Similar locations can mean similar behavior.", { left: 740, top: 480, width: 390, height: 90 }, { fontSize: 28, color: MUTED });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Dot products compare directions", "A dot product asks how much one vector lines up with another.", slideNumber);
  addText(slide, "dot-eq", "a . b = sum(a_i b_i)", { left: 110, top: 245, width: 560, height: 80 }, { fontSize: 50, bold: true });
  addRect(slide, "dot-high", { left: 740, top: 235, width: 360, height: 110 }, "#FFF0E8", HIGHLIGHT);
  addText(slide, "dot-high-text", "same direction\nlarge score", { left: 772, top: 260, width: 300, height: 65 }, { fontSize: 26, bold: true, alignment: "center" });
  addRect(slide, "dot-low", { left: 740, top: 385, width: 360, height: 110 }, PALE, RULE);
  addText(slide, "dot-low-text", "different direction\nsmall score", { left: 772, top: 410, width: 300, height: 65 }, { fontSize: 26, bold: true, alignment: "center" });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Neural nets are organized matrix math", "Layers repeatedly transform vectors into better vectors.", slideNumber);
  addProcessRow(slide, ["input vector", "matrix multiply", "nonlinear twist", "prediction"], 310);
  addText(slide, "nn-copy", "Each layer reshapes the representation, so later layers can predict from better features.", { left: 176, top: 520, width: 920, height: 46 }, { fontSize: 28, alignment: "center" });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Case study: used car prices", "We trained small neural nets on a public car-price dataset.", slideNumber);
  metricBox(slide, "rows", "rows used", `${car.rows_used.toLocaleString("en-US")}`, 70, 250);
  metricBox(slide, "split", "train / val / test", "70 / 15 / 15", 380, 250);
  metricBox(slide, "seeds", "random seeds", "5", 690, 250);
  metricBox(slide, "features", "categorical fields", `${car.categorical_features.length}`, 1000, 250);
  addText(slide, "dataset-note", "Features included year, mileage, engine horsepower, make, model, fuel, transmission, color, condition, and trim.", { left: 120, top: 510, width: 1040, height: 70 }, { fontSize: 24, color: MUTED, alignment: "center" });
  addFooter(slide, slideNumber, "Source: gsv24/car-price on Hugging Face");

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Some features are already numbers", "Other features are categories pretending to be words.", slideNumber);
  addText(slide, "num-title", "Numeric", { left: 150, top: 250, width: 360, height: 50 }, { fontSize: 34, bold: true });
  addText(slide, "cat-title", "Categorical", { left: 720, top: 250, width: 360, height: 50 }, { fontSize: 34, bold: true });
  bulletList(slide, "num-list", ["year", "mileage", "engine horsepower", "owner count"], 150, 320, 390, 27);
  bulletList(slide, "cat-list", ["make", "model", "fuel type", "transmission", "exterior color"], 720, 320, 420, 27);
  addText(slide, "cat-warning", "Feature choice matters: wheels are almost always 4.", { left: 250, top: 545, width: 780, height: 34 }, { fontSize: 22, color: HIGHLIGHT, bold: true, alignment: "center" });
  addText(slide, "cat-warning-2", "Mileage and horsepower vary. Categories need a different representation.", { left: 220, top: 594, width: 840, height: 34 }, { fontSize: 18, color: MUTED, alignment: "center" });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Three ways to feed categories to a model", "The representation changes what the model can learn.", slideNumber);
  addThreeColumns(slide, [
    { title: "Integer IDs", body: "Make each category a number. Fast, but it creates fake order." },
    { title: "One-hot", body: "One slot per category. Honest, but sparse and not semantic." },
    { title: "Embeddings", body: "Train a compact vector for each category as part of the network." },
  ], 252);
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "The experiment used the same recipe", "Same data and training loop; different category representation.", slideNumber);
  addProcessRow(slide, ["same split", "same numeric features", "same optimizer", "same test set"], 300);
  addText(slide, "setup-details", "Target: log1p(price). Metrics reported back on dollars. Validation selected checkpoints; test was used once per seed.", { left: 124, top: 500, width: 1030, height: 78 }, { fontSize: 24, color: MUTED, alignment: "center" });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Embeddings helped, with a caveat", "Integer IDs failed; embeddings were slightly better on RMSE/R2 in this run.", slideNumber);
  slide.charts.add("bar", {
    position: { left: 74, top: 245, width: 548, height: 310 },
    categories: ["One-hot", "Embeddings"],
    series: [{
      name: "RMSE",
      values: [Math.round(rmseHot), Math.round(rmseEmb)],
      fill: HIGHLIGHT,
      points: [{ idx: 0, fill: BLUE }, { idx: 1, fill: HIGHLIGHT }],
    }],
    hasLegend: false,
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 50 },
    yAxis: { title: "test RMSE, dollars", majorGridlines: line("#DDDDDD", 1), textStyle: { fill: MUTED, fontSize: 13 } },
    xAxis: { textStyle: { fill: BLACK, fontSize: 14 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: BLACK, fontSize: 14, bold: true } },
  });
  addText(slide, "results-table", `Encoding              RMSE          MAE          R2\nInteger IDs       ${currency(rmseInt)}   ${currency(sourceMetric(summary, "integer IDs", "test_mae_mean"))}   very poor\nOne-hot             ${currency(rmseHot)}     ${currency(maeHot)}      ${fixed(r2Hot)}\nEmbeddings       ${currency(rmseEmb)}     ${currency(maeEmb)}      ${fixed(r2Emb)}`, { left: 690, top: 250, width: 500, height: 210 }, { fontSize: 22, bold: true });
  addText(slide, "results-caveat", "Truthful read: the RMSE/R2 advantage is small, and one-hot had lower MAE.", { left: 690, top: 490, width: 500, height: 70 }, { fontSize: 22, color: MUTED });
  addText(slide, "integer-off-scale", `integer IDs were off-scale at ${currency(rmseInt)} RMSE`, { left: 152, top: 565, width: 430, height: 34 }, { fontSize: 18, color: MUTED, alignment: "center" });
  addFooter(slide, slideNumber, "Experiment: PyTorch MLP, 25k deterministic subset, 5 seeds");

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Embeddings create a learned map", "A category becomes a point the model can move during training.", slideNumber);
  addRect(slide, "scatter-frame", { left: 80, top: 220, width: 660, height: 380 }, PALE, RULE);
  const xs = projectionPoints.map((point) => point.x);
  const ys = projectionPoints.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  for (let idx = 0; idx < projectionPoints.length; idx += 1) {
    const point = projectionPoints[idx];
    const x = 110 + ((point.x - minX) / (maxX - minX)) * 595;
    const y = 560 - ((point.y - minY) / (maxY - minY)) * 300;
    addCircle(slide, `brand-dot-${idx}`, { left: x, top: y, width: 10, height: 10 }, idx % 3 === 0 ? HIGHLIGHT : BLACK);
    if (["tesla", "toyota", "bmw", "ford", "lexus", "subaru"].includes(point.label)) {
      const labelLeft = x > 610 ? x - 88 : x + 12;
      addText(slide, `brand-label-${idx}`, point.label, { left: labelLeft, top: y - 8, width: 120, height: 24 }, { fontSize: 14, color: MUTED });
    }
  }
  addText(slide, "embedding-map-copy", "This is a 2D projection of the learned make embeddings. It is not proof of real-world meaning; it shows geometry the model used for price prediction.", { left: 820, top: 265, width: 330, height: 170 }, { fontSize: 25 });
  addText(slide, "embedding-map-note", "The real vectors were 16-dimensional.", { left: 820, top: 470, width: 330, height: 50 }, { fontSize: 24, color: HIGHLIGHT, bold: true });
  addFooter(slide, slideNumber, "Projection: PCA of learned make embedding table");

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Words are categories too", "A word starts as a symbol. An embedding turns it into math.", slideNumber);
  addProcessRow(slide, ["word token", "token ID", "embedding vector", "contextual vector"], 300);
  addText(slide, "word-note", "This is the same representation-learning move, now applied to language.", { left: 174, top: 505, width: 930, height: 50 }, { fontSize: 28, alignment: "center" });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "ModernBERT predicts masked words", "Masked-language models learn by filling in blanks.", slideNumber);
  const maskRows = bert.fill_mask_examples.map((item) => {
    const top = item.predictions[0];
    return `${item.prompt}\nTop answer: ${top.token} (${Math.round(top.probability * 100)}%)`;
  });
  addText(slide, "mask-row-1", maskRows[0], { left: 82, top: 238, width: 1040, height: 76 }, { fontSize: 26, bold: true });
  addText(slide, "mask-row-2", maskRows[1], { left: 82, top: 350, width: 1040, height: 76 }, { fontSize: 26, bold: true });
  addText(slide, "mask-row-3", maskRows[2], { left: 82, top: 462, width: 1040, height: 76 }, { fontSize: 26, bold: true });
  addFooter(slide, slideNumber, "Model: answerdotai/ModernBERT-base");

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "ModernBERT is not classic word2vec", "The famous analogy is useful history, not a promise for every model.", slideNumber);
  addText(slide, "analogy-left", "Historical intuition\nking - man + woman ~= queen", { left: 92, top: 255, width: 470, height: 120 }, { fontSize: 32, bold: true });
  addText(slide, "analogy-right", "ModernBERT input vectors tried:\nking - man + woman -> ked, ving, women...", { left: 690, top: 245, width: 470, height: 130 }, { fontSize: 28, bold: true, color: HIGHLIGHT });
  addText(slide, "analogy-subword", "Those odd strings are subword token fragments, not normal words.", { left: 690, top: 392, width: 470, height: 54 }, { fontSize: 22, color: MUTED });
  addText(slide, "analogy-bottom", "The stronger ModernBERT idea is contextual representation: the vector changes with the sentence.", { left: 160, top: 520, width: 950, height: 62 }, { fontSize: 26, color: MUTED, alignment: "center" });
  addFooter(slide, slideNumber, "ModernBERT-only experiment");

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Context changes meaning", "The same token can become a different vector in a different sentence.", slideNumber);
  addText(slide, "bank-river", "river bank\nwater, shore, fishing", { left: 130, top: 270, width: 360, height: 130 }, { fontSize: 34, bold: true, alignment: "center" });
  addText(slide, "bank-money", "bank account\nmoney, teller, savings", { left: 780, top: 270, width: 360, height: 130 }, { fontSize: 34, bold: true, alignment: "center" });
  addText(slide, "context-arrow", "same word\nnew context\nnew vector", { left: 520, top: 278, width: 220, height: 130 }, { fontSize: 28, color: HIGHLIGHT, bold: true, alignment: "center" });
  addText(slide, "context-note", "Attention is one way transformers decide which other words matter right now.", { left: 180, top: 510, width: 920, height: 52 }, { fontSize: 26, color: MUTED, alignment: "center" });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Transformers scale the idea", "Tokens become vectors; layers compare and update them; outputs become useful.", slideNumber);
  addProcessRow(slide, ["tokens", "embeddings", "attention", "layers", "output"], 300);
  addText(slide, "chatgpt-note", "ChatGPT extends this transformer recipe to generate text one token at a time.", { left: 200, top: 505, width: 880, height: 50 }, { fontSize: 28, alignment: "center" });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Proteins are sequences too", "Protein language models apply representation learning to amino acid strings.", slideNumber);
  addText(slide, "protein-seq", "M K T A Y I A K Q R Q I S F V K S H F S R Q", { left: 90, top: 260, width: 1080, height: 64 }, { fontSize: 34, bold: true, alignment: "center" });
  addThreeColumns(slide, [
    { title: "Input", body: "A sequence of amino acid letters." },
    { title: "Representation", body: "A vector for each residue or whole protein." },
    { title: "Use", body: "Annotation, similarity search, design, and experiments." },
  ], 420);
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "Protein embedding demo", "A live bridge from the lesson to a real protein-language-model workflow.", slideNumber);
  addRect(slide, "demo-frame", { left: 122, top: 238, width: 1036, height: 290 }, "#FFF0E8", HIGHLIGHT, 2);
  addText(slide, "demo-title", "From amino acids to a useful vector", { left: 164, top: 286, width: 952, height: 60 }, { fontSize: 42, bold: true, alignment: "center" });
  addText(slide, "demo-copy", "Input: a protein sequence\nOutput: an embedding vector\nUse: annotation, similarity search, or design triage", { left: 230, top: 352, width: 820, height: 120 }, { fontSize: 26, alignment: "center" });
  addFooter(slide, slideNumber);

  slideNumber += 1;
  slide = addSlide();
  titleSlide(slide, "The big takeaways", "Representation learning lets math work on messy objects.", slideNumber);
  addThreeColumns(slide, [
    { title: "1", body: "Training tunes a function to reduce prediction error." },
    { title: "2", body: "Embeddings are trainable vectors for categories, words, and sequences." },
    { title: "3", body: "Transformers and protein language models are built on this same idea." },
  ], 250);
  addText(slide, "qa", "Questions", { left: 450, top: 575, width: 380, height: 60 }, { fontSize: 48, bold: true, alignment: "center" });
  addFooter(slide, slideNumber);

  if (presentation.slides.items.length !== 24) {
    throw new Error(`Expected 24 slides, built ${presentation.slides.items.length}`);
  }
  applySpeakerNotes(presentation);
  return presentation;
}

async function main() {
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });
  const car = JSON.parse(await fs.readFile(CAR_JSON, "utf8"));
  const bert = JSON.parse(await fs.readFile(BERT_JSON, "utf8"));
  const presentation = makeDeck(car, bert);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1.5 });
    await writeBlob(`${PREVIEW_DIR}\\${stem}.png`, png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${LAYOUT_DIR}\\${stem}.layout.json`, await layout.text());
  }

  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(`${QA_DIR}\\deck-montage.webp`, montage);
  const inspect = await presentation.inspect({ kind: "slide,textbox,shape,image,table,chart,notes,layout", maxChars: 50000 });
  await fs.writeFile(`${QA_DIR}\\inspect.ndjson`, inspect.ndjson, "utf8");
  await fs.writeFile(`${TMP_DIR}\\source-notes.txt`, [
    "Deck source notes",
    "Outline: rep_learn_pres/outline.md",
    "Car dataset: gsv24/car-price on Hugging Face. Used 25,000 deterministic rows for laptop-friendly PyTorch training.",
    "Language model: answerdotai/ModernBERT-base on Hugging Face. Used masked-token examples and an honest input-embedding caveat.",
    "Generated with @oai/artifact-tool from a plain JavaScript ES module.",
  ].join("\n"), "utf8");
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(JSON.stringify({ final_pptx: FINAL_PPTX, slides: presentation.slides.items.length }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

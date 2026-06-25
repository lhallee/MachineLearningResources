import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const SOURCE_DIR = process.env.REP_LEARN_SOURCE_DIR || path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(SOURCE_DIR, "..", "..");
const WORKSPACE = "C:\\tmp\\codex-presentations\\rep-learn-workshop-pivot";
const STARTER = path.join(WORKSPACE, "template-starter.pptx");
const FINAL = path.resolve(SOURCE_DIR, "..", "representation_learning_high_school_talk.pptx");
const BACKUP = path.join(
  WORKSPACE,
  `representation_learning_high_school_talk.backup.${new Date().toISOString().replace(/[:.]/g, "-")}.pptx`,
);
const QA_DIR = path.join(WORKSPACE, "qa-final");
const PREVIEW_DIR = path.join(QA_DIR, "slides");
const LAYOUT_DIR = path.join(QA_DIR, "layouts");

const CAR_STORY = path.join(SOURCE_DIR, "data", "car_story_metrics.json");
const CAR_FULL = path.join(SOURCE_DIR, "data", "car_experiment_metrics.json");
const BERT_JSON = path.join(SOURCE_DIR, "models", "modernbert_examples.json");
const HERO = path.join(SOURCE_DIR, "assets", "representation-hero.png");
const BACKPACK = path.join(SOURCE_DIR, "assets", "red-backpack-photo.png");
const PROTEIN = path.join(SOURCE_DIR, "assets", "protein-language-model.png");
const TOPK = path.join(SOURCE_DIR, "models", "modernbert_plots", "modernbert_topk_successes.png");
const TOPK_CAPITALS = path.join(SOURCE_DIR, "models", "modernbert_plots", "modernbert_topk_capitals_crop.png");
const TOPK_COMPARATIVES = path.join(SOURCE_DIR, "models", "modernbert_plots", "modernbert_topk_comparatives_crop.png");
const PCA = path.join(SOURCE_DIR, "models", "modernbert_plots", "modernbert_query_pca.png");

const INK = "#1E2D31";
const MUTED = "#607074";
const RULE = "#D6E2E6";
const CYAN = "#5FE3FD";
const BLUE = "#2F73D8";
const VIOLET = "#6E5AEF";
const TEAL = "#008C7E";
const RED = "#E45A3C";
const GOLD = "#DCA62B";
const WHITE = "#FFFFFF";
const FONT = "Helvetica Neue";

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function readBytes(filePath) {
  const bytes = await fs.readFile(filePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function slide(deck, n) {
  return deck.slides.getItem(n - 1);
}

function hasText(shape) {
  return "text" in shape;
}

function textShapes(s) {
  return s.shapes.items.filter((shape) => hasText(shape));
}

function isPageMarker(shape) {
  const frame = shape.frame;
  return frame.left > 810 && frame.top > 470;
}

function titleShape(s) {
  const candidates = textShapes(s).filter(
    (shape) => !isPageMarker(shape) && shape.frame.top < 60 && shape.frame.width > 450,
  );
  assert(candidates.length >= 1, "expected inherited title shape");
  return candidates[0];
}

function bodyShape(s) {
  const title = titleShape(s);
  const candidates = textShapes(s).filter((shape) => !isPageMarker(shape) && shape !== title);
  if (candidates.length === 0) return undefined;
  candidates.sort((a, b) => b.frame.width * b.frame.height - a.frame.width * a.frame.height);
  return candidates[0];
}

function setShapeText(shape, text) {
  shape.text.set(text);
}

function setTitleBody(s, title, body) {
  const titleTarget = titleShape(s);
  const bodyTarget = bodyShape(s);
  setShapeText(titleTarget, title);
  if (bodyTarget !== undefined) {
    setShapeText(bodyTarget, body);
  } else if (body.length > 0) {
    addText(s, `body-added-${title}`, body, { left: 36, top: 78, width: 860, height: 110 }, { fontSize: 24 });
  }
  for (const shape of textShapes(s)) {
    if (shape === titleTarget) continue;
    if (bodyTarget !== undefined && shape === bodyTarget) continue;
    if (isPageMarker(shape)) continue;
    shape.delete();
  }
}

function preserveLogo(image) {
  const frame = image.frame;
  return frame.left < 55 && frame.top < 60 && frame.width < 60 && frame.height < 60;
}

function clearImages(s) {
  for (const image of [...s.images.items]) {
    if (preserveLogo(image)) continue;
    image.delete();
  }
}

function addText(s, name, text, position, style = {}) {
  const shape = s.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text.set(text);
  shape.text.style = {
    fontSize: style.fontSize || 18,
    bold: style.bold || false,
    color: style.color || INK,
    alignment: style.alignment || "left",
  };
  shape.text.typeface = FONT;
  return shape;
}

function addRect(s, name, position, fill = WHITE, lineFill = RULE, lineWidth = 1) {
  return s.shapes.add({
    geometry: "rect",
    name,
    position,
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
  });
}

function addRoundRect(s, name, position, fill = WHITE, lineFill = RULE, lineWidth = 1) {
  return s.shapes.add({
    geometry: "roundRect",
    name,
    position,
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
    borderRadius: "rounded-md",
  });
}

function addCircle(s, name, cx, cy, r, fill, lineFill = fill) {
  return s.shapes.add({
    geometry: "ellipse",
    name,
    position: { left: cx - r, top: cy - r, width: r * 2, height: r * 2 },
    fill,
    line: { style: "solid", fill: lineFill, width: 1 },
  });
}

function addLine(s, name, x1, y1, x2, y2, color = INK, width = 2) {
  return s.shapes.add({
    geometry: "line",
    name,
    position: { left: x1, top: y1, width: x2 - x1, height: y2 - y1 },
    fill: "none",
    line: { style: "solid", fill: color, width },
  });
}

async function addImage(s, name, filePath, position, fit = "contain", crop = undefined) {
  const config = {
    blob: await readBytes(filePath),
    contentType: "image/png",
    alt: name,
    fit,
    position,
  };
  if (crop !== undefined) config.crop = crop;
  return s.images.add(config);
}

function metricCard(s, label, value, x, color = CYAN) {
  addRoundRect(s, `metric-${label}`, { left: x, top: 215, width: 150, height: 110 }, WHITE, RULE, 1);
  addText(s, `metric-value-${label}`, value, { left: x + 10, top: 235, width: 130, height: 34 }, {
    fontSize: 24,
    bold: true,
    color,
  });
  addText(s, `metric-label-${label}`, label, { left: x + 10, top: 275, width: 130, height: 42 }, {
    fontSize: 13,
    color: MUTED,
    alignment: "center",
  });
}

function flowBox(s, name, text, x, y, w, color = BLUE) {
  addRoundRect(s, `box-${name}`, { left: x, top: y, width: w, height: 62 }, WHITE, color, 1.5);
  addText(s, `label-${name}`, text, { left: x + 8, top: y + 16, width: w - 16, height: 28 }, {
    fontSize: 16,
    bold: true,
    color,
    alignment: "center",
  });
}

function drawTable(s, name, rows, x, y, widths, rowH) {
  let top = y;
  for (let r = 0; r < rows.length; r += 1) {
    let left = x;
    for (let c = 0; c < rows[r].length; c += 1) {
      const fill = r === 0 ? VIOLET : r % 2 === 0 ? "#F4F1EA" : WHITE;
      addRect(s, `${name}-cell-${r}-${c}`, { left, top, width: widths[c], height: rowH }, fill, RULE, 1);
      addText(s, `${name}-text-${r}-${c}`, String(rows[r][c]), {
        left: left + 8,
        top: top + 7,
        width: widths[c] - 12,
        height: rowH - 8,
      }, {
        fontSize: r === 0 ? 13 : 12,
        bold: r === 0 || c === 2,
        color: r === 0 ? WHITE : c === 2 ? TEAL : INK,
      });
      left += widths[c];
    }
    top += rowH;
  }
}

function drawBarChart(s, items, x, y, w, h, maxValue) {
  addRect(s, "bar-axis", { left: x, top: y, width: w, height: h }, WHITE, RULE, 1);
  const barW = 90;
  const gap = 62;
  for (let i = 0; i < items.length; i += 1) {
    const item = items[i];
    const bh = Math.max(8, (item.value / maxValue) * (h - 70));
    const bx = x + 58 + i * (barW + gap);
    const by = y + h - 46 - bh;
    addRect(s, `bar-${item.label}`, { left: bx, top: by, width: barW, height: bh }, item.color, item.color, 1);
    addText(s, `bar-val-${item.label}`, item.display, { left: bx - 6, top: by - 28, width: barW + 12, height: 22 }, {
      fontSize: 16,
      bold: true,
      color: item.color,
      alignment: "center",
    });
    addText(s, `bar-label-${item.label}`, item.label, { left: bx - 16, top: y + h - 36, width: barW + 32, height: 24 }, {
      fontSize: 12,
      color: MUTED,
      alignment: "center",
    });
  }
}

function drawScatter(s, points, frame, labels, color = BLUE) {
  addRect(s, "scatter-frame", frame, "none", RULE, 1);
  const usable = points.filter((p) => labels.includes(p.label));
  assert(usable.length > 4, "expected enough scatter points");
  const xs = usable.map((p) => p.x);
  const ys = usable.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  for (const p of usable) {
    const px = frame.left + 36 + ((p.x - minX) / (maxX - minX)) * (frame.width - 72);
    const py = frame.top + frame.height - 36 - ((p.y - minY) / (maxY - minY)) * (frame.height - 72);
    addCircle(s, `pt-${p.label}`, px, py, 5, color, color);
    addText(s, `label-${p.label}`, p.label, { left: px + 8, top: py - 10, width: 78, height: 20 }, {
      fontSize: 11,
      color: INK,
    });
  }
}

function drawHeatmap(s, x, y) {
  const words = ["the", "protein", "folds", "helix"];
  const values = [
    [0.9, 0.2, 0.1, 0.2],
    [0.2, 0.9, 0.7, 0.6],
    [0.1, 0.7, 0.9, 0.5],
    [0.2, 0.6, 0.5, 0.9],
  ];
  const size = 54;
  for (let r = 0; r < words.length; r += 1) {
    addText(s, `hm-row-${r}`, words[r], { left: x - 92, top: y + r * size + 16, width: 84, height: 20 }, {
      fontSize: 13,
      color: MUTED,
      alignment: "right",
    });
    addText(s, `hm-col-${r}`, words[r], { left: x + r * size - 8, top: y - 28, width: 72, height: 20 }, {
      fontSize: 12,
      color: MUTED,
      alignment: "center",
    });
    for (let c = 0; c < words.length; c += 1) {
      const v = values[r][c];
      const fill = v > 0.8 ? VIOLET : v > 0.55 ? BLUE : v > 0.3 ? GOLD : "#E7EFF2";
      addRect(s, `hm-${r}-${c}`, { left: x + c * size, top: y + r * size, width: size - 3, height: size - 3 }, fill, WHITE, 1);
    }
  }
}

function selectedRowText(item) {
  const positive = item.positive.join(" + ");
  const negative = item.negative.length ? ` - ${item.negative.join(" - ")}` : "";
  return `${positive}${negative}`;
}

async function main() {
  const car = await readJson(CAR_STORY);
  const carFull = await readJson(CAR_FULL);
  const bert = await readJson(BERT_JSON);
  assert(bert.analogy_experiment.selected_examples.length >= 10);
  assert(bert.analogy_experiment.anchor_examples.length >= 3);
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });
  await fs.copyFile(FINAL, BACKUP);

  const deck = await PresentationFile.importPptx(await FileBlob.load(STARTER));
  assert.equal(deck.slides.items.length, 30);
  for (let n = 1; n <= 30; n += 1) clearImages(slide(deck, n));

  let s = slide(deck, 1);
  setTitleBody(s, "Representation Learning", "");
  await addImage(s, "representation hero", HERO, { left: 205, top: 80, width: 570, height: 360 }, "cover");
  addText(s, "cover-subtitle", "Useful coordinates for cars, words, proteins, and anything else we want math to understand.", {
    left: 105,
    top: 455,
    width: 760,
    height: 34,
  }, { fontSize: 20, alignment: "center" });
  addText(s, "cover-footer", "High school math summer camp", { left: 105, top: 500, width: 300, height: 20 }, {
    fontSize: 12,
    color: MUTED,
  });

  s = slide(deck, 2);
  setTitleBody(s, "What Numbers Should We Give the World?", "A representation is a choice of coordinates for an object.");
  const world = [["car", "year, mileage, model"], ["sentence", "tokens and context"], ["protein", "amino acid sequence"], ["person", "taste and behavior"]];
  for (let i = 0; i < world.length; i += 1) {
    const x = 92 + i * 205;
    const color = [BLUE, VIOLET, TEAL, GOLD][i];
    addCircle(s, `world-${i}`, x + 60, 245, 42, WHITE, color);
    addText(s, `world-label-${i}`, world[i][0], { left: x + 20, top: 232, width: 80, height: 24 }, {
      fontSize: 15,
      bold: true,
      color,
      alignment: "center",
    });
    addText(s, `world-copy-${i}`, world[i][1], { left: x - 10, top: 310, width: 140, height: 32 }, {
      fontSize: 13,
      color: MUTED,
      alignment: "center",
    });
  }
  addText(s, "world-bottom", "Good coordinates make useful relationships visible.", { left: 190, top: 430, width: 580, height: 36 }, {
    fontSize: 25,
    bold: true,
    alignment: "center",
  });

  s = slide(deck, 3);
  setTitleBody(s, "An approximate syllabus", "What we will talk about:\n\n1  Vectors and similarity\n2  Learning useful coordinates\n3  Car prices and categorical embeddings\n4  Words, ModernBERT, and transformers\n5  Proteins and the live demo");

  s = slide(deck, 4);
  setTitleBody(s, "Vectors", "Collection of numbers with a location and a direction.");
  drawScatter(s, [
    { label: "sedan", x: -1.2, y: -0.3 },
    { label: "wagon", x: -0.9, y: 0.25 },
    { label: "protein", x: 0.8, y: 0.65 },
    { label: "enzyme", x: 1.0, y: 0.2 },
    { label: "word", x: 0.15, y: -0.8 },
    { label: "phrase", x: 0.45, y: -0.55 },
  ], { left: 70, top: 160, width: 480, height: 270 }, ["sedan", "wagon", "protein", "enzyme", "word", "phrase"]);
  addText(s, "vectors-note", "Nearby points often behave similarly. Directions can describe changes.", {
    left: 610,
    top: 230,
    width: 250,
    height: 90,
  }, { fontSize: 24, bold: true });

  s = slide(deck, 5);
  setTitleBody(s, "Dot product", "Vector multiplication that gives a similarity score.");
  addRoundRect(s, "same-dir", { left: 85, top: 180, width: 290, height: 210 }, "none", BLUE, 1.5);
  addText(s, "same-title", "same direction", { left: 110, top: 200, width: 240, height: 25 }, { fontSize: 16, bold: true, color: BLUE, alignment: "center" });
  addLine(s, "same-a", 145, 315, 300, 250, RED, 4);
  addLine(s, "same-b", 145, 315, 330, 285, BLUE, 4);
  addText(s, "same-score", "large positive score", { left: 120, top: 350, width: 220, height: 22 }, { fontSize: 15, bold: true, color: BLUE, alignment: "center" });
  addRoundRect(s, "diff-dir", { left: 465, top: 180, width: 290, height: 210 }, "none", RULE, 1.5);
  addText(s, "diff-title", "different directions", { left: 490, top: 200, width: 240, height: 25 }, { fontSize: 16, bold: true, color: MUTED, alignment: "center" });
  addLine(s, "diff-a", 530, 315, 530, 245, RED, 4);
  addLine(s, "diff-b", 530, 315, 700, 280, BLUE, 4);
  addText(s, "diff-score", "small or negative score", { left: 500, top: 350, width: 220, height: 22 }, { fontSize: 15, bold: true, color: MUTED, alignment: "center" });
  addText(s, "dot-bridge", "Attention asks this question many times.", { left: 735, top: 420, width: 180, height: 45 }, { fontSize: 17, bold: true, alignment: "center" });

  s = slide(deck, 6);
  setTitleBody(s, "Learning Moves the Points", "Training changes coordinates so the error gets smaller.");
  addRect(s, "loss-field", { left: 80, top: 165, width: 470, height: 270 }, WHITE, RULE, 1);
  for (let i = 0; i < 5; i += 1) addCircle(s, `contour-${i}`, 315, 300, 35 + i * 28, "none", "#E1EAED");
  const pathPts = [[430, 210], [390, 238], [356, 264], [335, 290], [318, 300]];
  for (let i = 0; i < pathPts.length - 1; i += 1) addLine(s, `desc-${i}`, pathPts[i][0], pathPts[i][1], pathPts[i + 1][0], pathPts[i + 1][1], RED, 2.5);
  for (let i = 0; i < pathPts.length; i += 1) addCircle(s, `step-${i}`, pathPts[i][0], pathPts[i][1], 5, i === pathPts.length - 1 ? TEAL : RED);
  addText(s, "learn-copy", "The model is learning a coordinate system that makes the task easier.", { left: 610, top: 235, width: 250, height: 115 }, { fontSize: 24, bold: true });

  s = slide(deck, 7);
  setTitleBody(s, "Not Every Number Means Something", "Some numerical encodings create fake structure.");
  addText(s, "fake-values", "red = 1\nblue = 2\ngreen = 3", { left: 140, top: 205, width: 180, height: 120 }, { fontSize: 27, bold: true });
  addText(s, "fake-bad", "This accidentally says:\nblue is twice red\ngreen is greater than blue", { left: 405, top: 205, width: 250, height: 120 }, { fontSize: 21, bold: true, color: RED });
  addText(s, "fake-note", "Feature choice matters too: most cars have 4 wheels, so wheel count carries little signal.", { left: 700, top: 220, width: 170, height: 130 }, { fontSize: 17, color: MUTED });

  s = slide(deck, 8);
  setTitleBody(s, "Token embedding", "In its simplest form, token embedding is a learned lookup table.");
  flowBox(s, "token", "word token", 95, 235, 150, BLUE);
  flowBox(s, "id", "token ID", 315, 235, 150, VIOLET);
  flowBox(s, "row", "embedding row", 535, 235, 170, TEAL);
  flowBox(s, "vector", "vector", 775, 235, 115, GOLD);
  addLine(s, "flow1", 245, 266, 315, 266, MUTED, 2);
  addLine(s, "flow2", 465, 266, 535, 266, MUTED, 2);
  addLine(s, "flow3", 705, 266, 775, 266, MUTED, 2);
  addText(s, "embed-note", "The table starts random. Training changes the rows until useful relationships become easier.", { left: 170, top: 360, width: 620, height: 52 }, { fontSize: 22, bold: true, alignment: "center" });

  s = slide(deck, 9);
  setTitleBody(s, "The Car Example", "Used car prices give a familiar place to test representation choices.");
  metricCard(s, "used listings", "25,000", 95, CYAN);
  metricCard(s, "train / val / test", "70 / 15 / 15", 305, GOLD);
  metricCard(s, "repeated runs", "5 seeds", 515, TEAL);
  metricCard(s, "categorical features", "12", 725, VIOLET);
  addText(s, "car-frame", "Question: what happens when categories become fake numbers, one-hot rows, or learned coordinates?", { left: 145, top: 410, width: 670, height: 46 }, { fontSize: 22, bold: true, alignment: "center" });

  s = slide(deck, 10);
  setTitleBody(s, "Bad Representations Can Break a Model", "The strongest lesson is the gap between fake order and useful coordinates.");
  drawBarChart(s, [
    { label: "Integer IDs", value: car.derived.integer_rmse_vs_embedding_multiple, display: "28.5x", color: RED },
    { label: "One-hot", value: car.summary[1].test_rmse_mean / car.summary[2].test_rmse_mean, display: "1.05x", color: GOLD },
    { label: "Embeddings", value: 1, display: "1.00x", color: BLUE },
  ], 85, 165, 500, 280, 30);
  addText(s, "car-result", `Embeddings had ${car.derived.embedding_rmse_vs_onehot_pct.toFixed(1)}% lower full-test RMSE than one-hot in this run, but MAE was mixed.`, { left: 640, top: 210, width: 230, height: 100 }, { fontSize: 22, bold: true });
  addText(s, "car-source", "Metric: test RMSE relative to learned embeddings, mean over 5 PyTorch runs.", { left: 640, top: 335, width: 230, height: 60 }, { fontSize: 14, color: MUTED });

  s = slide(deck, 11);
  setTitleBody(s, "A Learned Map Is the Payoff", "The embedding table becomes geometry the model can use.");
  drawScatter(s, carFull.embedding_projection.points, { left: 80, top: 145, width: 550, height: 330 }, ["lexus", "acura", "cadillac", "tesla", "bmw", "volvo", "toyota", "mazda", "subaru", "jeep", "honda", "ford"]);
  addText(s, "map-neighbor", "Nearest examples in the projection:\nlexus -> acura, ford\ntesla -> bmw, volvo\ntoyota -> mazda, tesla", { left: 680, top: 190, width: 220, height: 120 }, { fontSize: 17 });
  addText(s, "map-caveat", "A 2D projection is a teaching view; the real embedding vectors have more dimensions.", { left: 680, top: 350, width: 220, height: 70 }, { fontSize: 14, color: MUTED });

  s = slide(deck, 12);
  setTitleBody(s, "The Same Idea Shows Up Everywhere", "Representation learning is a general way to make comparison possible.");
  addText(s, "everywhere-core", "object -> vector -> useful action", { left: 245, top: 165, width: 470, height: 38 }, { fontSize: 26, bold: true, alignment: "center" });
  const actions = [["recommend", "people and items"], ["retrieve", "images and text"], ["translate", "words and context"], ["annotate", "protein sequences"]];
  for (let i = 0; i < actions.length; i += 1) {
    const x = 145 + i * 175;
    const color = [GOLD, BLUE, VIOLET, TEAL][i];
    addCircle(s, `act-${i}`, x + 45, 295, 34, WHITE, color);
    addText(s, `act-label-${i}`, actions[i][0], { left: x - 5, top: 350, width: 100, height: 25 }, { fontSize: 15, bold: true, alignment: "center" });
    addText(s, `act-copy-${i}`, actions[i][1], { left: x - 20, top: 378, width: 130, height: 26 }, { fontSize: 12, color: MUTED, alignment: "center" });
  }

  s = slide(deck, 13);
  setTitleBody(s, "Recommendations", "A recommender learns vectors for people and items.");
  addCircle(s, "student-dot", 210, 280, 20, GOLD, GOLD);
  addText(s, "student-label", "student", { left: 165, top: 240, width: 90, height: 24 }, { fontSize: 18, bold: true, color: GOLD, alignment: "center" });
  const items = [["robotics", 645, 210, BLUE], ["biology", 690, 280, TEAL], ["music", 595, 355, VIOLET], ["coding", 415, 365, RED], ["random", 735, 360, MUTED]];
  for (const item of items) {
    addCircle(s, `rec-${item[0]}`, item[1], item[2], 8, item[3], item[3]);
    addLine(s, `rec-line-${item[0]}`, 210, 280, item[1], item[2], item[3], item[0] === "random" ? 1 : 2);
    addText(s, `rec-label-${item[0]}`, item[0], { left: item[1] + 12, top: item[2] - 10, width: 80, height: 20 }, { fontSize: 12, color: item[3] });
  }
  addText(s, "rec-note", "Learning puts tastes and items into the same space. Similar directions become recommendations.", { left: 120, top: 430, width: 700, height: 40 }, { fontSize: 20, bold: true, alignment: "center" });

  s = slide(deck, 14);
  setTitleBody(s, "Images and Text Can Share a Space", "A search phrase and a picture can be represented by nearby vectors.");
  await addImage(s, "red backpack", BACKPACK, { left: 105, top: 185, width: 185, height: 150 }, "contain");
  flowBox(s, "text-query", "red hiking\nbackpack", 390, 225, 160, BLUE);
  flowBox(s, "shared-space", "shared vector\nspace", 690, 225, 160, TEAL);
  addLine(s, "clip-a", 290, 260, 390, 260, BLUE, 2);
  addLine(s, "clip-b", 550, 260, 690, 260, TEAL, 2);
  addText(s, "clip-note", "Search becomes geometry when pictures and phrases use compatible coordinates.", { left: 160, top: 415, width: 640, height: 36 }, { fontSize: 20, bold: true, alignment: "center" });

  s = slide(deck, 15);
  setTitleBody(s, "Training LLMs", "ModernBERT learns representations by filling in blanks.");
  const fills = bert.fill_mask_examples;
  assert(fills.length >= 3);
  for (let i = 0; i < 3; i += 1) {
    const y = 170 + i * 80;
    addRoundRect(s, `mask-row-${i}`, { left: 105, top: y, width: 590, height: 48 }, WHITE, RULE, 1);
    addText(s, `mask-prompt-${i}`, fills[i].prompt, { left: 125, top: y + 13, width: 460, height: 20 }, { fontSize: 15, bold: true });
    addRoundRect(s, `mask-answer-${i}`, { left: 595, top: y + 9, width: 80, height: 30 }, VIOLET, VIOLET, 1);
    addText(s, `mask-answer-text-${i}`, fills[i].predictions[0].token, { left: 600, top: y + 16, width: 70, height: 16 }, { fontSize: 12, bold: true, color: WHITE, alignment: "center" });
  }
  addText(s, "mask-note", "Prediction forces grammar, facts, and meaning into vectors.", { left: 735, top: 230, width: 155, height: 88 }, { fontSize: 19, bold: true, alignment: "center" });

  s = slide(deck, 16);
  setTitleBody(s, "Token Embedding", "A token starts as an ID. The model looks up a row in a learned matrix.");
  flowBox(s, "onehot", "one-hot ID", 120, 245, 145, BLUE);
  flowBox(s, "matrix", "embedding\nmatrix", 405, 230, 160, VIOLET);
  flowBox(s, "embedding", "input\nvector", 705, 245, 145, TEAL);
  addLine(s, "emb-line1", 265, 276, 405, 276, MUTED, 2);
  addLine(s, "emb-line2", 565, 276, 705, 276, MUTED, 2);
  addText(s, "emb-math", "This is the raw input-embedding table used for the arithmetic demo.", { left: 210, top: 385, width: 540, height: 34 }, { fontSize: 20, bold: true, alignment: "center" });

  s = slide(deck, 17);
  setTitleBody(s, "Vector Arithmetic as Top-k Lookup", "Do the arithmetic, then ask which vocabulary vectors are closest.");
  addText(s, "arith-eq", "query = king + woman - man", { left: 150, top: 190, width: 660, height: 46 }, { fontSize: 32, bold: true, alignment: "center" });
  addLine(s, "arith-line", 480, 245, 480, 305, VIOLET, 3);
  addText(s, "arith-top", "top-k nearest words", { left: 350, top: 315, width: 260, height: 26 }, { fontSize: 20, bold: true, color: VIOLET, alignment: "center" });
  const rankItems = ["prince", "king", "queen", "princess", "lady", "woman"];
  for (let i = 0; i < rankItems.length; i += 1) {
    const x = 230 + i * 85;
    const hit = rankItems[i] === "queen";
    addRoundRect(s, `arith-rank-${i}`, { left: x, top: 365, width: 72, height: 34 }, hit ? VIOLET : WHITE, hit ? VIOLET : RULE, 1);
    addText(s, `arith-rank-label-${i}`, rankItems[i], { left: x + 4, top: 374, width: 64, height: 16 }, { fontSize: 12, bold: true, color: hit ? WHITE : INK, alignment: "center" });
  }
  addText(s, "arith-note", "In this run, queen is rank 6 for king + woman - man.", { left: 260, top: 430, width: 440, height: 26 }, { fontSize: 18, bold: true, alignment: "center" });

  s = slide(deck, 18);
  setTitleBody(s, "ModernBERT Analogy Returns", "Best method only: raw, unit, and centered-unit were tried internally.");
  const summary = bert.analogy_experiment.summary;
  addText(s, "bert-summary", `${summary.example_count} analogies   median rank ${summary.median_expected_rank.toFixed(0)}   hit@10 ${(summary.hit_at_10 * 100).toFixed(0)}%   hit@25 ${(summary.hit_at_25 * 100).toFixed(0)}%`, { left: 95, top: 115, width: 770, height: 30 }, { fontSize: 21, bold: true, alignment: "center" });
  const selected = bert.analogy_experiment.selected_examples.slice(0, 9);
  const rows = [["expression", "expected", "rank", "top result"]];
  for (const item of selected) rows.push([selectedRowText(item), item.expected, item.expected_rank, item.top_result]);
  drawTable(s, "bert-table", rows, 70, 170, [390, 145, 75, 155], 29);

  s = slide(deck, 19);
  setTitleBody(s, "ModernBERT Top-k: Capitals", "The nearest-token list often makes the relationship visible.");
  await addImage(s, "modernbert topk capitals", TOPK_CAPITALS, { left: 58, top: 128, width: 844, height: 300 }, "contain");

  s = slide(deck, 20);
  setTitleBody(s, "ModernBERT Top-k: Comparatives", "The expected word is highlighted when it appears in the top-k list.");
  await addImage(s, "modernbert topk comparatives", TOPK_COMPARATIVES, { left: 78, top: 108, width: 804, height: 380 }, "contain");

  s = slide(deck, 21);
  setTitleBody(s, "ModernBERT Local PCA", "Local views are clearer than one global projection.");
  await addImage(s, "modernbert local pca", PCA, { left: 76, top: 86, width: 808, height: 410 }, "contain", {
    left: 0.01,
    top: 0.04,
    right: 0.01,
    bottom: 0.01,
  });

  s = slide(deck, 22);
  setTitleBody(s, "ModernBERT Is Not Classic word2vec", "Input embeddings show useful geometry, but ModernBERT is contextual.");
  const caveats = [
    ["king - royalty -> man", "rank 827", "not in top 50", RED],
    ["king + woman - man -> queen", "rank 6", "in top 50", VIOLET],
    ["queen - royalty -> woman", "rank 16", "in top 50", TEAL],
  ];
  for (let i = 0; i < caveats.length; i += 1) {
    const x = 120 + i * 255;
    addRoundRect(s, `caveat-${i}`, { left: x, top: 185, width: 210, height: 140 }, WHITE, caveats[i][3], 2);
    addText(s, `caveat-eq-${i}`, caveats[i][0], { left: x + 15, top: 208, width: 180, height: 48 }, { fontSize: 16, bold: true, alignment: "center" });
    addText(s, `caveat-rank-${i}`, caveats[i][1], { left: x + 20, top: 268, width: 170, height: 30 }, { fontSize: 22, bold: true, color: caveats[i][3], alignment: "center" });
    addText(s, `caveat-note-${i}`, caveats[i][2], { left: x + 30, top: 305, width: 150, height: 20 }, { fontSize: 13, color: MUTED, alignment: "center" });
  }
  addText(s, "caveat-bottom", "The teaching point is honest: vector arithmetic can work, but model architecture and tokenization matter.", { left: 145, top: 410, width: 670, height: 44 }, { fontSize: 20, bold: true, alignment: "center" });

  s = slide(deck, 23);
  setTitleBody(s, "Meaning Depends on Context", "The same word can get a different vector in a different sentence.");
  addRoundRect(s, "river-bank", { left: 120, top: 205, width: 245, height: 125 }, WHITE, BLUE, 1.5);
  addText(s, "river-text", "river bank\nwater, shore, fishing", { left: 140, top: 242, width: 205, height: 45 }, { fontSize: 20, bold: true, color: BLUE, alignment: "center" });
  addText(s, "same-bank", "same token\nnew context\nnew vector", { left: 410, top: 235, width: 140, height: 80 }, { fontSize: 18, bold: true, color: VIOLET, alignment: "center" });
  addRoundRect(s, "money-bank", { left: 605, top: 205, width: 245, height: 125 }, WHITE, GOLD, 1.5);
  addText(s, "money-text", "bank account\nmoney, teller, savings", { left: 625, top: 242, width: 205, height: 45 }, { fontSize: 20, bold: true, color: GOLD, alignment: "center" });

  s = slide(deck, 24);
  setTitleBody(s, "Transformers Build Contextual Representations", "Each layer updates token vectors by comparing them with other tokens.");
  flowBox(s, "tokens2", "tokens", 100, 250, 120, VIOLET);
  flowBox(s, "vectors2", "vectors", 305, 250, 120, BLUE);
  flowBox(s, "layers2", "repeated\nlayers", 505, 232, 145, GOLD);
  flowBox(s, "context2", "contextual\nvectors", 735, 250, 145, TEAL);
  addLine(s, "tr-a", 220, 281, 305, 281, MUTED, 2);
  addLine(s, "tr-b", 425, 281, 505, 281, MUTED, 2);
  addLine(s, "tr-c", 650, 281, 735, 281, MUTED, 2);
  addText(s, "tr-note", "Stack enough layers and the vector for a word can carry sentence-level meaning.", { left: 175, top: 405, width: 610, height: 34 }, { fontSize: 20, bold: true, alignment: "center" });

  s = slide(deck, 25);
  setTitleBody(s, "Attention Is Similarity at Scale", "Tokens ask dot-product questions about other tokens.");
  drawHeatmap(s, 230, 170);
  addText(s, "att-callout", "A whole sentence becomes a table of comparisons.", { left: 560, top: 205, width: 250, height: 90 }, { fontSize: 24, bold: true, alignment: "center" });
  addText(s, "att-steps", "1. compare token vectors\n2. turn scores into weights\n3. mix information into a new vector", { left: 560, top: 330, width: 265, height: 78 }, { fontSize: 16, color: MUTED });

  s = slide(deck, 26);
  setTitleBody(s, "Proteins Are Sequences Too", "Protein language models apply the same representation idea to amino acid strings.");
  await addImage(s, "protein language model", PROTEIN, { left: 515, top: 115, width: 420, height: 350 }, "contain");
  addText(s, "protein-seq", "MKTAYIAKQRQISFVKSHFSRQ", { left: 80, top: 220, width: 400, height: 36 }, { fontSize: 25, bold: true });
  addText(s, "protein-copy", "Amino acids are symbols. A protein language model turns a sequence into vectors for residues, domains, or the whole protein.", { left: 82, top: 300, width: 390, height: 105 }, { fontSize: 21 });

  s = slide(deck, 27);
  setTitleBody(s, "Protein Models Learn from Sequence", "Prediction over millions of sequences can reveal functional patterns.");
  flowBox(s, "seq", "sequence", 115, 255, 130, TEAL);
  flowBox(s, "plm", "protein\nlanguage model", 315, 240, 170, TEAL);
  flowBox(s, "prot-embed", "embedding", 565, 255, 150, TEAL);
  flowBox(s, "annot", "annotation\nsearch\ndesign", 785, 235, 145, TEAL);
  addLine(s, "plm-a", 245, 286, 315, 286, MUTED, 2);
  addLine(s, "plm-b", 485, 286, 565, 286, MUTED, 2);
  addLine(s, "plm-c", 715, 286, 785, 286, MUTED, 2);
  addText(s, "plm-note", "The model is not given a hand-written biology dictionary. It learns useful coordinates from sequence patterns.", { left: 135, top: 410, width: 690, height: 38 }, { fontSize: 20, bold: true, alignment: "center" });

  s = slide(deck, 28);
  setTitleBody(s, "Live Demo", "Sequence to neighbors, function, and design ideas.");
  flowBox(s, "demo-input", "input\nsequence", 105, 245, 150, TEAL);
  flowBox(s, "demo-vector", "embedding\nvector", 320, 245, 150, BLUE);
  flowBox(s, "demo-neighbors", "nearest\nproteins", 535, 245, 150, VIOLET);
  flowBox(s, "demo-design", "labels and\ndesign ideas", 750, 245, 150, GOLD);
  addLine(s, "demo-a", 255, 276, 320, 276, MUTED, 2);
  addLine(s, "demo-b", 470, 276, 535, 276, MUTED, 2);
  addLine(s, "demo-c", 685, 276, 750, 276, MUTED, 2);
  addText(s, "demo-question", "Live question: which proteins are nearby, what function is likely, and what design idea follows?", { left: 150, top: 410, width: 660, height: 38 }, { fontSize: 20, bold: true, alignment: "center" });

  s = slide(deck, 29);
  setTitleBody(s, "What Representation Learning Can Do in the Lab", "Vectors become tools for deciding what to test next.");
  const lab = [["annotate", "suggest a function for an unknown sequence", TEAL], ["search", "find related proteins across huge databases", BLUE], ["triage", "prioritize experiments before spending bench time", GOLD], ["design", "propose sequence changes worth testing", RED]];
  for (let i = 0; i < lab.length; i += 1) {
    const x = i % 2 === 0 ? 165 : 535;
    const y = i < 2 ? 185 : 330;
    addRoundRect(s, `lab-${i}`, { left: x, top: y, width: 280, height: 78 }, WHITE, lab[i][2], 1.5);
    addText(s, `lab-head-${i}`, lab[i][0], { left: x + 20, top: y + 24, width: 90, height: 22 }, { fontSize: 16, bold: true, color: lab[i][2], alignment: "center" });
    addText(s, `lab-copy-${i}`, lab[i][1], { left: x + 120, top: y + 17, width: 140, height: 40 }, { fontSize: 12, color: MUTED });
  }

  s = slide(deck, 30);
  setTitleBody(s, "Learn the Coordinates, Then Do the Math", "Cars, words, images, people, and proteins become points in spaces where math can act.");
  const finals = [["compare", BLUE], ["predict", RED], ["search", VIOLET], ["design", TEAL]];
  for (let i = 0; i < finals.length; i += 1) {
    const x = 210 + i * 150;
    addCircle(s, `final-${i}`, x, 250, 45, WHITE, finals[i][1]);
    addText(s, `final-label-${i}`, finals[i][0], { left: x - 44, top: 236, width: 88, height: 24 }, { fontSize: 15, bold: true, color: finals[i][1], alignment: "center" });
  }
  addText(s, "final-questions", "Questions?", { left: 380, top: 395, width: 200, height: 42 }, { fontSize: 34, bold: true, color: RED, alignment: "center" });

  for (let n = 1; n <= 30; n += 1) {
    const one = slide(deck, n);
    const png = await deck.export({ slide: one, format: "png", scale: 1 });
    await fs.writeFile(path.join(PREVIEW_DIR, `slide-${String(n).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await one.export({ format: "layout" });
    await fs.writeFile(path.join(LAYOUT_DIR, `slide-${String(n).padStart(2, "0")}.layout.json`), await layout.text());
  }
  const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(QA_DIR, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));
  const finalPptx = await PresentationFile.exportPptx(deck);
  await finalPptx.save(FINAL);
  console.log(JSON.stringify({ final: FINAL, backup: BACKUP, slides: deck.slides.items.length, qa: QA_DIR }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const TMP_DIR = path.dirname(fileURLToPath(import.meta.url));
const FINAL_PPTX = path.resolve(TMP_DIR, "..", "representation_learning_high_school_talk.pptx");
const PREVIEW_DIR = `${TMP_DIR}\\preview_v2`;
const LAYOUT_DIR = `${TMP_DIR}\\layout_v2`;
const QA_DIR = `${TMP_DIR}\\qa_v2`;
const CAR_JSON = `${TMP_DIR}\\data\\car_story_metrics.json`;
const CAR_SAMPLE_JSON = `${TMP_DIR}\\data\\car_sample_diagnostics.json`;
const BERT_JSON = `${TMP_DIR}\\models\\modernbert_examples.json`;
const HERO_IMAGE = `${TMP_DIR}\\assets\\representation-hero.png`;
const PROTEIN_IMAGE = `${TMP_DIR}\\assets\\protein-language-model.png`;
const BACKPACK_IMAGE = `${TMP_DIR}\\assets\\red-backpack-photo.png`;

const W = 1280;
const H = 720;
const CANVAS = "#FAF7F0";
const INK = "#17201C";
const MUTED = "#5D625F";
const RULE = "#D6D0C4";
const SOFT = "#EFE9DC";
const PANEL = "#FFFFFF";
const ORANGE = "#F05A28";
const BLUE = "#2F6FDB";
const VIOLET = "#6E5AEF";
const TEAL = "#008C7E";
const GOLD = "#DCA62B";
const RED = "#B83B2E";
const FONT = "Aptos";

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
  shape.text.style = {
    fontSize: style.fontSize ?? 24,
    bold: Boolean(style.bold),
    color: style.color ?? INK,
    alignment: style.alignment ?? "left",
    typeface: FONT,
  };
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

function addCircle(slide, name, position, fill = PANEL, stroke = "none", width = 1) {
  return slide.shapes.add({
    geometry: "ellipse",
    name,
    position,
    fill,
    line: stroke === "none" ? noLine() : line(stroke, width),
  });
}

function addRule(slide, left, top, width, color = RULE) {
  slide.shapes.add({
    geometry: "line",
    name: "rule",
    position: { left, top, width, height: 0 },
    fill: "none",
    line: line(color, 1),
  });
}

function addFooter(slide, slideNumber, source = "") {
  addText(slide, `footer-source-${slideNumber}`, source, { left: 54, top: 664, width: 880, height: 24 }, { fontSize: 12, color: MUTED });
  addText(slide, `footer-page-${slideNumber}`, String(slideNumber).padStart(2, "0"), { left: 1168, top: 664, width: 56, height: 24 }, { fontSize: 13, color: MUTED, alignment: "right" });
}

function title(slide, slideNumber, heading, subheading, color = ORANGE) {
  addText(slide, `title-${slideNumber}`, heading, { left: 54, top: 44, width: 1050, height: 74 }, { fontSize: 42, bold: true, color: INK });
  addText(slide, `subtitle-${slideNumber}`, subheading, { left: 56, top: 124, width: 930, height: 46 }, { fontSize: 21, color: MUTED });
  addRule(slide, 54, 184, 1170, RULE);
  addRect(slide, `section-mark-${slideNumber}`, { left: 54, top: 183, width: 90, height: 4 }, color);
}

function sectionPill(slide, name, textValue, x, y, color) {
  addRect(slide, `${name}-rect`, { left: x, top: y, width: 168, height: 34 }, color, "none");
  addText(slide, `${name}-text`, textValue, { left: x + 12, top: y + 6, width: 144, height: 22 }, { fontSize: 14, color: "#FFFFFF", bold: true, alignment: "center" });
}

function metricCard(slide, name, label, value, detail, x, y, color = BLUE) {
  addRect(slide, `${name}-card`, { left: x, top: y, width: 238, height: 136 }, PANEL, RULE, 1);
  addText(slide, `${name}-value`, value, { left: x + 18, top: y + 20, width: 200, height: 44 }, { fontSize: 34, bold: true, color });
  addText(slide, `${name}-label`, label, { left: x + 18, top: y + 70, width: 200, height: 28 }, { fontSize: 18, bold: true, color: INK });
  addText(slide, `${name}-detail`, detail, { left: x + 18, top: y + 100, width: 200, height: 26 }, { fontSize: 14, color: MUTED });
}

function iconLabel(slide, name, label, x, y, color, symbol) {
  addCircle(slide, `${name}-circle`, { left: x, top: y, width: 72, height: 72 }, "#FFFFFF", color, 2);
  addText(slide, `${name}-symbol`, symbol, { left: x, top: y + 14, width: 72, height: 34 }, { fontSize: 26, bold: true, color, alignment: "center" });
  addText(slide, `${name}-label`, label, { left: x - 35, top: y + 88, width: 142, height: 42 }, { fontSize: 18, bold: true, color: INK, alignment: "center" });
}

function connect(slide, from, to, color = MUTED, width = 2) {
  slide.shapes.connect(from, to, {
    kind: "straight",
    fromSide: "right",
    toSide: "left",
    line: line(color, width),
    tail: { type: "arrow", width: "med", length: "med" },
  });
}

function flowBox(slide, name, textValue, x, y, w, h, color = BLUE) {
  const box = addRect(slide, `${name}-box`, { left: x, top: y, width: w, height: h }, PANEL, color, 2);
  addText(slide, `${name}-text`, textValue, { left: x + 14, top: y + h / 2 - 18, width: w - 28, height: 42 }, { fontSize: 20, bold: true, color: INK, alignment: "center" });
  return box;
}

function currency(value) {
  return `$${Math.round(value).toLocaleString("en-US")}`;
}

function pct(value, digits = 1) {
  return `${Number(value).toFixed(digits)}%`;
}

function fixed(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function row(summary, encoding) {
  const found = summary.find((item) => item.encoding === encoding);
  if (!found) {
    throw new Error(`Missing summary row for ${encoding}`);
  }
  return found;
}

function sampleRow(carSample, encoding) {
  const found = carSample.by_encoding[encoding];
  if (!found) {
    throw new Error(`Missing sample diagnostics for ${encoding}`);
  }
  return found;
}

async function imageBytes(path) {
  const bytes = await fs.readFile(path);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function addImage(slide, name, path, position, fit = "cover") {
  slide.images.add({
    name,
    blob: await imageBytes(path),
    contentType: "image/png",
    alt: name,
    fit,
    position,
  });
}

function addSpeakerNotes(presentation) {
  const notes = [
    "Open with the core claim: representation learning gives useful coordinates to things that did not start as numbers. Keep this concrete and energetic.",
    "Ask students what numbers they would assign to a car, a sentence, or a protein. The point is that the choice of coordinates is already part of the model.",
    "Set expectations for the talk: first the math spine, then car categories, then breadth across recommendations, image-text search, language, and proteins.",
    "Connect this to familiar functions. The new piece is that the function has parameters that training can tune.",
    "Define vectors as locations. Students do not need high-dimensional intuition yet; two dimensions are enough to understand distance and direction.",
    "Use alignment as the bridge to dot products and attention. Similar vectors pointing the same way get large scores.",
    "Explain gradient descent as moving coordinates and knobs so the useful answer becomes easier. This is the moment where training becomes geometry.",
    "Use deliberately bad category IDs. Make clear that assigning red=1 and blue=2 does not mean blue is twice red.",
    "Compare the three representations as design choices. Integer IDs are compact but misleading, one-hot is honest but sparse, embeddings are learned coordinates.",
    "Introduce the car case study as a concrete experiment. Do not frame it as the whole argument; it is a small laboratory for representation choice.",
    "Emphasize the strong result: fake order breaks the model. One-hot and embeddings both work well; the embedding map is useful but the metric comparison is mixed.",
    "Explain the projection with care. The map is learned for a prediction task and compressed to two dimensions for viewing.",
    "Widen the lens. Representation learning is the same move behind recommendations, retrieval, language, biology, and many other systems.",
    "A recommender learns coordinates for people and items. Nearby directions mean a likely match, not magic.",
    "Show multimodal search as a powerful breadth example: images and text can live in a shared vector space.",
    "ModernBERT fills blanks by learning language representations. Keep this as a clean example of prediction creating meaning.",
    "Use bank as the memorable context example. The spelling stays the same, but the vector changes when the surrounding words change.",
    "Describe a transformer layer as repeated vector updating: compare, mix information, update the representation.",
    "Attention is a scaled-up dot-product idea. Keep it high-level and visual.",
    "Transition to biology. Proteins are sequences over a small alphabet, so the language-model idea naturally transfers.",
    "Protein language models learn patterns from sequence data that can support annotation, similarity search, mutation reasoning, and design.",
    "This is the live demo bridge. Name the input, embedding, neighbors, predicted function labels, and design question.",
    "Make the applications concrete: annotation, search, design triage, and experiment prioritization.",
    "Close on the unifying idea: once a thing has useful coordinates, math can compare it, organize it, predict from it, and help design with it.",
  ];
  if (notes.length !== presentation.slides.items.length) {
    throw new Error(`Expected ${presentation.slides.items.length} notes, got ${notes.length}`);
  }
  for (let idx = 0; idx < presentation.slides.items.length; idx += 1) {
    const slide = presentation.slides.items[idx];
    slide.speakerNotes.textFrame.setText(notes[idx]);
    slide.speakerNotes.setVisible(true);
  }
}

async function buildDeck(car, bert, carSample) {
  const deck = Presentation.create({ slideSize: { width: W, height: H } });
  const summary = car.summary;
  const integer = row(summary, "integer IDs");
  const onehot = row(summary, "one-hot");
  const embed = row(summary, "learned embeddings");
  const sampleInteger = sampleRow(carSample, "integer IDs");
  const sampleOnehot = sampleRow(carSample, "one-hot");
  const sampleEmbed = sampleRow(carSample, "learned embeddings");
  if (sampleInteger.sample_count !== 1250 || sampleOnehot.sample_count !== 1250 || sampleEmbed.sample_count !== 1250) {
    throw new Error("Expected 1,250 stored prediction samples for each encoding");
  }
  const derived = car.derived;
  const projection = JSON.parse(JSON.stringify(car.embedding_projection_neighbors));
  const fillMask = bert.fill_mask_examples;

  let n = 1;
  function slide() {
    const s = deck.slides.add();
    s.background.fill = CANVAS;
    return s;
  }

  let s = slide();
  await addImage(s, "representation-hero", HERO_IMAGE, { left: 410, top: 0, width: 870, height: 720 });
  addRect(s, "cover-panel", { left: 0, top: 0, width: 548, height: 720 }, CANVAS, "none");
  addText(s, "cover-title", "Representation Learning", { left: 58, top: 82, width: 640, height: 160 }, { fontSize: 68, bold: true, color: INK });
  addText(s, "cover-subtitle", "Useful coordinates for cars, words, proteins, and anything else we want math to understand.", { left: 62, top: 276, width: 450, height: 132 }, { fontSize: 26, color: MUTED });
  addText(s, "cover-bottom", "High school math summer camp", { left: 62, top: 600, width: 380, height: 34 }, { fontSize: 18, color: MUTED });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "What Numbers Should We Give the World?", "A representation is a choice of coordinates for an object.", ORANGE);
  const worldItems = [
    ["car", "year, mileage, model", BLUE],
    ["sentence", "tokens and context", VIOLET],
    ["protein", "amino acid sequence", TEAL],
    ["person", "taste and behavior", GOLD],
  ];
  for (let idx = 0; idx < worldItems.length; idx += 1) {
    const item = worldItems[idx];
    const x = 94 + idx * 290;
    addCircle(s, `world-${idx}`, { left: x, top: 270, width: 118, height: 118 }, "#FFFFFF", item[2], 3);
    addText(s, `world-name-${idx}`, item[0], { left: x - 10, top: 305, width: 138, height: 34 }, { fontSize: 25, bold: true, color: item[2], alignment: "center" });
    addText(s, `world-detail-${idx}`, item[1], { left: x - 40, top: 420, width: 198, height: 60 }, { fontSize: 18, color: MUTED, alignment: "center" });
  }
  addText(s, "coordinate-promise", "Good coordinates make useful relationships visible.", { left: 218, top: 556, width: 846, height: 52 }, { fontSize: 32, bold: true, color: INK, alignment: "center" });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "A Roadmap from Math to Biology", "The talk follows one idea through several domains.", ORANGE);
  const stops = [
    ["1", "vectors", ORANGE],
    ["2", "training", ORANGE],
    ["3", "car categories", BLUE],
    ["4", "language", VIOLET],
    ["5", "proteins", TEAL],
  ];
  let previous = null;
  for (let idx = 0; idx < stops.length; idx += 1) {
    const x = 92 + idx * 232;
    const node = addCircle(s, `road-node-${idx}`, { left: x, top: 308, width: 92, height: 92 }, "#FFFFFF", stops[idx][2], 3);
    addText(s, `road-num-${idx}`, stops[idx][0], { left: x, top: 326, width: 92, height: 34 }, { fontSize: 28, bold: true, color: stops[idx][2], alignment: "center" });
    addText(s, `road-label-${idx}`, stops[idx][1], { left: x - 50, top: 430, width: 192, height: 44 }, { fontSize: 20, bold: true, color: INK, alignment: "center" });
    if (previous) {
      connect(s, previous, node, MUTED, 2);
    }
    previous = node;
  }
  addText(s, "road-note", "We start with familiar math, then use it to make sense of modern machine learning.", { left: 180, top: 540, width: 920, height: 54 }, { fontSize: 26, color: MUTED, alignment: "center" });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "A Model Is a Function With Knobs", "Machine learning tunes the knobs so predictions become useful.", ORANGE);
  const f1 = flowBox(s, "input", "input x", 112, 310, 190, 92, ORANGE);
  const f2 = flowBox(s, "function", "function f", 410, 286, 230, 140, ORANGE);
  const f3 = flowBox(s, "prediction", "prediction y", 748, 310, 218, 92, ORANGE);
  connect(s, f1, f2, MUTED, 2);
  connect(s, f2, f3, MUTED, 2);
  addText(s, "knobs", "parameters are tunable numbers", { left: 408, top: 470, width: 240, height: 52 }, { fontSize: 21, bold: true, color: ORANGE, alignment: "center" });
  addText(s, "function-example", "car details -> price\nsentence -> missing word\nprotein sequence -> function label", { left: 990, top: 276, width: 230, height: 170 }, { fontSize: 22, color: MUTED });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "Vectors Are Locations", "Once an object becomes a vector, geometry becomes available.", ORANGE);
  addRect(s, "vector-plane", { left: 82, top: 220, width: 640, height: 400 }, "#FFFFFF", RULE, 1);
  addRule(s, 146, 418, 510, "#BEB6A8");
  s.shapes.add({ geometry: "line", name: "axis-y", position: { left: 400, top: 260, width: 0, height: 312 }, fill: "none", line: line("#BEB6A8", 1) });
  const vectorPoints = [
    [230, 470, BLUE, "sedan"],
    [302, 442, BLUE, "wagon"],
    [520, 338, TEAL, "protein"],
    [565, 366, TEAL, "enzyme"],
    [262, 315, VIOLET, "word"],
    [322, 300, VIOLET, "phrase"],
  ];
  for (let idx = 0; idx < vectorPoints.length; idx += 1) {
    const p = vectorPoints[idx];
    addCircle(s, `vec-dot-${idx}`, { left: p[0], top: p[1], width: 18, height: 18 }, p[2], "none");
    addText(s, `vec-label-${idx}`, p[3], { left: p[0] + 20, top: p[1] - 4, width: 110, height: 26 }, { fontSize: 16, color: MUTED });
  }
  addText(s, "vector-copy", "Distances and directions become clues:\nnearby points often behave similarly, and directions can encode useful changes.", { left: 790, top: 284, width: 360, height: 170 }, { fontSize: 28, color: INK });
  addText(s, "vector-foot", "Real models usually use hundreds or thousands of dimensions. The 2D picture is the intuition.", { left: 790, top: 496, width: 360, height: 70 }, { fontSize: 20, color: MUTED });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "Dot Products Measure Alignment", "A dot product gives a fast similarity score.", ORANGE);
  addRect(s, "dot-good-panel", { left: 78, top: 232, width: 440, height: 330 }, "#FFFFFF", BLUE, 2);
  addRect(s, "dot-low-panel", { left: 568, top: 232, width: 440, height: 330 }, "#FFFFFF", RULE, 1);
  addText(s, "dot-good-title", "same direction", { left: 104, top: 258, width: 220, height: 34 }, { fontSize: 24, bold: true, color: BLUE });
  addText(s, "dot-low-title", "different directions", { left: 594, top: 258, width: 260, height: 34 }, { fontSize: 24, bold: true, color: MUTED });
  const dotOriginA = { x: 192, y: 334 };
  const dotOriginB = { x: 684, y: 334 };
  s.shapes.add({ geometry: "line", name: "dot-good-a", position: { left: dotOriginA.x, top: dotOriginA.y, width: 220, height: 110 }, fill: "none", line: line(BLUE, 6) });
  s.shapes.add({ geometry: "line", name: "dot-good-b", position: { left: dotOriginA.x, top: dotOriginA.y, width: 252, height: 132 }, fill: "none", line: line(ORANGE, 6) });
  s.shapes.add({ geometry: "line", name: "dot-low-a", position: { left: dotOriginB.x, top: dotOriginB.y, width: 222, height: 108 }, fill: "none", line: line(BLUE, 6) });
  s.shapes.add({ geometry: "line", name: "dot-low-b", position: { left: dotOriginB.x, top: dotOriginB.y, width: 0, height: 144 }, fill: "none", line: line(ORANGE, 6) });
  addCircle(s, "dot-good-origin", { left: dotOriginA.x - 8, top: dotOriginA.y - 8, width: 16, height: 16 }, INK, "none");
  addCircle(s, "dot-low-origin", { left: dotOriginB.x - 8, top: dotOriginB.y - 8, width: 16, height: 16 }, INK, "none");
  addCircle(s, "dot-good-end-a", { left: 404, top: 436, width: 18, height: 18 }, BLUE, "none");
  addCircle(s, "dot-good-end-b", { left: 436, top: 458, width: 18, height: 18 }, ORANGE, "none");
  addCircle(s, "dot-low-end-a", { left: 898, top: 434, width: 18, height: 18 }, BLUE, "none");
  addCircle(s, "dot-low-end-b", { left: 676, top: 470, width: 18, height: 18 }, ORANGE, "none");
  addText(s, "dot-good-score", "large positive score", { left: 126, top: 504, width: 330, height: 34 }, { fontSize: 25, bold: true, color: BLUE, alignment: "center" });
  addText(s, "dot-low-score", "small or negative score", { left: 616, top: 504, width: 330, height: 34 }, { fontSize: 25, bold: true, color: MUTED, alignment: "center" });
  addText(s, "dot-eq", "dot product =\nadd matching\npieces", { left: 1038, top: 270, width: 176, height: 126 }, { fontSize: 29, bold: true, color: INK, alignment: "center" });
  addText(s, "dot-bridge", "Attention repeats this comparison many times.", { left: 1038, top: 524, width: 176, height: 86 }, { fontSize: 23, bold: true, color: ORANGE, alignment: "center" });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "Learning Moves the Points", "Gradient descent changes coordinates so error gets smaller.", ORANGE);
  addRect(s, "loss-frame", { left: 90, top: 222, width: 640, height: 398 }, "#FFFFFF", RULE, 1);
  const rings = [
    [208, 254, 440, 282, "#CCC4B8"],
    [260, 288, 338, 214, "#B6AA9B"],
    [320, 326, 220, 140, "#958676"],
    [374, 360, 110, 72, ORANGE],
  ];
  for (let idx = 0; idx < rings.length; idx += 1) {
    const r = rings[idx];
    addCircle(s, `loss-ring-${idx}`, { left: r[0], top: r[1], width: r[2], height: r[3] }, "none", r[4], 1);
  }
  const steps = [
    [606, 292],
    [552, 324],
    [506, 350],
    [466, 374],
    [430, 396],
  ];
  for (let idx = 0; idx < steps.length; idx += 1) {
    addCircle(s, `loss-step-${idx}`, { left: steps[idx][0], top: steps[idx][1], width: 16, height: 16 }, idx === steps.length - 1 ? ORANGE : INK);
    if (idx < steps.length - 1) {
      s.shapes.add({ geometry: "line", name: `loss-path-${idx}`, position: { left: steps[idx][0] + 8, top: steps[idx][1] + 8, width: steps[idx + 1][0] - steps[idx][0], height: steps[idx + 1][1] - steps[idx][1] }, fill: "none", line: line(ORANGE, 2) });
    }
  }
  addText(s, "loss-x", "parameter 1", { left: 320, top: 574, width: 200, height: 24 }, { fontSize: 16, color: MUTED, alignment: "center" });
  addText(s, "loss-y", "parameter 2", { left: 120, top: 386, width: 110, height: 24 }, { fontSize: 16, color: MUTED, alignment: "center" });
  addText(s, "learning-points", "Training changes parameters and embeddings together.\n\nThe model is not just fitting a line. It is learning a coordinate system that makes the task easier.", { left: 800, top: 270, width: 360, height: 240 }, { fontSize: 27, color: INK });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "Not Every Number Means Something", "Some numerical encodings create fake structure.", BLUE);
  addText(s, "bad-ids", "red = 1\nblue = 2\ngreen = 3", { left: 110, top: 270, width: 260, height: 160 }, { fontSize: 34, bold: true, color: INK });
  addText(s, "bad-math", "This accidentally says:\nblue is twice red\ngreen is greater than blue", { left: 450, top: 272, width: 360, height: 170 }, { fontSize: 29, color: RED });
  addText(s, "wheel", "Feature choice matters too:\nmost cars have 4 wheels, so wheel count carries almost no signal.", { left: 860, top: 270, width: 310, height: 180 }, { fontSize: 26, color: MUTED });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "Embeddings Are Learned Coordinates", "A category gets a trainable vector instead of a fake number.", BLUE);
  const encA = flowBox(s, "id", "integer ID\ncompact, fake order", 88, 292, 250, 120, RED);
  const encB = flowBox(s, "hot", "one-hot\nhonest, sparse", 514, 292, 250, 120, GOLD);
  const encC = flowBox(s, "emb", "embedding\ncompact, trainable", 940, 292, 250, 120, BLUE);
  addText(s, "enc-a", "[7]", { left: 146, top: 452, width: 130, height: 44 }, { fontSize: 30, bold: true, color: RED, alignment: "center" });
  addText(s, "enc-b", "[0, 0, 1, 0, ...]", { left: 536, top: 452, width: 210, height: 44 }, { fontSize: 25, bold: true, color: GOLD, alignment: "center" });
  addText(s, "enc-c", "[0.2, -1.4, 0.7, ...]", { left: 942, top: 452, width: 250, height: 44 }, { fontSize: 25, bold: true, color: BLUE, alignment: "center" });
  connect(s, encA, encB, MUTED, 1);
  connect(s, encB, encC, MUTED, 1);
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "The Car Example: Structure, Not a Scoreboard", "Used car prices give a familiar place to test representation choices.", BLUE);
  metricCard(s, "rows", "examples", car.rows_used.toLocaleString("en-US"), "used car listings", 70, 254, BLUE);
  metricCard(s, "split", "split", "70 / 15 / 15", "train / validation / test", 360, 254, GOLD);
  metricCard(s, "seeds", "repeatability", "5 seeds", "same held-out test split", 650, 254, TEAL);
  metricCard(s, "cats", "categories", `${car.categorical_features.length}`, "make, model, color, trim...", 940, 254, VIOLET);
  addText(s, "case-framing", "The question is not whether one small neural net wins forever. The question is how representation changes what the model can learn.", { left: 130, top: 520, width: 1020, height: 66 }, { fontSize: 26, bold: true, color: INK, alignment: "center" });
  addFooter(s, n, "Dataset: gsv24/car-price");

  n += 1;
  s = slide();
  title(s, n, "Bad Representations Can Break a Model", "The strongest lesson is the gap between fake order and useful coordinates.", BLUE);
  const relInteger = derived.integer_rmse_vs_embedding_multiple;
  const relOnehot = onehot.test_rmse_mean / embed.test_rmse_mean;
  s.charts.add("bar", {
    position: { left: 78, top: 250, width: 560, height: 300 },
    categories: ["Integer IDs", "One-hot", "Embeddings"],
    series: [{
      name: "Relative RMSE",
      values: [Number(relInteger.toFixed(1)), Number(relOnehot.toFixed(2)), 1],
      fill: BLUE,
      points: [{ idx: 0, fill: RED }, { idx: 1, fill: GOLD }, { idx: 2, fill: BLUE }],
    }],
    hasLegend: false,
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 42 },
    yAxis: { title: "RMSE relative to embeddings", majorGridlines: line("#E2DACD", 1), textStyle: { fill: MUTED, fontSize: 13 } },
    xAxis: { textStyle: { fill: INK, fontSize: 14 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: INK, fontSize: 13, bold: true } },
  });
  addText(s, "metric-table", `Encoding           RMSE        MAE        R2\nInteger IDs     ${currency(integer.test_rmse_mean)}   ${currency(integer.test_mae_mean)}   < 0\nOne-hot           ${currency(onehot.test_rmse_mean)}     ${currency(onehot.test_mae_mean)}   ${fixed(onehot.test_r2_mean)}\nEmbeddings     ${currency(embed.test_rmse_mean)}     ${currency(embed.test_mae_mean)}   ${fixed(embed.test_r2_mean)}`, { left: 710, top: 236, width: 490, height: 166 }, { fontSize: 21, bold: true, color: INK });
  addText(s, "metric-interpretation", `Embeddings had ${pct(derived.embedding_rmse_vs_onehot_pct)} lower full-test RMSE than one-hot, but ${pct(derived.embedding_mae_vs_onehot_pct)} higher MAE.`, { left: 710, top: 420, width: 470, height: 64 }, { fontSize: 20, color: MUTED });
  addText(s, "seed-wins", `Seed wins vs one-hot: RMSE ${car.win_counts.rmse}/5, MAE ${car.win_counts.mae}/5, R2 ${car.win_counts.r2}/5`, { left: 710, top: 492, width: 470, height: 28 }, { fontSize: 17, bold: true, color: BLUE });
  addText(s, "sample-bars-title", "Stored prediction sample within $5k", { left: 710, top: 532, width: 420, height: 26 }, { fontSize: 18, bold: true, color: INK });
  const sampleBar = (name, label, value, y, color) => {
    addText(s, `${name}-label`, label, { left: 710, top: y, width: 120, height: 24 }, { fontSize: 16, bold: true, color: INK });
    addRect(s, `${name}-rail`, { left: 836, top: y + 3, width: 232, height: 16 }, SOFT, "none");
    addRect(s, `${name}-bar`, { left: 836, top: y + 3, width: 232 * value / 100, height: 16 }, color, "none");
    addText(s, `${name}-value`, pct(value, 0), { left: 1084, top: y - 1, width: 70, height: 24 }, { fontSize: 16, bold: true, color, alignment: "right" });
  };
  sampleBar("sample-int", "Integer", sampleInteger.within_5000_pct, 566, RED);
  sampleBar("sample-onehot", "One-hot", sampleOnehot.within_5000_pct, 590, GOLD);
  sampleBar("sample-embed", "Embedding", sampleEmbed.within_5000_pct, 614, BLUE);
  addFooter(s, n, "Metrics: mean over 5 PyTorch runs on held-out test rows");

  n += 1;
  s = slide();
  title(s, n, "A Learned Map Is the Payoff", "The embedding table becomes geometry the model can use.", BLUE);
  addRect(s, "map-frame", { left: 76, top: 218, width: 650, height: 410 }, "#FFFFFF", RULE, 1);
  const points = car.embedding_projection_neighbors;
  const labels = [
    ["lexus", 236, 302, BLUE],
    ["acura", 300, 278, BLUE],
    ["cadillac", 250, 220, BLUE],
    ["tesla", 548, 408, TEAL],
    ["bmw", 498, 520, TEAL],
    ["volvo", 610, 470, TEAL],
    ["toyota", 592, 292, GOLD],
    ["mazda", 520, 245, GOLD],
    ["subaru", 330, 542, VIOLET],
    ["jeep", 378, 510, VIOLET],
    ["honda", 290, 470, INK],
  ];
  for (let idx = 0; idx < labels.length; idx += 1) {
    const p = labels[idx];
    addCircle(s, `map-${idx}`, { left: p[1], top: p[2], width: 16, height: 16 }, p[3], "none");
    addText(s, `map-label-${idx}`, p[0], { left: p[1] + 18, top: p[2] - 5, width: 110, height: 24 }, { fontSize: 15, color: MUTED });
  }
  addText(s, "neighbor-list", `Nearest examples in the 2D projection:\n\nlexus -> acura, ford, cadillac\ntesla -> bmw, volvo, toyota\ntoyota -> mazda, tesla, volvo`, { left: 790, top: 258, width: 390, height: 170 }, { fontSize: 23, color: INK });
  addText(s, "map-caveat", "This is task-learned geometry, not a claim that the model understands car brands like a person.", { left: 790, top: 482, width: 382, height: 82 }, { fontSize: 22, color: MUTED });
  addFooter(s, n, "Projection: PCA of learned make embeddings");

  n += 1;
  s = slide();
  title(s, n, "The Same Idea Shows Up Everywhere", "Representation learning is a general way to make comparison possible.", GOLD);
  addText(s, "breadth-core", "object -> vector -> useful action", { left: 372, top: 220, width: 536, height: 44 }, { fontSize: 30, bold: true, color: INK, alignment: "center" });
  const breadth = [
    ["recommend", "people and items", GOLD],
    ["retrieve", "images and text", BLUE],
    ["translate", "words and context", VIOLET],
    ["annotate", "protein sequences", TEAL],
    ["cluster", "cells or patients", ORANGE],
    ["design", "new candidates", INK],
  ];
  for (let idx = 0; idx < breadth.length; idx += 1) {
    const col = idx % 3;
    const rowIdx = Math.floor(idx / 3);
    const x = 104 + col * 360;
    const y = 292 + rowIdx * 152;
    iconLabel(s, `breadth-${idx}`, breadth[idx][0], x, y, breadth[idx][2], String(idx + 1));
    addText(s, `breadth-detail-${idx}`, breadth[idx][1], { left: x + 100, top: y + 14, width: 190, height: 52 }, { fontSize: 20, color: MUTED });
  }
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "Recommendations: Taste as Geometry", "A recommender learns vectors for people and items.", GOLD);
  addText(s, "person-label", "student", { left: 136, top: 292, width: 170, height: 36 }, { fontSize: 28, bold: true, color: GOLD, alignment: "center" });
  addCircle(s, "person-vector", { left: 170, top: 342, width: 92, height: 92 }, GOLD, "none");
  const itemDots = [
    [600, 260, BLUE, "robotics"],
    [728, 318, TEAL, "biology"],
    [828, 422, VIOLET, "music"],
    [640, 502, ORANGE, "coding"],
    [970, 282, MUTED, "random"],
  ];
  for (let idx = 0; idx < itemDots.length; idx += 1) {
    const p = itemDots[idx];
    addCircle(s, `rec-dot-${idx}`, { left: p[0], top: p[1], width: 34, height: 34 }, p[2], "none");
    addText(s, `rec-label-${idx}`, p[3], { left: p[0] + 44, top: p[1] + 2, width: 130, height: 28 }, { fontSize: 18, color: MUTED });
    if (idx < 4) {
      s.shapes.add({ geometry: "line", name: `rec-sim-${idx}`, position: { left: 216, top: 388, width: p[0] - 216, height: p[1] - 388 }, fill: "none", line: line(p[2], idx === 1 ? 4 : 2) });
    }
  }
  addText(s, "recommend-copy", "Learning puts tastes and items into the same space. Similar directions become recommendations.", { left: 110, top: 518, width: 430, height: 88 }, { fontSize: 26, color: INK });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "Images and Text Can Share a Space", "A search phrase and a picture can be represented by nearby vectors.", BLUE);
  addRect(s, "image-photo-frame", { left: 82, top: 242, width: 300, height: 226 }, "#FFFFFF", BLUE, 2);
  await addImage(s, "backpack-photo", BACKPACK_IMAGE, { left: 102, top: 258, width: 260, height: 170 }, "contain");
  addText(s, "image-photo-label", "picture vector", { left: 132, top: 430, width: 200, height: 28 }, { fontSize: 18, bold: true, color: BLUE, alignment: "center" });
  addRect(s, "text-query-frame", { left: 456, top: 272, width: 292, height: 116 }, "#FFFFFF", TEAL, 2);
  addText(s, "text-query", "\"red hiking backpack\"", { left: 486, top: 312, width: 232, height: 34 }, { fontSize: 24, bold: true, color: TEAL, alignment: "center" });
  addText(s, "text-query-label", "text vector", { left: 506, top: 406, width: 190, height: 28 }, { fontSize: 18, bold: true, color: TEAL, alignment: "center" });
  addRect(s, "shared-space-frame", { left: 844, top: 242, width: 320, height: 250 }, "#FFFFFF", GOLD, 2);
  addText(s, "shared-space-title", "shared vector space", { left: 884, top: 268, width: 240, height: 28 }, { fontSize: 20, bold: true, color: GOLD, alignment: "center" });
  addCircle(s, "clip-photo-dot", { left: 972, top: 352, width: 22, height: 22 }, BLUE);
  addCircle(s, "clip-text-dot", { left: 1016, top: 372, width: 22, height: 22 }, TEAL);
  addCircle(s, "clip-other-dot-1", { left: 904, top: 416, width: 16, height: 16 }, MUTED);
  addCircle(s, "clip-other-dot-2", { left: 1094, top: 330, width: 16, height: 16 }, MUTED);
  addText(s, "shared-nearby", "nearby = likely match", { left: 898, top: 430, width: 220, height: 28 }, { fontSize: 18, bold: true, color: INK, alignment: "center" });
  s.shapes.add({ geometry: "rightArrow", name: "image-to-space", position: { left: 378, top: 330, width: 70, height: 24 }, fill: BLUE, line: noLine() });
  s.shapes.add({ geometry: "rightArrow", name: "text-to-space", position: { left: 752, top: 330, width: 84, height: 24 }, fill: TEAL, line: noLine() });
  addText(s, "clip-note", "Search becomes geometry when pictures and phrases use compatible coordinates.", { left: 250, top: 542, width: 780, height: 54 }, { fontSize: 26, color: INK, alignment: "center" });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "Language Models Learn by Prediction", "ModernBERT learns representations by filling in blanks.", VIOLET);
  for (let idx = 0; idx < 3; idx += 1) {
    const y = 242 + idx * 120;
    const example = fillMask[idx];
    const answer = example.predictions[0];
    addRect(s, `mask-card-${idx}`, { left: 82, top: y, width: 812, height: 86 }, "#FFFFFF", RULE, 1);
    addText(s, `mask-sentence-${idx}`, example.prompt, { left: 112, top: y + 18, width: 610, height: 34 }, { fontSize: 24, bold: true, color: INK });
    addRect(s, `mask-answer-box-${idx}`, { left: 740, top: y + 20, width: 120, height: 44 }, VIOLET, "none");
    addText(s, `mask-answer-${idx}`, answer.token, { left: 748, top: y + 29, width: 104, height: 26 }, { fontSize: 18, bold: true, color: "#FFFFFF", alignment: "center" });
  }
  addText(s, "mask-note", "Prediction forces the model to store grammar, facts, and meaning in vectors.", { left: 926, top: 296, width: 270, height: 130 }, { fontSize: 25, color: INK });
  addFooter(s, n, "Model: answerdotai/ModernBERT-base");

  n += 1;
  s = slide();
  title(s, n, "Meaning Depends on Context", "The same word can get a different vector in a different sentence.", VIOLET);
  addRect(s, "bank-left", { left: 100, top: 250, width: 390, height: 230 }, "#FFFFFF", BLUE, 2);
  addRect(s, "bank-right", { left: 790, top: 250, width: 390, height: 230 }, "#FFFFFF", GOLD, 2);
  addText(s, "river-bank", "river bank\nwater, shore, fishing", { left: 136, top: 312, width: 318, height: 92 }, { fontSize: 32, bold: true, color: BLUE, alignment: "center" });
  addText(s, "money-bank", "bank account\nmoney, teller, savings", { left: 826, top: 312, width: 318, height: 92 }, { fontSize: 32, bold: true, color: GOLD, alignment: "center" });
  addText(s, "same-token", "same token\nnew context\nnew vector", { left: 528, top: 312, width: 224, height: 100 }, { fontSize: 28, bold: true, color: VIOLET, alignment: "center" });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "Transformers Build Contextual Representations", "Each layer updates vectors by comparing tokens with other tokens.", VIOLET);
  const t1 = flowBox(s, "tok", "tokens", 82, 302, 160, 98, VIOLET);
  const t2 = flowBox(s, "vec", "vectors", 316, 302, 170, 98, VIOLET);
  const t3 = addRect(s, "layer-stack", { left: 568, top: 256, width: 230, height: 190 }, "#FFFFFF", VIOLET, 2);
  const t4 = flowBox(s, "ctx", "contextual\nvectors", 882, 302, 190, 98, VIOLET);
  connect(s, t1, t2, MUTED, 2);
  connect(s, t2, t3, MUTED, 2);
  connect(s, t3, t4, MUTED, 2);
  addText(s, "tok-detail", "the | protein | folds", { left: 52, top: 424, width: 220, height: 28 }, { fontSize: 18, bold: true, color: MUTED, alignment: "center" });
  const vectorBars = [BLUE, VIOLET, GOLD, TEAL, ORANGE];
  for (let idx = 0; idx < vectorBars.length; idx += 1) {
    addRect(s, `vec-bar-${idx}`, { left: 344 + idx * 22, top: 426, width: 14, height: 54 - idx * 4 }, vectorBars[idx], "none");
  }
  addText(s, "vec-detail", "numbers for each token", { left: 290, top: 488, width: 220, height: 28 }, { fontSize: 18, color: MUTED, alignment: "center" });
  addText(s, "layer-title", "repeated layer", { left: 594, top: 278, width: 178, height: 32 }, { fontSize: 24, bold: true, color: VIOLET, alignment: "center" });
  addRect(s, "layer-compare", { left: 612, top: 324, width: 142, height: 28 }, SOFT, VIOLET, 1);
  addRect(s, "layer-mix", { left: 612, top: 362, width: 142, height: 28 }, SOFT, VIOLET, 1);
  addRect(s, "layer-update", { left: 612, top: 400, width: 142, height: 28 }, SOFT, VIOLET, 1);
  addText(s, "layer-compare-label", "compare", { left: 624, top: 327, width: 118, height: 22 }, { fontSize: 18, bold: true, color: INK, alignment: "center" });
  addText(s, "layer-mix-label", "mix", { left: 624, top: 365, width: 118, height: 22 }, { fontSize: 18, bold: true, color: INK, alignment: "center" });
  addText(s, "layer-update-label", "update", { left: 624, top: 403, width: 118, height: 22 }, { fontSize: 18, bold: true, color: INK, alignment: "center" });
  addText(s, "ctx-detail", "same tokens,\nnew vectors", { left: 884, top: 426, width: 188, height: 58 }, { fontSize: 19, bold: true, color: VIOLET, alignment: "center" });
  addText(s, "transformer-note", "Stack enough layers and the vector for a word can carry sentence-level meaning.", { left: 236, top: 552, width: 808, height: 48 }, { fontSize: 26, color: MUTED, alignment: "center" });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "Attention Is Similarity at Scale", "Tokens ask dot-product questions about other tokens.", VIOLET);
  const attWords = ["the", "protein", "folds", "into", "a", "helix"];
  const attScores = [
    [0.86, 0.20, 0.12, 0.10, 0.24, 0.08],
    [0.18, 0.94, 0.62, 0.18, 0.12, 0.74],
    [0.10, 0.66, 0.92, 0.36, 0.14, 0.68],
    [0.08, 0.16, 0.42, 0.88, 0.62, 0.28],
    [0.30, 0.12, 0.12, 0.68, 0.90, 0.34],
    [0.08, 0.72, 0.64, 0.24, 0.36, 0.96],
  ];
  const heatColor = (value) => {
    if (value >= 0.80) {
      return VIOLET;
    }
    if (value >= 0.60) {
      return BLUE;
    }
    if (value >= 0.35) {
      return TEAL;
    }
    if (value >= 0.18) {
      return GOLD;
    }
    return "#F2EEE4";
  };
  addText(s, "att-query", "query token", { left: 266, top: 218, width: 238, height: 28 }, { fontSize: 18, bold: true, color: VIOLET, alignment: "center" });
  addText(s, "att-key", "tokens being compared", { left: 54, top: 396, width: 116, height: 62 }, { fontSize: 18, bold: true, color: MUTED, alignment: "center" });
  const gridLeft = 178;
  const gridTop = 274;
  const cell = 52;
  const gap = 7;
  for (let idx = 0; idx < attWords.length; idx += 1) {
    const x = gridLeft + idx * (cell + gap);
    addText(s, `att-col-${idx}`, attWords[idx], { left: x - 10, top: 236, width: cell + 20, height: 26 }, { fontSize: 15, bold: true, color: INK, alignment: "center" });
    const y = gridTop + idx * (cell + gap);
    addText(s, `att-row-${idx}`, attWords[idx], { left: 84, top: y + 14, width: 78, height: 24 }, { fontSize: 15, bold: true, color: INK, alignment: "right" });
  }
  for (let rowIdx = 0; rowIdx < attScores.length; rowIdx += 1) {
    for (let colIdx = 0; colIdx < attScores[rowIdx].length; colIdx += 1) {
      const value = attScores[rowIdx][colIdx];
      const x = gridLeft + colIdx * (cell + gap);
      const y = gridTop + rowIdx * (cell + gap);
      addRect(s, `att-cell-${rowIdx}-${colIdx}`, { left: x, top: y, width: cell, height: cell }, heatColor(value), "#FFFFFF", 1);
    }
  }
  addRect(s, "att-legend-low", { left: 214, top: 640, width: 46, height: 16 }, "#F2EEE4", RULE, 1);
  addRect(s, "att-legend-mid", { left: 274, top: 640, width: 46, height: 16 }, TEAL, "none");
  addRect(s, "att-legend-high", { left: 334, top: 640, width: 46, height: 16 }, VIOLET, "none");
  addText(s, "att-legend-text", "weak -> strong similarity", { left: 400, top: 636, width: 310, height: 24 }, { fontSize: 17, color: MUTED });
  addRect(s, "att-explain-panel", { left: 780, top: 236, width: 390, height: 352 }, "#FFFFFF", VIOLET, 2);
  addText(s, "att-explain-title", "A layer does this many times", { left: 812, top: 272, width: 326, height: 72 }, { fontSize: 25, bold: true, color: VIOLET, alignment: "center" });
  addText(s, "att-explain-copy", "1. compare token vectors\n2. turn scores into weights\n3. mix information into a new vector", { left: 832, top: 356, width: 300, height: 142 }, { fontSize: 22, color: INK });
  addText(s, "att-explain-foot", "The dot product becomes a whole table of comparisons.", { left: 826, top: 522, width: 302, height: 42 }, { fontSize: 18, bold: true, color: MUTED, alignment: "center" });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "Proteins Are Sequences Too", "Protein language models apply the same representation idea to amino acid strings.", TEAL);
  await addImage(s, "protein-support", PROTEIN_IMAGE, { left: 620, top: 188, width: 660, height: 510 });
  addText(s, "protein-sequence", "MKTAYIAKQRQISFVKSHFSRQ", { left: 70, top: 286, width: 500, height: 48 }, { fontSize: 32, bold: true, color: INK });
  addText(s, "protein-copy", "Amino acids are symbols. A protein language model turns a sequence into vectors for residues, domains, or the whole protein.", { left: 72, top: 390, width: 470, height: 154 }, { fontSize: 27, color: INK });
  addFooter(s, n, "Protein language model examples: ProtTrans and ESM literature");

  n += 1;
  s = slide();
  title(s, n, "Protein Models Learn from Sequence", "Prediction over millions of sequences can reveal functional patterns.", TEAL);
  const p1 = flowBox(s, "seq", "sequence", 94, 302, 170, 84, TEAL);
  const p2 = flowBox(s, "plm", "protein\nlanguage model", 346, 282, 230, 124, TEAL);
  const p3 = flowBox(s, "embed-prot", "embedding", 660, 302, 170, 84, TEAL);
  const p4 = flowBox(s, "tasks", "annotation\nsearch\ndesign", 914, 270, 220, 150, TEAL);
  connect(s, p1, p2, MUTED, 2);
  connect(s, p2, p3, MUTED, 2);
  connect(s, p3, p4, MUTED, 2);
  addText(s, "plm-copy", "The model is not given a hand-written biology dictionary. It learns useful coordinates from patterns in sequences.", { left: 190, top: 520, width: 900, height: 64 }, { fontSize: 26, color: MUTED, alignment: "center" });
  addFooter(s, n, "Sources: ProtTrans arXiv:2007.06225; ESM protein language model literature");

  n += 1;
  s = slide();
  title(s, n, "Live Demo: Sequence to Neighbors, Function, Design", "The demo connects the math to a real protein-model workflow.", TEAL);
  const d1 = flowBox(s, "demo-input", "input\nsequence", 90, 298, 170, 112, TEAL);
  const d2 = flowBox(s, "demo-vector", "embedding\nvector", 340, 298, 180, 112, TEAL);
  const d3 = flowBox(s, "demo-neighbor", "nearest\nproteins", 600, 298, 190, 112, TEAL);
  const d4 = flowBox(s, "demo-output", "labels and\ndesign ideas", 880, 288, 230, 132, TEAL);
  connect(s, d1, d2, MUTED, 2);
  connect(s, d2, d3, MUTED, 2);
  connect(s, d3, d4, MUTED, 2);
  addText(s, "demo-cue", "Live question: which proteins are nearby, what function is likely, and what design idea follows?", { left: 166, top: 514, width: 948, height: 62 }, { fontSize: 24, color: MUTED, alignment: "center" });
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "What Representation Learning Can Do in the Lab", "Vectors become tools for deciding what to test next.", TEAL);
  const labItems = [
    ["annotate", "suggest a function for an unknown sequence", TEAL],
    ["search", "find related proteins across huge databases", BLUE],
    ["triage", "prioritize experiments before spending bench time", GOLD],
    ["design", "propose sequence changes worth testing", ORANGE],
  ];
  for (let idx = 0; idx < labItems.length; idx += 1) {
    const x = idx < 2 ? 120 : 690;
    const y = idx % 2 === 0 ? 260 : 440;
    addRect(s, `lab-${idx}`, { left: x, top: y, width: 450, height: 120 }, "#FFFFFF", labItems[idx][2], 2);
    addText(s, `lab-title-${idx}`, labItems[idx][0], { left: x + 28, top: y + 22, width: 160, height: 34 }, { fontSize: 25, bold: true, color: labItems[idx][2] });
    addText(s, `lab-copy-${idx}`, labItems[idx][1], { left: x + 190, top: y + 20, width: 230, height: 62 }, { fontSize: 19, color: MUTED });
  }
  addFooter(s, n);

  n += 1;
  s = slide();
  title(s, n, "Learn the Coordinates, Then Do the Math", "Representation learning turns messy objects into useful geometry.", ORANGE);
  const endItems = [
    ["compare", BLUE],
    ["predict", ORANGE],
    ["search", VIOLET],
    ["design", TEAL],
  ];
  for (let idx = 0; idx < endItems.length; idx += 1) {
    const x = 156 + idx * 250;
    addCircle(s, `end-${idx}`, { left: x, top: 262, width: 156, height: 156 }, "#FFFFFF", endItems[idx][1], 4);
    addText(s, `end-label-${idx}`, endItems[idx][0], { left: x + 10, top: 318, width: 136, height: 42 }, { fontSize: 25, bold: true, color: endItems[idx][1], alignment: "center" });
  }
  addText(s, "final-line", "Cars, words, images, people, and proteins become points in spaces where math can act.", { left: 150, top: 510, width: 980, height: 70 }, { fontSize: 31, bold: true, color: INK, alignment: "center" });
  addText(s, "questions", "Questions", { left: 454, top: 610, width: 372, height: 48 }, { fontSize: 38, bold: true, color: ORANGE, alignment: "center" });
  addFooter(s, n);

  if (deck.slides.items.length !== 24) {
    throw new Error(`Expected 24 slides, built ${deck.slides.items.length}`);
  }
  addSpeakerNotes(deck);
  return deck;
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });
  const car = JSON.parse(await fs.readFile(CAR_JSON, "utf8"));
  const carSample = JSON.parse(await fs.readFile(CAR_SAMPLE_JSON, "utf8"));
  const bert = JSON.parse(await fs.readFile(BERT_JSON, "utf8"));
  const deck = await buildDeck(car, bert, carSample);

  for (const [index, slide] of deck.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await deck.export({ slide, format: "png", scale: 1.5 });
    await writeBlob(`${PREVIEW_DIR}\\${stem}.png`, png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${LAYOUT_DIR}\\${stem}.layout.json`, await layout.text(), "utf8");
  }

  const montage = await deck.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(`${QA_DIR}\\deck-montage.webp`, montage);
  const inspect = await deck.inspect({ kind: "slide,textbox,shape,image,table,chart,notes,layout", maxChars: 50000 });
  await fs.writeFile(`${QA_DIR}\\inspect.ndjson`, inspect.ndjson, "utf8");
  await fs.writeFile(`${TMP_DIR}\\source-notes-v2.txt`, [
    "Deck source notes",
    "Car dataset: gsv24/car-price on Hugging Face.",
    "Car sample diagnostics: derived from stored prediction samples in the experiment ledger; full-test metrics remain the main reported result.",
    "Language model: answerdotai/ModernBERT-base on Hugging Face.",
    "Protein language model references: ProtTrans arXiv:2007.06225 and ESM protein language model literature.",
    "Generated visual assets are stored in the presentation scratch assets folder.",
  ].join("\n"), "utf8");
  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(FINAL_PPTX);
  console.log(JSON.stringify({ final_pptx: FINAL_PPTX, slides: deck.slides.items.length }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

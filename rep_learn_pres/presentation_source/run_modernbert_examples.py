import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYDEPS = ROOT_DIR / "pydeps"
if PYDEPS.is_dir():
    sys.path.insert(0, str(PYDEPS))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


DEFAULT_OUT_DIR = ROOT_DIR / "models"
DEFAULT_RESULTS_PATH = DEFAULT_OUT_DIR / "modernbert_examples.json"
DEFAULT_PLOTS_DIR = DEFAULT_OUT_DIR / "modernbert_plots"
DEFAULT_MODEL_ID = "answerdotai/ModernBERT-base"
METHODS = ("raw", "unit", "centered_unit")
ANCHOR_LABELS = (
    "king - royalty",
    "queen - royalty",
    "king - man + woman",
    "queen - woman + man",
)
METHOD_LABELS = {
    "raw": "raw",
    "unit": "unit operands",
    "centered_unit": "centered unit",
}
COLORS = {
    "ink": "#17201C",
    "muted": "#5D625F",
    "rule": "#D6D0C4",
    "canvas": "#FAF7F0",
    "panel": "#FFFFFF",
    "violet": "#6E5AEF",
    "blue": "#2F6FDB",
    "teal": "#008C7E",
    "gold": "#DCA62B",
    "orange": "#F05A28",
    "red": "#B83B2E",
}


ANALOGY_BANK = [
    {
        "category": "royalty/personhood",
        "positive": ["king"],
        "negative": ["royalty"],
        "expected": "man",
        "label": "king - royalty",
    },
    {
        "category": "royalty/personhood",
        "positive": ["queen"],
        "negative": ["royalty"],
        "expected": "woman",
        "label": "queen - royalty",
    },
    {
        "category": "royalty/personhood",
        "positive": ["king", "woman"],
        "negative": ["man"],
        "expected": "queen",
        "label": "king - man + woman",
    },
    {
        "category": "royalty/personhood",
        "positive": ["queen", "man"],
        "negative": ["woman"],
        "expected": "king",
        "label": "queen - woman + man",
    },
    {
        "category": "royalty/personhood",
        "positive": ["prince", "woman"],
        "negative": ["man"],
        "expected": "princess",
        "label": "prince - man + woman",
    },
    {
        "category": "royalty/personhood",
        "positive": ["princess", "man"],
        "negative": ["woman"],
        "expected": "prince",
        "label": "princess - woman + man",
    },
    {
        "category": "gender pairs",
        "positive": ["father", "woman"],
        "negative": ["man"],
        "expected": "mother",
        "label": "father - man + woman",
    },
    {
        "category": "gender pairs",
        "positive": ["mother", "man"],
        "negative": ["woman"],
        "expected": "father",
        "label": "mother - woman + man",
    },
    {
        "category": "gender pairs",
        "positive": ["brother", "woman"],
        "negative": ["man"],
        "expected": "sister",
        "label": "brother - man + woman",
    },
    {
        "category": "gender pairs",
        "positive": ["sister", "man"],
        "negative": ["woman"],
        "expected": "brother",
        "label": "sister - woman + man",
    },
    {
        "category": "gender pairs",
        "positive": ["husband", "woman"],
        "negative": ["man"],
        "expected": "wife",
        "label": "husband - man + woman",
    },
    {
        "category": "gender pairs",
        "positive": ["wife", "man"],
        "negative": ["woman"],
        "expected": "husband",
        "label": "wife - woman + man",
    },
    {
        "category": "gender pairs",
        "positive": ["boy", "woman"],
        "negative": ["man"],
        "expected": "girl",
        "label": "boy - man + woman",
    },
    {
        "category": "gender pairs",
        "positive": ["girl", "man"],
        "negative": ["woman"],
        "expected": "boy",
        "label": "girl - woman + man",
    },
    {
        "category": "plurals",
        "positive": ["cats", "dog"],
        "negative": ["cat"],
        "expected": "dogs",
        "label": "cats - cat + dog",
    },
    {
        "category": "plurals",
        "positive": ["dogs", "cat"],
        "negative": ["dog"],
        "expected": "cats",
        "label": "dogs - dog + cat",
    },
    {
        "category": "plurals",
        "positive": ["cars", "truck"],
        "negative": ["car"],
        "expected": "trucks",
        "label": "cars - car + truck",
    },
    {
        "category": "plurals",
        "positive": ["trucks", "car"],
        "negative": ["truck"],
        "expected": "cars",
        "label": "trucks - truck + car",
    },
    {
        "category": "plurals",
        "positive": ["birds", "tree"],
        "negative": ["bird"],
        "expected": "trees",
        "label": "birds - bird + tree",
    },
    {
        "category": "plurals",
        "positive": ["trees", "bird"],
        "negative": ["tree"],
        "expected": "birds",
        "label": "trees - tree + bird",
    },
    {
        "category": "tense",
        "positive": ["walked", "play"],
        "negative": ["walk"],
        "expected": "played",
        "label": "walked - walk + play",
    },
    {
        "category": "tense",
        "positive": ["played", "walk"],
        "negative": ["play"],
        "expected": "walked",
        "label": "played - play + walk",
    },
    {
        "category": "tense",
        "positive": ["running", "swim"],
        "negative": ["run"],
        "expected": "swimming",
        "label": "running - run + swim",
    },
    {
        "category": "tense",
        "positive": ["swimming", "run"],
        "negative": ["swim"],
        "expected": "running",
        "label": "swimming - swim + run",
    },
    {
        "category": "comparatives",
        "positive": ["bigger", "small"],
        "negative": ["big"],
        "expected": "smaller",
        "label": "bigger - big + small",
    },
    {
        "category": "comparatives",
        "positive": ["smaller", "big"],
        "negative": ["small"],
        "expected": "bigger",
        "label": "smaller - small + big",
    },
    {
        "category": "comparatives",
        "positive": ["faster", "slow"],
        "negative": ["fast"],
        "expected": "slower",
        "label": "faster - fast + slow",
    },
    {
        "category": "comparatives",
        "positive": ["slower", "fast"],
        "negative": ["slow"],
        "expected": "faster",
        "label": "slower - slow + fast",
    },
    {
        "category": "comparatives",
        "positive": ["taller", "short"],
        "negative": ["tall"],
        "expected": "shorter",
        "label": "taller - tall + short",
    },
    {
        "category": "capitals/countries",
        "positive": ["paris", "germany"],
        "negative": ["france"],
        "expected": "berlin",
        "label": "paris - france + germany",
    },
    {
        "category": "capitals/countries",
        "positive": ["berlin", "france"],
        "negative": ["germany"],
        "expected": "paris",
        "label": "berlin - germany + france",
    },
    {
        "category": "capitals/countries",
        "positive": ["rome", "france"],
        "negative": ["italy"],
        "expected": "paris",
        "label": "rome - italy + france",
    },
    {
        "category": "capitals/countries",
        "positive": ["madrid", "italy"],
        "negative": ["spain"],
        "expected": "rome",
        "label": "madrid - spain + italy",
    },
    {
        "category": "professions/places",
        "positive": ["teacher", "hospital"],
        "negative": ["school"],
        "expected": "doctor",
        "label": "teacher - school + hospital",
    },
    {
        "category": "professions/places",
        "positive": ["doctor", "school"],
        "negative": ["hospital"],
        "expected": "teacher",
        "label": "doctor - hospital + school",
    },
    {
        "category": "professions/places",
        "positive": ["chef", "hospital"],
        "negative": ["kitchen"],
        "expected": "doctor",
        "label": "chef - kitchen + hospital",
    },
    {
        "category": "professions/places",
        "positive": ["doctor", "kitchen"],
        "negative": ["hospital"],
        "expected": "chef",
        "label": "doctor - hospital + kitchen",
    },
    {
        "category": "semantic attributes",
        "positive": ["puppy", "cat"],
        "negative": ["dog"],
        "expected": "kitten",
        "label": "puppy - dog + cat",
    },
    {
        "category": "semantic attributes",
        "positive": ["kitten", "dog"],
        "negative": ["cat"],
        "expected": "puppy",
        "label": "kitten - cat + dog",
    },
    {
        "category": "semantic attributes",
        "positive": ["bird", "water"],
        "negative": ["air"],
        "expected": "fish",
        "label": "bird - air + water",
    },
    {
        "category": "semantic attributes",
        "positive": ["fish", "air"],
        "negative": ["water"],
        "expected": "bird",
        "label": "fish - water + air",
    },
]


def default_cache_dir():
    if "CODEX_MODEL_CACHE" in os.environ:
        return Path(os.environ["CODEX_MODEL_CACHE"])
    if "TEMP" in os.environ:
        return Path(os.environ["TEMP"]) / "codex_modernbert_cache"
    return DEFAULT_OUT_DIR / "hf_cache"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ModernBERT input-embedding vector arithmetic examples."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-selected", type=int, default=12)
    parser.add_argument("--rank-threshold", type=int, default=25)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def banner(title):
    rule = "=" * len(title)
    print(f"\n{rule}\n{title}\n{rule}")


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    assert np.all(norms > 0)
    return matrix / norms


def cosine_scores(query_vec, matrix):
    query_norm = np.linalg.norm(query_vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    assert query_norm > 0
    assert np.all(matrix_norms > 0)
    return (matrix @ query_vec) / (matrix_norms * query_norm)


def clean_token(token):
    cleaned = token
    prefixes = ("##", "Ġ", "▁", "Ċ", "Â", "Ä ", "â–")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
                changed = True
    cleaned = cleaned.strip().lower()
    if not cleaned.isalpha():
        return ""
    if len(cleaned) < 3:
        return ""
    return cleaned


def token_pieces(tokenizer, word):
    pieces = tokenizer.tokenize(word)
    return [str(piece) for piece in pieces]


def build_vocab(tokenizer, emb):
    token_ids = list(range(emb.shape[0]))
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    special_ids = set(int(value) for value in tokenizer.all_special_ids)
    rows = []
    seen = set()
    skipped = {
        "special": 0,
        "non_word": 0,
        "duplicate": 0,
    }

    for idx, token in tqdm(list(enumerate(tokens)), desc="Filtering vocabulary", unit="token"):
        if idx in special_ids:
            skipped["special"] += 1
            continue
        label = clean_token(token)
        if not label:
            skipped["non_word"] += 1
            continue
        if label in seen:
            skipped["duplicate"] += 1
            continue
        rows.append({"word": label, "token": token, "id": idx})
        seen.add(label)

    words = [row["word"] for row in rows]
    ids = np.array([row["id"] for row in rows], dtype=np.int64)
    word_to_pos = {word: idx for idx, word in enumerate(words)}
    matrix = emb[ids]
    assert len(words) == matrix.shape[0]
    return {
        "rows": rows,
        "words": words,
        "ids": ids,
        "word_to_pos": word_to_pos,
        "matrix": matrix,
        "skipped": skipped,
        "source_token_count": emb.shape[0],
    }


def preflight_analogies(tokenizer, vocab):
    valid = []
    dropped = []
    word_to_pos = vocab["word_to_pos"]

    for item in tqdm(ANALOGY_BANK, desc="Preflighting analogies", unit="example"):
        all_words = item["positive"] + item["negative"] + [item["expected"]]
        missing = []
        tokenization_pieces = []
        for word in all_words:
            if word not in word_to_pos:
                missing.append(word)
            pieces = token_pieces(tokenizer, word)
            if len(pieces) != 1:
                tokenization_pieces.append({"word": word, "pieces": pieces})
        if missing:
            dropped.append(
                {
                    "category": item["category"],
                    "label": item["label"],
                    "expected": item["expected"],
                    "missing": missing,
                    "tokenization_pieces": tokenization_pieces,
                }
            )
            continue
        valid_item = dict(item)
        valid_item["tokenization_pieces"] = tokenization_pieces
        valid.append(valid_item)

    return valid, dropped


def method_embeddings(vocab_matrix, method):
    if method == "raw":
        return vocab_matrix.copy()
    if method == "unit":
        return l2_normalize(vocab_matrix)
    if method == "centered_unit":
        centered = vocab_matrix - vocab_matrix.mean(axis=0, keepdims=True)
        return l2_normalize(centered)
    raise AssertionError(f"Unknown method: {method}")


def query_for(method_matrix, vocab, positive, negative):
    positions_pos = [vocab["word_to_pos"][word] for word in positive]
    positions_neg = [vocab["word_to_pos"][word] for word in negative]
    pos_vec = method_matrix[np.array(positions_pos, dtype=np.int64)].sum(axis=0)
    neg_vec = method_matrix[np.array(positions_neg, dtype=np.int64)].sum(axis=0)
    return pos_vec - neg_vec


def score_one(item, method, method_matrix, vocab, top_k):
    query_vec = query_for(method_matrix, vocab, item["positive"], item["negative"])
    scores = cosine_scores(query_vec, method_matrix)
    excluded_words = set(item["positive"] + item["negative"])
    excluded_positions = [vocab["word_to_pos"][word] for word in excluded_words]
    scores_for_rank = scores.copy()
    scores_for_rank[np.array(excluded_positions, dtype=np.int64)] = -np.inf

    expected_pos = vocab["word_to_pos"][item["expected"]]
    expected_score = float(scores_for_rank[expected_pos])
    better_count = int(np.sum(scores_for_rank > expected_score))
    rank = better_count + 1

    candidate_count = int(np.sum(np.isfinite(scores_for_rank)))
    sorted_positions = np.argsort(-scores_for_rank)
    top_positions = sorted_positions[:top_k]
    top_results = []
    for pos in top_positions:
        if not np.isfinite(scores_for_rank[pos]):
            continue
        top_results.append(
            {
                "word": vocab["words"][int(pos)],
                "score": float(scores_for_rank[int(pos)]),
                "is_expected": vocab["words"][int(pos)] == item["expected"],
            }
        )

    return {
        "method": method,
        "rank": int(rank),
        "candidate_count": candidate_count,
        "expected_score": expected_score,
        "hit_at_1": rank <= 1,
        "hit_at_5": rank <= 5,
        "hit_at_10": rank <= 10,
        "hit_at_25": rank <= 25,
        "top_results": top_results,
        "query_vector": query_vec,
    }


def score_analogies(valid_analogies, vocab, top_k):
    method_matrices = {
        method: method_embeddings(vocab["matrix"], method)
        for method in tqdm(METHODS, desc="Preparing scoring methods", unit="method")
    }
    scored = []
    total = len(valid_analogies) * len(METHODS)
    with tqdm(total=total, desc="Scoring vector arithmetic", unit="score") as bar:
        for item in valid_analogies:
            method_scores = {}
            for method in METHODS:
                method_scores[method] = score_one(
                    item=item,
                    method=method,
                    method_matrix=method_matrices[method],
                    vocab=vocab,
                    top_k=top_k,
                )
                bar.update(1)

            best_method = min(
                METHODS,
                key=lambda method: (
                    method_scores[method]["rank"],
                    -method_scores[method]["expected_score"],
                ),
            )
            best = method_scores[best_method]
            scored.append(
                {
                    "category": item["category"],
                    "label": item["label"],
                    "positive": item["positive"],
                    "negative": item["negative"],
                    "expected": item["expected"],
                    "tokenization_pieces": item["tokenization_pieces"],
                    "best_method": best_method,
                    "best_rank": best["rank"],
                    "best_expected_score": best["expected_score"],
                    "best_top_result": best["top_results"][0]["word"],
                    "methods": {
                        method: {
                            "rank": method_scores[method]["rank"],
                            "candidate_count": method_scores[method]["candidate_count"],
                            "expected_score": method_scores[method]["expected_score"],
                            "hit_at_1": method_scores[method]["hit_at_1"],
                            "hit_at_5": method_scores[method]["hit_at_5"],
                            "hit_at_10": method_scores[method]["hit_at_10"],
                            "hit_at_25": method_scores[method]["hit_at_25"],
                            "top_results": method_scores[method]["top_results"],
                        }
                        for method in METHODS
                    },
                    "query_vectors": {
                        method: method_scores[method]["query_vector"]
                        for method in METHODS
                    },
                }
            )

    return scored, method_matrices


def selected_examples(scored, max_selected, rank_threshold):
    successful = [item for item in scored if item["best_rank"] <= rank_threshold]
    successful.sort(key=lambda item: (item["best_rank"], item["category"], item["label"]))
    fallback = [item for item in scored if item["best_rank"] > rank_threshold]
    fallback.sort(key=lambda item: (item["best_rank"], item["category"], item["label"]))
    selected = (successful + fallback)[:max_selected]
    return selected


def anchor_examples(scored):
    rows = []
    for label in ANCHOR_LABELS:
        matches = [item for item in scored if item["label"] == label]
        if matches:
            rows.append(matches[0])
    return rows


def serialize_scored(scored):
    serializable = []
    for item in scored:
        method = item["best_method"]
        serializable.append(
            {
                "category": item["category"],
                "label": item["label"],
                "expression": expression_text(item),
                "positive": item["positive"],
                "negative": item["negative"],
                "expected": item["expected"],
                "tokenization_pieces": item["tokenization_pieces"],
                "method": method,
                "expected_rank": item["best_rank"],
                "expected_score": item["best_expected_score"],
                "top_result": item["best_top_result"],
                "top_results": item["methods"][method]["top_results"],
            }
        )
    return serializable


def summarize_best_results(scored):
    ranks = np.array([item["best_rank"] for item in scored], dtype=np.float64)
    counts = {}
    for method in METHODS:
        counts[method] = int(sum(1 for item in scored if item["best_method"] == method))
    return {
        "example_count": len(scored),
        "median_expected_rank": float(np.median(ranks)),
        "hit_at_1": float(np.mean(ranks <= 1)),
        "hit_at_5": float(np.mean(ranks <= 5)),
        "hit_at_10": float(np.mean(ranks <= 10)),
        "hit_at_25": float(np.mean(ranks <= 25)),
        "best_method_counts": counts,
        "note": "Each example was scored with raw, unit, and centered-unit arithmetic; only the best-ranking method is exported.",
    }


def expression_text(item):
    positives = " + ".join(item["positive"])
    negatives = " - ".join(item["negative"])
    if negatives:
        return f"{positives} - {negatives}"
    return positives


def legacy_embedding_equations(selected):
    rows = []
    for item in selected[:8]:
        method = item["best_method"]
        rows.append(
            {
                "positive": item["positive"],
                "negative": item["negative"],
                "expected": item["expected"],
                "best_method": method,
                "expected_rank": item["best_rank"],
                "results": item["methods"][method]["top_results"][:8],
            }
        )
    return rows


def curated_directions(selected):
    rows = []
    for item in selected[:8]:
        method = item["best_method"]
        rows.append(
            {
                "positive": item["positive"],
                "negative": item["negative"],
                "expected": item["expected"],
                "method": method,
                "results": item["methods"][method]["top_results"][:8],
            }
        )
    return rows


def fill_mask(tokenizer, model, text, top_k=5):
    encoded = tokenizer(text, return_tensors="pt")
    mask_positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    assert len(mask_positions) == 1
    with torch.no_grad():
        out = model(**encoded)
    logits = out.logits[0, mask_positions[0]]
    values, indices = torch.topk(torch.softmax(logits, dim=-1), k=top_k)
    return {
        "prompt": text,
        "predictions": [
            {"token": tokenizer.decode([int(indices[i])]).strip(), "probability": float(values[i])}
            for i in range(top_k)
        ],
    }


def contextual_bank(tokenizer, model):
    sentences = [
        "The fisherman sat on the river bank.",
        "She opened a savings account at the bank.",
        "The river flooded after the storm.",
        "The teller counted the money.",
    ]
    vectors = []
    for sentence in tqdm(sentences, desc="Encoding contextual examples", unit="sentence"):
        encoded = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True)
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
        bank_positions = [idx for idx, token in enumerate(tokens) if clean_token(token) == "bank"]
        if bank_positions:
            pos = bank_positions[0]
        else:
            pos = 1
        vectors.append(out.hidden_states[-1][0, pos].detach().cpu().numpy())
    sims = l2_normalize(np.stack(vectors)) @ l2_normalize(np.stack(vectors)).T
    return {
        "sentences": sentences,
        "similarities": sims.round(4).tolist(),
        "note": "Rows/columns are the sentence order; values are cosine similarities of final-layer token vectors.",
    }


def plot_topk_successes(selected, path):
    examples = selected[: min(6, len(selected))]
    assert examples
    cols = 2
    rows = math.ceil(len(examples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows), facecolor=COLORS["canvas"])
    axes_array = np.array(axes).reshape(-1)

    for idx, item in enumerate(examples):
        axis = axes_array[idx]
        method = item["best_method"]
        top = item["methods"][method]["top_results"][:8]
        words = [row["word"] for row in top]
        scores = [row["score"] for row in top]
        colors = [COLORS["violet"] if row["word"] == item["expected"] else COLORS["blue"] for row in top]
        axis.set_facecolor(COLORS["panel"])
        axis.barh(np.arange(len(words)), scores, color=colors)
        axis.set_yticks(np.arange(len(words)))
        axis.set_yticklabels(words)
        axis.invert_yaxis()
        axis.set_xlim(min(scores) - 0.03, max(scores) + 0.03)
        axis.set_title(
            f"{expression_text(item)} -> {item['expected']} | rank {item['best_rank']}",
            color=COLORS["ink"],
            weight="bold",
            fontsize=11,
        )
        axis.tick_params(colors=COLORS["muted"])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(COLORS["rule"])
        axis.spines["bottom"].set_color(COLORS["rule"])

    for idx in range(len(examples), len(axes_array)):
        axes_array[idx].axis("off")

    fig.suptitle("Best Top-k Returns from ModernBERT Input Embeddings", color=COLORS["ink"], weight="bold", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_rank_table(selected, path):
    rows = selected[: min(12, len(selected))]
    assert rows
    table_rows = []
    for item in rows:
        table_rows.append(
            [
                expression_text(item),
                item["expected"],
                f"{item['best_rank']}",
                item["best_top_result"],
                METHOD_LABELS[item["best_method"]],
            ]
        )

    fig, axis = plt.subplots(figsize=(15, 0.62 * len(table_rows) + 2.2), facecolor=COLORS["canvas"])
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        colLabels=["expression", "expected", "rank", "top result", "method"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.42, 0.14, 0.08, 0.16, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.45)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["rule"])
        if row_idx == 0:
            cell.set_facecolor(COLORS["violet"])
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(COLORS["panel"] if row_idx % 2 else "#F3EFE6")
            if col_idx == 2:
                rank = int(table_rows[row_idx - 1][2])
                if rank <= 5:
                    cell.set_text_props(color=COLORS["teal"], weight="bold")
                elif rank <= 25:
                    cell.set_text_props(color=COLORS["orange"], weight="bold")

    axis.set_title("Selected ModernBERT Analogy Returns", color=COLORS["ink"], weight="bold", fontsize=16, pad=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pca_examples(selected):
    chosen = []
    seen_categories = set()
    for item in selected:
        if item["category"] in seen_categories:
            continue
        chosen.append(item)
        seen_categories.add(item["category"])
        if len(chosen) == 4:
            return chosen
    for item in selected:
        if item in chosen:
            continue
        chosen.append(item)
        if len(chosen) == 4:
            return chosen
    return chosen


def plot_query_pca(selected, vocab, path):
    examples = pca_examples(selected)
    assert examples
    cols = 2
    rows = math.ceil(len(examples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 5.2 * rows), facecolor=COLORS["canvas"])
    axes_array = np.array(axes).reshape(-1)

    for item_idx, item in enumerate(examples):
        axis = axes_array[item_idx]
        method = item["best_method"]
        method_matrix = method_embeddings(vocab["matrix"], method)
        point_words = item["positive"] + item["negative"] + [item["expected"]]
        point_vectors = []
        point_roles = []
        for word in point_words:
            pos = vocab["word_to_pos"][word]
            point_vectors.append(method_matrix[pos])
            point_roles.append("expected" if word == item["expected"] else "source")
        query_vec = query_for(method_matrix, vocab, item["positive"], item["negative"])

        projection = PCA(n_components=2, random_state=0).fit_transform(np.vstack(point_vectors + [query_vec]))
        point_projection = projection[: len(point_words)]
        query_point = projection[len(point_words)]
        target_point = point_projection[-1]
        x_values = projection[:, 0]
        y_values = projection[:, 1]
        x_span = max(float(x_values.max() - x_values.min()), 0.1)
        y_span = max(float(y_values.max() - y_values.min()), 0.1)
        x_pad = x_span * 0.18
        y_pad = y_span * 0.22

        axis.set_facecolor(COLORS["panel"])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(COLORS["rule"])
        axis.spines["bottom"].set_color(COLORS["rule"])
        axis.tick_params(colors=COLORS["muted"], labelsize=8)
        axis.set_xlim(float(x_values.min()) - x_pad, float(x_values.max()) + x_pad)
        axis.set_ylim(float(y_values.min()) - y_pad, float(y_values.max()) + y_pad)
        axis.set_title(
            f"{expression_text(item)} -> {item['expected']} | rank {item['best_rank']}",
            color=COLORS["ink"],
            weight="bold",
            fontsize=11,
        )

        for idx, word in enumerate(point_words):
            role = point_roles[idx]
            color = COLORS["violet"] if role == "expected" else COLORS["muted"]
            axis.scatter(point_projection[idx, 0], point_projection[idx, 1], s=72, color=color, zorder=3)
            offset_x = x_span * (0.025 if point_projection[idx, 0] <= query_point[0] else -0.055)
            offset_y = y_span * (0.04 if idx % 2 == 0 else -0.055)
            axis.text(
                point_projection[idx, 0] + offset_x,
                point_projection[idx, 1] + offset_y,
                word,
                color=COLORS["ink"],
                fontsize=9,
                weight="bold" if role == "expected" else "normal",
                ha="left" if offset_x > 0 else "right",
            )

        axis.scatter(query_point[0], query_point[1], s=100, marker="x", color=COLORS["orange"], zorder=4)
        axis.text(
            query_point[0] + x_span * 0.025,
            query_point[1] + y_span * 0.04,
            "query",
            color=COLORS["orange"],
            fontsize=9,
            weight="bold",
        )
        axis.annotate(
            "",
            xy=(target_point[0], target_point[1]),
            xytext=(query_point[0], query_point[1]),
            arrowprops={"arrowstyle": "->", "color": COLORS["orange"], "lw": 2.2},
        )
        axis.set_xlabel("local PCA 1", color=COLORS["muted"], fontsize=9)
        axis.set_ylabel("local PCA 2", color=COLORS["muted"], fontsize=9)

    for idx in range(len(examples), len(axes_array)):
        axes_array[idx].axis("off")

    fig.suptitle("Local PCA Views of Best Arithmetic Queries", color=COLORS["ink"], weight="bold", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_plots(selected, vocab, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("modernbert_topk_successes.png", lambda path: plot_topk_successes(selected, path)),
        ("modernbert_rank_table.png", lambda path: plot_rank_table(selected, path)),
        ("modernbert_query_pca.png", lambda path: plot_query_pca(selected, vocab, path)),
    ]
    paths = []
    for filename, writer in tqdm(plot_specs, desc="Generating plots", unit="plot"):
        path = plots_dir / filename
        writer(path)
        paths.append(str(path))
    return paths


def print_summary(scored, selected, anchors, dropped, vocab, args):
    banner("ModernBERT Vector Arithmetic Summary")
    print(f"Model: {args.model_id}")
    print(f"Cache: {args.cache_dir}")
    print(f"Output: {args.out_dir / DEFAULT_RESULTS_PATH.name}")
    print(f"Clean vocab words: {len(vocab['words']):,}")
    print(f"Skipped vocab tokens: {vocab['skipped']}")
    print(f"Valid analogies: {len(scored):,}")
    print(f"Dropped analogies: {len(dropped):,}")
    best_summary = summarize_best_results(scored)
    print(
        "Best-method hits: "
        f"hit@5={best_summary['hit_at_5'] * 100:.1f}% "
        f"hit@10={best_summary['hit_at_10'] * 100:.1f}% "
        f"hit@25={best_summary['hit_at_25'] * 100:.1f}% "
        f"median rank={best_summary['median_expected_rank']:.0f}"
    )

    print("\nSelected examples")
    for item in selected:
        method = item["best_method"]
        top_words = ", ".join(row["word"] for row in item["methods"][method]["top_results"][:6])
        found = "yes" if item["best_rank"] <= args.top_k else "no"
        print(
            f"- {expression_text(item)} -> {item['expected']} "
            f"| best={METHOD_LABELS[method]} rank={item['best_rank']} "
            f"| in top{args.top_k}: {found} | top: {top_words}"
        )

    if anchors:
        print("\nClassic royalty probes")
        for item in anchors:
            method = item["best_method"]
            top_words = ", ".join(row["word"] for row in item["methods"][method]["top_results"][:6])
            found = "yes" if item["best_rank"] <= args.top_k else "no"
            print(
                f"- {expression_text(item)} -> {item['expected']} "
                f"| best={METHOD_LABELS[method]} rank={item['best_rank']} "
                f"| in top{args.top_k}: {found} | top: {top_words}"
            )


def build_payload(args, tokenizer, model, vocab, scored, selected, anchors, dropped, plot_paths):
    masks = [
        fill_mask(tokenizer, model, f"The {tokenizer.mask_token} wore a crown."),
        fill_mask(tokenizer, model, f"The scientist mixed chemicals in the {tokenizer.mask_token}."),
        fill_mask(tokenizer, model, f"The protein sequence folds into a {tokenizer.mask_token}."),
    ]
    return {
        "model_id": args.model_id,
        "embedding_equations": legacy_embedding_equations(selected),
        "curated_directions": curated_directions(selected),
        "analogy_experiment": {
            "description": "ModernBERT input embedding vector arithmetic over cleaned token embeddings. Raw, unit, and centered-unit arithmetic are scored internally; exported results show the best method per example.",
            "top_k": args.top_k,
            "rank_threshold": args.rank_threshold,
            "max_selected": args.max_selected,
            "vocab": {
                "clean_word_count": len(vocab["words"]),
                "source_token_count": int(vocab["source_token_count"]),
                "skipped": vocab["skipped"],
            },
            "summary": summarize_best_results(scored),
            "selected_examples": serialize_scored(selected),
            "anchor_examples": serialize_scored(anchors),
            "all_examples": serialize_scored(scored),
            "dropped_examples": dropped,
            "plot_paths": plot_paths,
        },
        "fill_mask_examples": masks,
        "contextual_similarity": contextual_bank(tokenizer, model),
        "caveat": "These examples use ModernBERT input embeddings. They are useful for a representation-learning demonstration, but ModernBERT is a contextual masked-language model, not a classic static word2vec analogy model.",
    }


def main():
    args = parse_args()
    assert args.top_k > 0
    assert args.max_selected > 0
    assert args.rank_threshold > 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        args.plots_dir.mkdir(parents=True, exist_ok=True)

    banner("ModernBERT Input Embedding Arithmetic")
    print(f"Model id: {args.model_id}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Output dir: {args.out_dir}")
    if args.no_plots:
        print("Plot generation: disabled")
    else:
        print(f"Plots dir: {args.plots_dir}")

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=str(args.cache_dir))
    model = AutoModelForMaskedLM.from_pretrained(args.model_id, cache_dir=str(args.cache_dir))
    model.eval()

    emb = model.get_input_embeddings().weight.detach().cpu().numpy()
    print(f"Embedding table: {emb.shape[0]:,} tokens x {emb.shape[1]:,} dimensions")

    vocab = build_vocab(tokenizer, emb)
    valid_analogies, dropped = preflight_analogies(tokenizer, vocab)
    assert valid_analogies

    scored, _method_matrices = score_analogies(valid_analogies, vocab, args.top_k)
    assert scored
    selected = selected_examples(scored, args.max_selected, args.rank_threshold)
    assert selected
    anchors = anchor_examples(scored)

    plot_paths = []
    if not args.no_plots:
        plot_paths = make_plots(selected, vocab, args.plots_dir)

    print_summary(scored, selected, anchors, dropped, vocab, args)

    payload = build_payload(args, tokenizer, model, vocab, scored, selected, anchors, dropped, plot_paths)
    results_path = args.out_dir / DEFAULT_RESULTS_PATH.name
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    banner("Wrote ModernBERT Artifacts")
    print(json.dumps({"results_path": str(results_path), "plot_paths": plot_paths}, indent=2))


if __name__ == "__main__":
    main()

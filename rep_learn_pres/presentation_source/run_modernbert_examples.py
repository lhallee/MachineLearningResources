import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYDEPS = ROOT_DIR / "pydeps"
if PYDEPS.is_dir():
    sys.path.insert(0, str(PYDEPS))

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


OUT_DIR = ROOT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
if "CODEX_MODEL_CACHE" in os.environ:
    CACHE_DIR = Path(os.environ["CODEX_MODEL_CACHE"])
elif "TEMP" in os.environ:
    CACHE_DIR = Path(os.environ["TEMP"]) / "codex_modernbert_cache"
else:
    CACHE_DIR = ROOT_DIR / "models" / "hf_cache"
RESULTS_PATH = OUT_DIR / "modernbert_examples.json"
MODEL_ID = "answerdotai/ModernBERT-base"


def cosine_matrix(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def single_token_id(tokenizer, word):
    pieces = tokenizer.tokenize(word)
    if len(pieces) != 1:
        return None
    return tokenizer.convert_tokens_to_ids(pieces[0])


def clean_token(token):
    cleaned = token.replace("##", "").replace("Ġ", "").replace("▁", "")
    if not cleaned.isalpha():
        return ""
    if len(cleaned) < 3:
        return ""
    return cleaned.lower()


def nearest_words(tokenizer, emb, query_vec, exclude_words, top_k=8):
    tokens = tokenizer.convert_ids_to_tokens(list(range(emb.shape[0])))
    labels = []
    indices = []
    seen = set()
    for idx, token in enumerate(tokens):
        label = clean_token(token)
        if not label:
            continue
        if label in seen:
            continue
        if label in exclude_words:
            continue
        labels.append(label)
        indices.append(idx)
        seen.add(label)
    matrix = emb[np.array(indices)]
    sims = cosine_matrix(query_vec.reshape(1, -1), matrix)[0]
    order = np.argsort(-sims)[:top_k]
    return [{"word": labels[i], "score": float(sims[i])} for i in order]


def analogy(tokenizer, emb, positive, negative):
    ids_pos = [single_token_id(tokenizer, word) for word in positive]
    ids_neg = [single_token_id(tokenizer, word) for word in negative]
    if any(value is None for value in ids_pos + ids_neg):
        return {"positive": positive, "negative": negative, "results": [], "note": "One or more words were not single tokens."}
    vec = np.sum(emb[np.array(ids_pos)], axis=0) - np.sum(emb[np.array(ids_neg)], axis=0)
    exclude = set(positive + negative)
    return {"positive": positive, "negative": negative, "results": nearest_words(tokenizer, emb, vec, exclude)}


def candidate_direction(tokenizer, emb, positive, negative, candidates):
    ids_pos = [single_token_id(tokenizer, word) for word in positive]
    ids_neg = [single_token_id(tokenizer, word) for word in negative]
    candidate_ids = [single_token_id(tokenizer, word) for word in candidates]
    if any(value is None for value in ids_pos + ids_neg + candidate_ids):
        return {"positive": positive, "negative": negative, "candidates": candidates, "results": [], "note": "One or more words were not single tokens."}
    vec = np.sum(emb[np.array(ids_pos)], axis=0) - np.sum(emb[np.array(ids_neg)], axis=0)
    matrix = emb[np.array(candidate_ids)]
    sims = cosine_matrix(vec.reshape(1, -1), matrix)[0]
    order = np.argsort(-sims)
    return {
        "positive": positive,
        "negative": negative,
        "results": [{"word": candidates[index], "score": float(sims[index])} for index in order],
    }


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
    for sentence in sentences:
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
    sims = cosine_matrix(np.stack(vectors), np.stack(vectors))
    return {
        "sentences": sentences,
        "similarities": sims.round(4).tolist(),
        "note": "Rows/columns are the sentence order; values are cosine similarities of final-layer token vectors.",
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, cache_dir=str(CACHE_DIR))
    model.eval()

    emb = model.get_input_embeddings().weight.detach().cpu().numpy()
    equations = [
        analogy(tokenizer, emb, ["king", "woman"], ["man"]),
        analogy(tokenizer, emb, ["paris", "germany"], ["france"]),
        analogy(tokenizer, emb, ["doctor", "school"], ["hospital"]),
    ]
    curated_directions = [
        candidate_direction(tokenizer, emb, ["dogs", "cat"], ["dog"], ["cats", "kitten", "pet", "car", "table"]),
        candidate_direction(tokenizer, emb, ["cars", "truck"], ["car"], ["trucks", "bus", "road", "banana", "teacher"]),
        candidate_direction(tokenizer, emb, ["played", "walk"], ["play"], ["walked", "walking", "walks", "run", "table"]),
    ]
    masks = [
        fill_mask(tokenizer, model, f"The {tokenizer.mask_token} wore a crown."),
        fill_mask(tokenizer, model, f"The scientist mixed chemicals in the {tokenizer.mask_token}."),
        fill_mask(tokenizer, model, f"The protein sequence folds into a {tokenizer.mask_token}."),
    ]
    payload = {
        "model_id": MODEL_ID,
        "embedding_equations": equations,
        "curated_directions": curated_directions,
        "fill_mask_examples": masks,
        "contextual_similarity": contextual_bank(tokenizer, model),
        "caveat": "These examples use ModernBERT only. Its input embeddings can illustrate vector arithmetic, but its strongest behavior is contextual representation and masked-token prediction, not classic static word2vec-style analogies.",
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"results_path": str(RESULTS_PATH), "model_id": MODEL_ID}, indent=2))


if __name__ == "__main__":
    main()

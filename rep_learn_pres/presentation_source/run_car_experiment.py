import json
import math
import os
import random
import sys
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYDEPS = ROOT_DIR / "pydeps"
if PYDEPS.is_dir():
    sys.path.insert(0, str(PYDEPS))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


OUT_DIR = ROOT_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "hf_cache"
METRICS_PATH = OUT_DIR / "car_experiment_metrics.json"
MAX_ROWS = 25000
SEEDS = [7, 17, 29, 43, 71]
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
DATASETS = [
    {
        "id": "gsv24/car-price",
        "url": "https://huggingface.co/datasets/gsv24/car-price/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "local": OUT_DIR / "gsv24_car_price.parquet",
    },
    {
        "id": "VarunKumarGupta2003/Car-Price-Dataset",
        "url": "https://huggingface.co/datasets/VarunKumarGupta2003/Car-Price-Dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "local": OUT_DIR / "varun_car_price.parquet",
    },
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def download_parquet(item):
    local = item["local"]
    if local.is_file() and local.stat().st_size > 0:
        return local
    tmp = local.with_suffix(".parquet.incomplete")
    urllib.request.urlretrieve(item["url"], tmp)
    assert tmp.is_file()
    assert tmp.stat().st_size > 0
    tmp.replace(local)
    return local


def load_frame():
    errors = []
    for item in DATASETS:
        try:
            path = download_parquet(item)
            df = pd.read_parquet(path)
            source = item["id"]
            break
        except Exception as error:
            errors.append(f"{item['id']}: {error}")
    else:
        raise RuntimeError("\n".join(errors))
    if errors:
        (OUT_DIR / "primary_dataset_fallback_reason.txt").write_text("\n".join(errors), encoding="utf-8")
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    if "price" not in df.columns:
        raise AssertionError("Expected a price target column.")
    return df, source


def clean_frame(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 1000]
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=20260624)

    categorical = [
        col
        for col in [
            "make",
            "model",
            "transmission",
            "fuel_type",
            "drivetrain",
            "body_type",
            "exterior_color",
            "interior_color",
            "accident_history",
            "seller_type",
            "condition",
            "trim",
        ]
        if col in df.columns
    ]
    if not categorical:
        categorical = [col for col in df.columns if col != "price" and df[col].dtype == object]

    numeric = [col for col in df.columns if col != "price" and col not in categorical and pd.api.types.is_numeric_dtype(df[col])]
    if not numeric:
        raise AssertionError("Expected at least one numeric feature.")
    if not categorical:
        raise AssertionError("Expected at least one categorical feature.")

    for col in categorical:
        df[col] = df[col].astype("string").fillna("__missing__").str.strip().str.lower()
        df[col] = df[col].replace("", "__missing__")
    for col in numeric:
        median = df[col].median()
        if not np.isfinite(median):
            median = 0.0
        df[col] = df[col].fillna(float(median)).astype(float)

    df = df.reset_index(drop=True)
    return df, numeric, categorical


def split_frame(df):
    y_log = np.log1p(df["price"].to_numpy(dtype=np.float64))
    bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")
    if pd.Series(bins).nunique() < 2:
        bins = None
    train_idx, hold_idx = train_test_split(
        np.arange(len(df)),
        train_size=0.70,
        random_state=20260624,
        stratify=bins,
    )
    hold_bins = None if bins is None else np.asarray(bins)[hold_idx]
    val_idx, test_idx = train_test_split(
        hold_idx,
        train_size=0.50,
        random_state=20260625,
        stratify=hold_bins,
    )
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(val_idx).isdisjoint(set(test_idx))
    return train_idx, val_idx, test_idx


def fit_preprocessors(df, numeric, categorical, train_idx):
    scaler = StandardScaler()
    x_num_train = df.loc[train_idx, numeric].to_numpy(dtype=np.float32)
    scaler.fit(x_num_train)

    cat_maps = {}
    vocab_sizes = []
    for col in categorical:
        counts = df.loc[train_idx, col].value_counts()
        keep = counts[counts >= 10].index.tolist()
        mapping = {"__unknown__": 0}
        for value in keep:
            mapping[str(value)] = len(mapping)
        cat_maps[col] = mapping
        vocab_sizes.append(len(mapping))

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_cats = np.column_stack([encode_column(df.loc[train_idx, col], cat_maps[col]) for col in categorical])
    encoder.fit(train_cats.astype(str))
    return scaler, cat_maps, vocab_sizes, encoder


def encode_column(values, mapping):
    result = np.zeros(len(values), dtype=np.int64)
    for idx, value in enumerate(values.astype("string")):
        key = str(value)
        if key in mapping:
            result[idx] = mapping[key]
    return result


def transform_split(df, numeric, categorical, idx, scaler, cat_maps, encoder):
    x_num = scaler.transform(df.loc[idx, numeric].to_numpy(dtype=np.float32)).astype(np.float32)
    x_cat = np.column_stack([encode_column(df.loc[idx, col], cat_maps[col]) for col in categorical]).astype(np.int64)
    cat_scale = np.array([max(len(cat_maps[col]) - 1, 1) for col in categorical], dtype=np.float32)
    x_int = np.concatenate([x_num, x_cat.astype(np.float32) / cat_scale], axis=1).astype(np.float32)
    x_hot = np.concatenate([x_num, encoder.transform(x_cat.astype(str)).astype(np.float32)], axis=1).astype(np.float32)
    y = np.log1p(df.loc[idx, "price"].to_numpy(dtype=np.float32)).reshape(-1, 1)
    y_price = df.loc[idx, "price"].to_numpy(dtype=np.float32)
    return x_num, x_cat, x_int, x_hot, y, y_price


class DenseModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class EmbeddingModel(nn.Module):
    def __init__(self, num_numeric, vocab_sizes, emb_dim=16):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, emb_dim) for size in vocab_sizes])
        self.net = nn.Sequential(
            nn.Linear(num_numeric + emb_dim * len(vocab_sizes), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_num, x_cat):
        embedded = [table(x_cat[:, index]) for index, table in enumerate(self.embeddings)]
        return self.net(torch.cat([x_num, *embedded], dim=1))


def count_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train_dense(name, seed, train_x, val_x, test_x, train_y, val_y, test_y, test_price):
    set_seed(seed)
    model = DenseModel(train_x.shape[1])
    return train_loop(name, model, seed, (train_x,), (val_x,), (test_x,), train_y, val_y, test_y, test_price)


def train_embedding(seed, train_num, train_cat, val_num, val_cat, test_num, test_cat, train_y, val_y, test_y, test_price, vocab_sizes):
    set_seed(seed)
    model = EmbeddingModel(train_num.shape[1], vocab_sizes)
    return train_loop(
        "learned embeddings",
        model,
        seed,
        (train_num, train_cat),
        (val_num, val_cat),
        (test_num, test_cat),
        train_y,
        val_y,
        test_y,
        test_price,
    )


def make_loader(inputs, y, shuffle):
    tensors = [torch.tensor(arr) for arr in inputs]
    tensors.append(torch.tensor(y, dtype=torch.float32))
    return DataLoader(TensorDataset(*tensors), batch_size=1024, shuffle=shuffle)


def train_loop(name, model, seed, train_inputs, val_inputs, test_inputs, train_y, val_y, test_y, test_price):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = make_loader(train_inputs, train_y, True)
    val_loader = make_loader(val_inputs, val_y, False)
    best_state = None
    best_val = math.inf
    wait = 0
    curves = []
    for epoch in range(1, 17):
        model.train()
        train_losses = []
        for batch in train_loader:
            inputs = batch[:-1]
            target = batch[-1]
            optimizer.zero_grad(set_to_none=True)
            pred = model(*inputs) if len(inputs) > 1 else model(inputs[0])
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = evaluate_loss(model, val_loader, loss_fn)
        train_loss = float(np.mean(train_losses))
        curves.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 4:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    pred_log = predict(model, test_inputs)
    pred_price = np.maximum(np.expm1(pred_log), 0)
    true_price = test_price.astype(np.float64)
    rmse = float(mean_squared_error(true_price, pred_price) ** 0.5)
    mae = float(mean_absolute_error(true_price, pred_price))
    r2 = float(r2_score(true_price, pred_price))
    return {
        "encoding": name,
        "seed": seed,
        "best_val_loss": best_val,
        "epochs": len(curves),
        "params": count_params(model),
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
        "curves": curves,
        "predicted_sample": pred_price[:250].tolist(),
        "actual_sample": true_price[:250].tolist(),
        "model": model,
    }


def evaluate_loss(model, loader, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch[:-1]
            target = batch[-1]
            pred = model(*inputs) if len(inputs) > 1 else model(inputs[0])
            losses.append(float(loss_fn(pred, target).detach().cpu()))
    return float(np.mean(losses))


def predict(model, inputs):
    model.eval()
    loader = make_loader(inputs, np.zeros((len(inputs[0]), 1), dtype=np.float32), False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            features = batch[:-1]
            pred = model(*features) if len(features) > 1 else model(features[0])
            preds.append(pred.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds)


def summarize(results):
    summaries = []
    for name in ["integer IDs", "one-hot", "learned embeddings"]:
        rows = [row for row in results if row["encoding"] == name]
        assert rows
        summaries.append({
            "encoding": name,
            "test_rmse_mean": float(np.mean([row["test_rmse"] for row in rows])),
            "test_rmse_std": float(np.std([row["test_rmse"] for row in rows], ddof=1)),
            "test_mae_mean": float(np.mean([row["test_mae"] for row in rows])),
            "test_mae_std": float(np.std([row["test_mae"] for row in rows], ddof=1)),
            "test_r2_mean": float(np.mean([row["test_r2"] for row in rows])),
            "test_r2_std": float(np.std([row["test_r2"] for row in rows], ddof=1)),
            "params_mean": float(np.mean([row["params"] for row in rows])),
        })
    return summaries


def embedding_projection(best_result, cat_maps, categorical):
    model = best_result["model"]
    assert isinstance(model, EmbeddingModel)
    preferred = "make" if "make" in categorical else categorical[0]
    index = categorical.index(preferred)
    weights = model.embeddings[index].weight.detach().cpu().numpy()
    inv = {idx: value for value, idx in cat_maps[preferred].items()}
    labels = [inv[idx] for idx in range(len(inv)) if idx != 0]
    if len(labels) < 3:
        return {"feature": preferred, "points": []}
    usable = weights[1 : len(labels) + 1]
    coords = PCA(n_components=2, random_state=20260624).fit_transform(usable)
    return {
        "feature": preferred,
        "points": [
            {"label": labels[idx], "x": float(coords[idx, 0]), "y": float(coords[idx, 1])}
            for idx in range(len(labels))
        ],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, source = load_frame()
    df, numeric, categorical = clean_frame(df)
    train_idx, val_idx, test_idx = split_frame(df)
    scaler, cat_maps, vocab_sizes, encoder = fit_preprocessors(df, numeric, categorical, train_idx)

    train_num, train_cat, train_int, train_hot, train_y, train_price = transform_split(df, numeric, categorical, train_idx, scaler, cat_maps, encoder)
    val_num, val_cat, val_int, val_hot, val_y, val_price = transform_split(df, numeric, categorical, val_idx, scaler, cat_maps, encoder)
    test_num, test_cat, test_int, test_hot, test_y, test_price = transform_split(df, numeric, categorical, test_idx, scaler, cat_maps, encoder)

    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)
    assert train_int.shape[0] == train_y.shape[0]
    assert test_hot.shape[0] == test_y.shape[0]

    results = []
    for seed in SEEDS:
        results.append(train_dense("integer IDs", seed, train_int, val_int, test_int, train_y, val_y, test_y, test_price))
        results.append(train_dense("one-hot", seed, train_hot, val_hot, test_hot, train_y, val_y, test_y, test_price))
        results.append(train_embedding(seed, train_num, train_cat, val_num, val_cat, test_num, test_cat, train_y, val_y, test_y, test_price, vocab_sizes))

    summary = summarize(results)
    best_embedding = min([row for row in results if row["encoding"] == "learned embeddings"], key=lambda row: row["best_val_loss"])
    projection = embedding_projection(best_embedding, cat_maps, categorical)

    serializable_results = []
    for row in results:
        next_row = {}
        for key, value in row.items():
            if key != "model":
                next_row[key] = value
        serializable_results.append(next_row)

    payload = {
        "source_dataset": source,
        "rows_used": int(len(df)),
        "max_rows_cap": MAX_ROWS,
        "split_counts": {"train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))},
        "numeric_features": numeric,
        "categorical_features": categorical,
        "vocab_sizes": {col: int(len(cat_maps[col])) for col in categorical},
        "summary": summary,
        "runs": serializable_results,
        "embedding_projection": projection,
        "notes": [
            "Training target was log1p(price); reported metrics are on the original price scale.",
            "Preprocessors and category vocabularies were fit on the train split only.",
            "The row cap is a deterministic laptop-friendly sample used to keep training light.",
        ],
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"metrics_path": str(METRICS_PATH), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

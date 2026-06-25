import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import pandas as pd
import torch
from sklearn.decomposition import PCA

from run_categorical_benchmark import (
    DATASET_REGISTRY,
    ROOT_DIR,
    build_model,
    clean_frame,
    color_for_label,
    configure_torch,
    dataset_config_payload,
    fit_preprocessors,
    inputs_for_encoding,
    load_dataset_frame,
    output_dim_for_task,
    parse_int_list,
    plot_color_embedding_projection,
    plot_embedding_projection,
    prepare_target,
    resolve_device,
    resolve_preprocess_backend,
    serializable_result,
    set_seed,
    split_indices,
    train_loop,
    transform_split,
)


DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "car_color_pca"


def parse_args():
    parser = argparse.ArgumentParser(description="Train one car-price embedding model and write PCA plots for color embeddings.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--full-dataset", dest="full_dataset", action="store_true", default=True)
    parser.add_argument("--no-full-dataset", dest="full_dataset", action="store_false")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--validation-rows", type=int, default=100_000)
    parser.add_argument("--test-rows", type=int, default=100_000)
    parser.add_argument("--eval-every-examples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-sizes", type=parse_int_list, default=[1024, 8192, 128])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--category-min-count", type=int, default=10)
    parser.add_argument("--sample-predictions", type=int, default=1000)
    parser.add_argument("--max-total-vocab", type=int, default=50_000)
    parser.add_argument("--max-vocab-per-column", type=int, default=10_000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--require-cuda", action="store_true", default=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--feature-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--preprocess-backend", choices=["auto", "pandas", "cudf"], default="cudf")
    return parser.parse_args()


def validate_args(args):
    assert args.rows > 0
    assert args.validation_rows > 0
    assert args.test_rows > 0
    assert args.eval_every_examples > 0
    assert args.batch_size > 0
    assert args.learning_rate > 0
    assert args.weight_decay >= 0
    assert args.embedding_size > 0
    assert 0 <= args.dropout < 1
    assert len(args.hidden_sizes) == 3
    assert all(hidden_size > 0 for hidden_size in args.hidden_sizes)
    assert args.epochs > 0
    assert args.patience > 0
    assert args.min_delta >= 0
    assert args.category_min_count > 0
    assert args.sample_predictions > 0
    assert args.max_total_vocab >= 2
    assert args.max_vocab_per_column >= 2
    assert args.threads > 0
    assert args.interop_threads > 0
    assert args.num_workers >= 0
    assert args.feature_workers >= 0
    assert args.prefetch_factor > 0


def color_pca_rows(result, inv_maps, categorical):
    rows = []
    color_features = [feature for feature in ("exterior_color", "interior_color") if feature in categorical]
    weights_list = result["embedding_weights"]
    for feature in color_features:
        feature_index = categorical.index(feature)
        weights = weights_list[feature_index]
        inverse = inv_maps[feature]
        labels = []
        for idx in range(1, min(weights.shape[0], 41)):
            labels.append(inverse[idx])
        if len(labels) < 3:
            continue
        coords = PCA(n_components=2, random_state=20260624).fit_transform(weights[1 : len(labels) + 1])
        for idx, label in enumerate(labels):
            rows.append(
                {
                    "feature": feature,
                    "category_id": int(idx + 1),
                    "label": str(label),
                    "pc1": float(coords[idx, 0]),
                    "pc2": float(coords[idx, 1]),
                    "plot_color": color_for_label(label),
                }
            )
    return rows


def train_embedding_for_color_pca(args):
    start = time.perf_counter()
    spec = DATASET_REGISTRY["car-price"]
    cache_dir = args.output_dir / "_cache" / spec.slug
    plot_dir = args.output_dir / spec.slug / "plots"
    result_dir = args.output_dir / spec.slug
    plot_dir.mkdir(parents=True, exist_ok=True)

    configure_torch(args)
    device = resolve_device(args)
    print(f"Running fast car color PCA on {device}.")
    print(f"Architecture: input -> {' -> '.join(str(value) for value in args.hidden_sizes)} -> 1")
    print(f"Embedding size: {args.embedding_size}; epochs={args.epochs}; patience={args.patience}")

    frame = load_dataset_frame(spec, args, cache_dir)
    frame, numeric, categorical = clean_frame(frame, spec)
    preprocess_backend, cudf_module = resolve_preprocess_backend(args)
    print(f"Preprocessing backend: {preprocess_backend}")
    model_target, original_target, class_names = prepare_target(frame, spec)
    train_idx, val_idx, test_idx = split_indices(spec, model_target, original_target, args)
    scaler, cat_maps, inv_maps, vocab_sizes = fit_preprocessors(
        frame, numeric, categorical, train_idx, args, preprocess_backend, cudf_module
    )
    train_num, train_cat, train_int, train_y, train_eval = transform_split(
        frame, numeric, categorical, train_idx, scaler, cat_maps, model_target, original_target, args, preprocess_backend, cudf_module, "train"
    )
    val_num, val_cat, val_int, val_y, val_eval = transform_split(
        frame, numeric, categorical, val_idx, scaler, cat_maps, model_target, original_target, args, preprocess_backend, cudf_module, "validation"
    )
    test_num, test_cat, test_int, test_y, test_eval = transform_split(
        frame, numeric, categorical, test_idx, scaler, cat_maps, model_target, original_target, args, preprocess_backend, cudf_module, "test"
    )

    split_counts = {"train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))}
    output_dim = output_dim_for_task(spec.task, class_names)
    run_args = Namespace(**vars(args))
    run_args.seeds = [int(args.seed)]
    set_seed(args.seed)
    model = build_model("learned embeddings", train_num.shape[1], vocab_sizes, run_args, output_dim).to(device)
    train_inputs, val_inputs, test_inputs = inputs_for_encoding(
        "learned embeddings",
        train_num,
        train_cat,
        train_int,
        val_num,
        val_cat,
        val_int,
        test_num,
        test_cat,
        test_int,
    )
    result = train_loop(
        spec.slug,
        "learned embeddings",
        model,
        args.seed,
        train_inputs,
        val_inputs,
        test_inputs,
        train_y,
        val_y,
        test_eval,
        spec,
        class_names,
        run_args,
        device,
    )
    result["embedding_weights"] = [table.weight.detach().cpu().numpy() for table in model.embeddings]
    torch.cuda.empty_cache()

    color_plot = plot_dir / "color_embedding_projection.png"
    embedding_plot = plot_dir / "embedding_projection.png"
    points_csv = result_dir / "color_embedding_projection_points.csv"
    plot_color_embedding_projection(result, inv_maps, categorical, color_plot)
    plot_embedding_projection(result, inv_maps, categorical, embedding_plot)
    pd.DataFrame(color_pca_rows(result, inv_maps, categorical)).to_csv(points_csv, index=False)

    config_payload = dataset_config_payload(spec, run_args, device, numeric, categorical, vocab_sizes, split_counts)
    payload = config_payload
    payload["seconds"] = round(float(time.perf_counter() - start), 3)
    payload["run"] = serializable_result(result)
    payload["outputs"] = {
        "color_embedding_projection": str(color_plot),
        "embedding_projection": str(embedding_plot),
        "color_embedding_projection_points": str(points_csv),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "color_pca_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([serializable_result(result)]).to_csv(result_dir / "color_pca_metrics.csv", index=False)
    print(f"Wrote color PCA: {color_plot}")
    print(f"Wrote embedding PCA: {embedding_plot}")
    print(f"Wrote metrics: {result_dir / 'color_pca_metrics.json'}")


def main():
    args = parse_args()
    validate_args(args)
    train_embedding_for_color_pca(args)


if __name__ == "__main__":
    main()

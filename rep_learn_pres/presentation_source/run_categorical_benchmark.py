import argparse
import json
import math
import os
import random
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYDEPS = ROOT_DIR / "pydeps"
PYDEPS_PYTHON_VERSION = (3, 12)


def add_compatible_pydeps(pydeps):
    if not pydeps.is_dir():
        return "not found"

    current_version = sys.version_info[:2]
    if current_version != PYDEPS_PYTHON_VERSION:
        expected = ".".join(str(part) for part in PYDEPS_PYTHON_VERSION)
        current = ".".join(str(part) for part in current_version)
        print(f"Skipping local pydeps because it targets Python {expected}, but this run is Python {current}.")
        return "skipped"

    sys.path.insert(0, str(pydeps))
    return "added"


PYDEPS_STATUS = add_compatible_pydeps(PYDEPS)

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from scipy.special import expit, softmax
    from scipy.stats import pearsonr, spearmanr, t as student_t, ttest_rel, wilcoxon
    from sklearn.decomposition import PCA
    from sklearn.datasets import fetch_openml
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm.auto import tqdm
except ImportError as error:
    raise ImportError(
        f"{error}\n\n"
        f"Dependency import failed for Python {sys.version.split()[0]} at {sys.executable}. "
        f"Local pydeps status: {PYDEPS_STATUS}. "
        "Run with the Python version that matches pydeps, for example "
        "`py -3.12 -m run_categorical_benchmark`, or install `requirements.txt` into this Python."
    ) from error


OUT_DIR = ROOT_DIR / "data" / "categorical_benchmarks"
HF_DATASET_SERVER = "https://datasets-server.huggingface.co"
DEFAULT_ROWS = 1_000_000
DEFAULT_VALIDATION_ROWS = 100_000
DEFAULT_TEST_ROWS = 100_000
DEFAULT_EVAL_EVERY_EXAMPLES = 100_000
DEFAULT_BATCH_SIZE = 4096
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EMBEDDING_SIZE = 16
DEFAULT_DROPOUT = 0.2
DEFAULT_HIDDEN_SIZES = [256, 8192, 128]
DEFAULT_SEEDS = [7, 17, 29]
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 3
DEFAULT_SAMPLE_PREDICTIONS = 1000
DEFAULT_CATEGORY_MIN_COUNT = 10
DEFAULT_MAX_TOTAL_VOCAB = 50_000
DEFAULT_MAX_VOCAB_PER_COLUMN = 10_000
DEFAULT_NUM_WORKERS = 16
DEFAULT_FEATURE_WORKERS = 16
DEFAULT_PREFETCH_FACTOR = 4
ENCODINGS = ["integer IDs", "learned embeddings"]
SKIP_BY_DEFAULT = {"bank-marketing"}
PLOT_COLORS = {
    "integer IDs": "#B83B2E",
    "learned embeddings": "#2F6FDB",
}
METHOD_PAIRS = [
    ("learned embeddings", "integer IDs", "Embedding vs integer"),
]


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    name: str
    task: str
    source_kind: str
    target: str
    citation_url: str
    categorical: tuple
    numeric: tuple
    drop: tuple = ()
    repo_id: str = ""
    hf_config: str = "default"
    hf_split: str = "train"
    openml_id: int = 0
    urls: tuple = ()
    local_name: str = ""
    target_transform: str = "none"
    min_target: float = math.nan


def names(prefix, start, end):
    return tuple(f"{prefix}{idx}" for idx in range(start, end + 1))


DATASET_REGISTRY = {
    "cat-in-the-dat": DatasetSpec(
        slug="cat-in-the-dat",
        name="Cat in the Dat",
        task="binary",
        source_kind="csv",
        target="target",
        categorical=(
            "bin_0",
            "bin_1",
            "bin_2",
            "bin_3",
            "bin_4",
            "nom_0",
            "nom_1",
            "nom_2",
            "nom_3",
            "nom_4",
            "nom_5",
            "nom_6",
            "nom_7",
            "nom_8",
            "nom_9",
            "ord_0",
            "ord_1",
            "ord_2",
            "ord_3",
            "ord_4",
            "ord_5",
            "day",
            "month",
        ),
        numeric=(),
        drop=("id",),
        urls=("https://huggingface.co/datasets/michaelmallari/cat-in-the-dat/resolve/main/train.csv",),
        local_name="cat-in-the-dat-train.csv",
        citation_url="https://hf.co/datasets/michaelmallari/cat-in-the-dat",
    ),
    "criteo-x1": DatasetSpec(
        slug="criteo-x1",
        name="Criteo x1",
        task="binary",
        source_kind="hf",
        repo_id="reczoo/Criteo_x1",
        target="label",
        categorical=names("C", 1, 26),
        numeric=names("I", 1, 13),
        citation_url="https://hf.co/datasets/reczoo/Criteo_x1",
    ),
    "avazu-x1": DatasetSpec(
        slug="avazu-x1",
        name="Avazu x1",
        task="binary",
        source_kind="hf",
        repo_id="reczoo/Avazu_x1",
        target="label",
        categorical=names("feat_", 1, 22),
        numeric=(),
        citation_url="https://hf.co/datasets/reczoo/Avazu_x1",
    ),
    "movielens-1m": DatasetSpec(
        slug="movielens-1m",
        name="MovieLens 1M",
        task="regression",
        source_kind="movielens",
        target="rating",
        categorical=("user_id", "movie_id", "gender", "age", "occupation", "zip", "genres"),
        numeric=("timestamp",),
        urls=("https://files.grouplens.org/datasets/movielens/ml-1m.zip",),
        local_name="ml-1m.zip",
        citation_url="https://grouplens.org/datasets/movielens/",
    ),
    "adult": DatasetSpec(
        slug="adult",
        name="Adult",
        task="binary",
        source_kind="openml",
        openml_id=1590,
        target="class",
        categorical=(
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ),
        numeric=("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"),
        citation_url="https://archive.ics.uci.edu/dataset/2/adult",
    ),
    "bank-marketing": DatasetSpec(
        slug="bank-marketing",
        name="Bank Marketing",
        task="binary",
        source_kind="openml",
        openml_id=1461,
        target="Class",
        categorical=("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"),
        numeric=("age", "balance", "day", "duration", "campaign", "pdays", "previous"),
        citation_url="https://archive.ics.uci.edu/dataset/222/bank+marketing",
    ),
    "mushroom": DatasetSpec(
        slug="mushroom",
        name="Mushroom",
        task="binary",
        source_kind="openml",
        openml_id=24,
        target="class",
        categorical=(),
        numeric=(),
        citation_url="https://archive.ics.uci.edu/dataset/73/mushroom",
    ),
    "nursery": DatasetSpec(
        slug="nursery",
        name="Nursery",
        task="multiclass",
        source_kind="openml",
        openml_id=26,
        target="class",
        categorical=(),
        numeric=(),
        citation_url="https://archive.ics.uci.edu/dataset/76/nursery",
    ),
    "car-price": DatasetSpec(
        slug="car-price",
        name="Car Price",
        task="regression",
        source_kind="car",
        target="price",
        categorical=(
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
        ),
        numeric=("year", "mileage", "engine_hp", "owner_count", "vehicle_age", "mileage_per_year", "brand_popularity"),
        urls=(
            "https://huggingface.co/datasets/gsv24/car-price/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
            "https://huggingface.co/datasets/VarunKumarGupta2003/Car-Price-Dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        ),
        local_name="car_price.parquet",
        target_transform="log1p",
        min_target=1000.0,
        citation_url="https://hf.co/datasets/gsv24/car-price",
    ),
}


def parse_int_list(value):
    values = []
    for part in value.split(","):
        stripped = part.strip()
        if stripped:
            values.append(int(stripped))
    assert values
    return values


def parse_str_list(value):
    values = []
    for part in value.split(","):
        stripped = part.strip()
        if stripped:
            values.append(stripped)
    assert values
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Run categorical representation benchmarks.")
    parser.add_argument("--datasets", type=parse_str_list, default=None)
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--full-dataset", dest="full_dataset", action="store_true", default=True)
    parser.add_argument("--no-full-dataset", dest="full_dataset", action="store_false")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--validation-rows", type=int, default=DEFAULT_VALIDATION_ROWS)
    parser.add_argument("--test-rows", type=int, default=DEFAULT_TEST_ROWS)
    parser.add_argument("--eval-every-examples", type=int, default=DEFAULT_EVAL_EVERY_EXAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--embedding-size", type=int, default=DEFAULT_EMBEDDING_SIZE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--hidden-sizes", type=parse_int_list, default=list(DEFAULT_HIDDEN_SIZES))
    parser.add_argument("--seeds", type=parse_int_list, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--category-min-count", type=int, default=DEFAULT_CATEGORY_MIN_COUNT)
    parser.add_argument("--sample-predictions", type=int, default=DEFAULT_SAMPLE_PREDICTIONS)
    parser.add_argument("--max-total-vocab", type=int, default=DEFAULT_MAX_TOTAL_VOCAB)
    parser.add_argument("--max-vocab-per-column", type=int, default=DEFAULT_MAX_VOCAB_PER_COLUMN)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--feature-workers", type=int, default=DEFAULT_FEATURE_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--preprocess-backend", choices=["auto", "pandas", "cudf"], default="auto")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--loader-only", action="store_true")
    args = parser.parse_args()
    row_cap_requested = False
    full_dataset_requested = False
    full_dataset_disabled = False
    for arg in sys.argv[1:]:
        if arg == "--rows" or arg.startswith("--rows="):
            row_cap_requested = True
        if arg == "--full-dataset":
            full_dataset_requested = True
        if arg == "--no-full-dataset":
            full_dataset_disabled = True
    if row_cap_requested and not full_dataset_requested and not full_dataset_disabled:
        args.full_dataset = False
    return args


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
    assert len(args.seeds) >= 2
    selected = selected_dataset_slugs(args)
    for slug in selected:
        assert slug in DATASET_REGISTRY, f"Unknown dataset slug: {slug}"


def selected_dataset_slugs(args):
    if args.all_datasets or args.datasets is None:
        return [slug for slug in DATASET_REGISTRY.keys() if slug not in SKIP_BY_DEFAULT]
    return args.datasets


def configure_torch(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_feature_workers(args, categorical_count):
    if categorical_count <= 1:
        return 1
    if args.feature_workers > 0:
        return min(args.feature_workers, categorical_count)
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 1
    return min(max(1, cpu_count // 2), categorical_count)


def import_cudf_module():
    try:
        import cudf
    except ImportError:
        return None
    return cudf


def resolve_preprocess_backend(args):
    if args.preprocess_backend == "pandas":
        return "pandas", None
    cudf_module = import_cudf_module()
    if args.preprocess_backend == "cudf":
        assert cudf_module is not None, "RAPIDS cuDF was requested with --preprocess-backend cudf, but cudf is not importable."
        return "cudf", cudf_module
    if cudf_module is None:
        return "pandas", None
    return "cudf", cudf_module


def resolve_device(args):
    if args.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        assert not args.require_cuda, "CUDA is required but torch.cuda.is_available() is false."
        return torch.device("cpu")
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA was requested, but torch.cuda.is_available() is false."
        return torch.device("cuda")
    assert args.device == "cpu"
    assert not args.require_cuda, "CUDA is required but --device cpu was provided."
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def download_file(url, local):
    if local.is_file() and local.stat().st_size > 0:
        print(f"Using cached file: {local}")
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".incomplete")
    print(f"Downloading {url}")
    with tqdm(desc=f"download {local.name}", unit="B", unit_scale=True, leave=False) as progress:

        def reporthook(block_count, block_size, total_size):
            if total_size > 0:
                progress.total = total_size
            downloaded = block_count * block_size
            delta = downloaded - progress.n
            if delta > 0:
                progress.update(delta)

        urllib.request.urlretrieve(url, tmp, reporthook=reporthook)
    assert tmp.is_file()
    assert tmp.stat().st_size > 0
    tmp.replace(local)
    return local


def row_limit_for_args(args):
    if args.full_dataset:
        return 0
    return args.rows


def cache_label(args):
    if args.full_dataset:
        return "full"
    return f"rows_{args.rows}"


def load_csv_frame(spec, args, cache_dir):
    local = download_file(spec.urls[0], cache_dir / "downloads" / spec.local_name)
    print(f"Loading CSV dataset {spec.slug} from {local}.")
    frame = pd.read_csv(local)
    row_limit = row_limit_for_args(args)
    if row_limit and len(frame) > row_limit:
        frame = frame.sample(n=row_limit, random_state=20260624)
    return frame


def load_hf_frame(spec, args, cache_dir):
    from datasets import load_dataset

    row_limit = row_limit_for_args(args)
    split = spec.hf_split
    if row_limit:
        split = f"{split}[:{row_limit}]"
    print(f"Loading Hugging Face dataset {spec.repo_id} split {split}.")
    dataset = load_dataset(spec.repo_id, spec.hf_config, split=split, cache_dir=str(cache_dir / "hf"))
    frame = dataset.to_pandas()
    return frame


def load_openml_frame(spec, args, cache_dir):
    print(f"Loading OpenML dataset {spec.openml_id} ({spec.slug}).")
    fetched = fetch_openml(data_id=spec.openml_id, as_frame=True, parser="auto", data_home=str(cache_dir / "openml"))
    frame = fetched.data.copy()
    frame[spec.target] = fetched.target
    row_limit = row_limit_for_args(args)
    if row_limit and len(frame) > row_limit:
        frame = frame.sample(n=row_limit, random_state=20260624)
    return frame


def load_movielens_frame(spec, args, cache_dir):
    local = download_file(spec.urls[0], cache_dir / "downloads" / spec.local_name)
    extract_dir = cache_dir / "movielens-1m"
    ratings_path = extract_dir / "ml-1m" / "ratings.dat"
    users_path = extract_dir / "ml-1m" / "users.dat"
    movies_path = extract_dir / "ml-1m" / "movies.dat"
    if not ratings_path.is_file():
        with zipfile.ZipFile(local) as archive:
            archive.extractall(extract_dir)
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    frame = ratings.merge(users, on="user_id", how="left").merge(movies[["movie_id", "genres"]], on="movie_id", how="left")
    row_limit = row_limit_for_args(args)
    if row_limit and len(frame) > row_limit:
        frame = frame.sample(n=row_limit, random_state=20260624)
    return frame


def load_car_frame(spec, args, cache_dir):
    errors = []
    for index, url in enumerate(spec.urls):
        local_name = f"car_price_{index}.parquet"
        local = cache_dir / "downloads" / local_name
        try:
            path = download_file(url, local)
            frame = pd.read_parquet(path)
            break
        except Exception as error:
            errors.append(f"{url}: {error}")
    else:
        raise RuntimeError("\n".join(errors))
    frame.columns = [str(col).strip().lower().replace(" ", "_") for col in frame.columns]
    row_limit = row_limit_for_args(args)
    if row_limit and len(frame) > row_limit:
        frame = frame.sample(n=row_limit, random_state=20260624)
    return frame


def load_dataset_frame(spec, args, cache_dir):
    frame_cache = cache_dir / "frames" / f"{spec.slug}_{cache_label(args)}.parquet"
    if frame_cache.is_file():
        print(f"Loading cached frame: {frame_cache}")
        return pd.read_parquet(frame_cache)
    if spec.source_kind == "csv":
        frame = load_csv_frame(spec, args, cache_dir)
    elif spec.source_kind == "hf":
        frame = load_hf_frame(spec, args, cache_dir)
    elif spec.source_kind == "openml":
        frame = load_openml_frame(spec, args, cache_dir)
    elif spec.source_kind == "movielens":
        frame = load_movielens_frame(spec, args, cache_dir)
    elif spec.source_kind == "car":
        frame = load_car_frame(spec, args, cache_dir)
    else:
        raise AssertionError(f"Unsupported source kind: {spec.source_kind}")
    frame_cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(frame_cache, index=False)
    print(f"Wrote cached frame: {frame_cache}")
    return frame


def clean_frame(frame, spec):
    frame = frame.copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    assert spec.target in frame.columns, f"Missing target column {spec.target} for {spec.slug}."
    frame = frame.dropna(subset=[spec.target])
    if spec.task == "regression" and np.isfinite(spec.min_target):
        frame = frame[frame[spec.target].astype(float) > spec.min_target]
    for col in spec.drop:
        if col in frame.columns:
            frame = frame.drop(columns=[col])

    if spec.categorical:
        categorical = [col for col in spec.categorical if col in frame.columns and col != spec.target]
    else:
        categorical = [
            col
            for col in frame.columns
            if col != spec.target and (pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_categorical_dtype(frame[col]))
        ]

    if spec.numeric:
        numeric = [col for col in spec.numeric if col in frame.columns and col != spec.target and col not in categorical]
    else:
        numeric = [
            col
            for col in frame.columns
            if col != spec.target and col not in categorical and pd.api.types.is_numeric_dtype(frame[col])
        ]

    assert categorical, f"{spec.slug} has no categorical feature columns after cleaning."
    for col in tqdm(categorical, desc=f"{spec.slug} categorical clean", unit="feature", leave=False):
        if pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].fillna(-1)
        else:
            frame[col] = frame[col].astype("string").fillna("__missing__")
            frame[col] = frame[col].replace("", "__missing__")
    for col in tqdm(numeric, desc=f"{spec.slug} numeric clean", unit="feature", leave=False):
        median = frame[col].median()
        if not np.isfinite(median):
            median = 0.0
        frame[col] = frame[col].fillna(float(median)).astype(float)

    frame = frame.reset_index(drop=True)
    return frame, numeric, categorical


def prepare_target(frame, spec):
    if spec.task == "regression":
        original = frame[spec.target].to_numpy(dtype=np.float64)
        if spec.target_transform == "log1p":
            model_target = np.log1p(original).astype(np.float32).reshape(-1, 1)
        else:
            model_target = original.astype(np.float32).reshape(-1, 1)
        return model_target, original, []

    labels = frame[spec.target].astype("string").fillna("__missing__")
    classes = sorted(labels.unique().tolist())
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    encoded = labels.map(class_to_index).to_numpy(dtype=np.int64)
    if spec.task == "binary":
        assert len(classes) == 2, f"{spec.slug} expected binary target, found {classes}."
        model_target = encoded.astype(np.float32).reshape(-1, 1)
    else:
        assert len(classes) >= 3, f"{spec.slug} expected multiclass target, found {classes}."
        model_target = encoded.astype(np.int64)
    return model_target, encoded, classes


def stratify_labels(spec, model_target, original_target):
    if spec.task == "regression":
        values = np.asarray(original_target, dtype=np.float64)
        bins = pd.qcut(values, q=10, labels=False, duplicates="drop")
        if pd.Series(bins).nunique() < 2:
            return None
        return np.asarray(bins)
    labels = np.asarray(original_target)
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        return None
    if int(counts.min()) < 2:
        return None
    return labels


def split_indices(spec, model_target, original_target, args):
    row_count = len(model_target)
    assert row_count >= 10, f"{spec.slug} has too few rows after cleaning."
    if row_count > args.validation_rows + args.test_rows + 10:
        validation_rows = args.validation_rows
        test_rows = args.test_rows
    else:
        validation_rows = max(1, int(round(row_count * 0.15)))
        test_rows = max(1, int(round(row_count * 0.15)))
        if validation_rows + test_rows >= row_count:
            validation_rows = max(1, row_count // 5)
            test_rows = max(1, row_count // 5)
    train_rows = row_count - validation_rows - test_rows
    assert train_rows > 0

    stratify = stratify_labels(spec, model_target, original_target)
    train_idx, hold_idx = train_test_split(
        np.arange(row_count),
        train_size=train_rows,
        random_state=20260624,
        stratify=stratify,
    )
    hold_stratify = None if stratify is None else np.asarray(stratify)[hold_idx]
    if hold_stratify is not None:
        counts = pd.Series(hold_stratify).value_counts()
        if int(counts.min()) < 2:
            hold_stratify = None
    val_idx, test_idx = train_test_split(
        hold_idx,
        train_size=validation_rows,
        test_size=test_rows,
        random_state=20260625,
        stratify=hold_stratify,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def effective_vocab_cap(args, categorical_count):
    fair_cap = max(2, args.max_total_vocab // max(1, categorical_count))
    return min(args.max_vocab_per_column, fair_cap)


def encode_column_pandas(values, categories):
    codes = pd.Categorical(values, categories=categories).codes.astype(np.int64) + 1
    codes[codes < 0] = 0
    return codes


def encode_column_cudf(values, categories, cudf_module):
    mapping = {category: idx for idx, category in enumerate(categories, start=1)}
    gpu_values = cudf_module.Series(values.reset_index(drop=True))
    codes = gpu_values.map(mapping).fillna(0).astype("int64")
    return codes.to_pandas().to_numpy(dtype=np.int64)


def encode_column(values, categories, backend, cudf_module):
    if backend == "cudf":
        return encode_column_cudf(values, categories, cudf_module)
    return encode_column_pandas(values, categories)


def value_counts_pandas(frame, col, train_idx):
    return frame[col].iloc[train_idx].value_counts(sort=True, dropna=False)


def value_counts_cudf(frame, col, train_idx, cudf_module):
    gpu_values = cudf_module.Series(frame[col].iloc[train_idx].reset_index(drop=True))
    counts = gpu_values.value_counts(dropna=False)
    counts = counts.sort_values(ascending=False)
    return counts.to_pandas()


def fit_vocab_column(frame, col, train_idx, min_count, per_column_cap, backend, cudf_module):
    if backend == "cudf":
        counts = value_counts_cudf(frame, col, train_idx, cudf_module)
    else:
        counts = value_counts_pandas(frame, col, train_idx)
    keep = counts[counts >= min_count].head(per_column_cap - 1).index.tolist()
    categories = tuple(keep)
    inverse = {0: "__unknown__"}
    for idx, value in enumerate(categories, start=1):
        inverse[idx] = str(value)
    return col, categories, inverse, len(categories) + 1


def fit_preprocessors(frame, numeric, categorical, train_idx, args, backend, cudf_module):
    scaler = StandardScaler()
    if numeric:
        scaler.fit(frame.loc[train_idx, numeric].to_numpy(dtype=np.float32))
    cat_maps = {}
    inv_maps = {}
    vocab_sizes = []
    per_column_cap = effective_vocab_cap(args, len(categorical))
    feature_workers = 1 if backend == "cudf" else resolve_feature_workers(args, len(categorical))
    print(f"Fitting categorical vocabularies with {backend} using {feature_workers} feature workers.")
    if feature_workers == 1:
        for col in tqdm(categorical, desc="vocabulary", unit="feature", leave=False):
            name, categories, inverse, vocab_size = fit_vocab_column(
                frame, col, train_idx, args.category_min_count, per_column_cap, backend, cudf_module
            )
            cat_maps[name] = categories
            inv_maps[name] = inverse
    else:
        futures = []
        with ThreadPoolExecutor(max_workers=feature_workers) as executor:
            for col in categorical:
                futures.append(executor.submit(fit_vocab_column, frame, col, train_idx, args.category_min_count, per_column_cap, backend, cudf_module))
            for future in tqdm(as_completed(futures), total=len(futures), desc="vocabulary", unit="feature", leave=False):
                name, categories, inverse, vocab_size = future.result()
                cat_maps[name] = categories
                inv_maps[name] = inverse
    for col in categorical:
        vocab_sizes.append(len(cat_maps[col]) + 1)
    return scaler, cat_maps, inv_maps, vocab_sizes


def encode_categorical_columns(frame, categorical, idx, cat_maps, args, backend, cudf_module, desc):
    x_cat = np.empty((len(idx), len(categorical)), dtype=np.int64)
    feature_workers = 1 if backend == "cudf" else resolve_feature_workers(args, len(categorical))

    def encode_position(position):
        col = categorical[position]
        encoded = encode_column(frame[col].iloc[idx], cat_maps[col], backend, cudf_module)
        return position, encoded

    if feature_workers == 1:
        for position in tqdm(range(len(categorical)), desc=desc, unit="feature", leave=False):
            column_position, encoded = encode_position(position)
            x_cat[:, column_position] = encoded
    else:
        futures = []
        with ThreadPoolExecutor(max_workers=feature_workers) as executor:
            for position in range(len(categorical)):
                futures.append(executor.submit(encode_position, position))
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="feature", leave=False):
                column_position, encoded = future.result()
                x_cat[:, column_position] = encoded
    return x_cat


def transform_split(frame, numeric, categorical, idx, scaler, cat_maps, model_target, original_target, args, backend, cudf_module, split_name):
    if numeric:
        x_num = scaler.transform(frame.loc[idx, numeric].to_numpy(dtype=np.float32)).astype(np.float32)
    else:
        x_num = np.zeros((len(idx), 0), dtype=np.float32)
    x_cat = encode_categorical_columns(frame, categorical, idx, cat_maps, args, backend, cudf_module, desc=f"{split_name} categorical encode")
    cat_scale = np.array([max(len(cat_maps[col]), 1) for col in categorical], dtype=np.float32)
    x_int = np.concatenate([x_num, x_cat.astype(np.float32) / cat_scale], axis=1).astype(np.float32)
    y = model_target[idx]
    y_eval = original_target[idx]
    return x_num, x_cat, x_int, y, y_eval


def make_mlp_tail(hidden_sizes, dropout, output_dim):
    layers = []
    current_dim = hidden_sizes[0]
    for hidden_size in hidden_sizes[1:]:
        layers.append(nn.Linear(current_dim, hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        current_dim = hidden_size
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class DenseModel(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout, output_dim):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        layers.extend(list(make_mlp_tail(hidden_sizes, dropout, output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EmbeddingModel(nn.Module):
    def __init__(self, num_numeric, vocab_sizes, emb_dims, hidden_sizes, dropout, output_dim):
        super().__init__()
        assert len(vocab_sizes) == len(emb_dims)
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_sizes[idx], emb_dims[idx]) for idx in range(len(vocab_sizes))])
        input_dim = num_numeric + sum(emb_dims)
        self.net = DenseModel(input_dim, hidden_sizes, dropout, output_dim)

    def forward(self, x_num, x_cat):
        embedded = [table(x_cat[:, index]) for index, table in enumerate(self.embeddings)]
        return self.net(torch.cat([x_num, *embedded], dim=1))


def count_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def make_loader(inputs, y, batch_size, shuffle, num_workers, pin_memory, prefetch_factor, task):
    tensors = [torch.tensor(arr) for arr in inputs]
    if task == "multiclass":
        tensors.append(torch.tensor(y, dtype=torch.long))
    else:
        tensors.append(torch.tensor(y, dtype=torch.float32))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(TensorDataset(*tensors), **kwargs)


def move_inputs(inputs, device):
    return [item.to(device, non_blocking=True) for item in inputs]


def forward_model(model, inputs):
    if len(inputs) == 1:
        return model(inputs[0])
    return model(inputs[0], inputs[1])


def output_dim_for_task(task, class_names):
    if task == "multiclass":
        return len(class_names)
    return 1


def loss_for_task(task):
    if task == "multiclass":
        return nn.CrossEntropyLoss()
    if task == "binary":
        return nn.BCEWithLogitsLoss()
    return nn.MSELoss()


def inverse_regression_target(values, spec):
    if spec.target_transform == "log1p":
        return np.maximum(np.expm1(values), 0)
    return values


def train_loop(dataset_slug, encoding, model, seed, train_inputs, val_inputs, test_inputs, train_y, val_y, test_y_eval, spec, class_names, args, device):
    run_start = time.perf_counter()
    loss_fn = loss_for_task(spec.task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_inputs, train_y, args.batch_size, True, args.num_workers, pin_memory, args.prefetch_factor, spec.task)
    val_loader = make_loader(val_inputs, val_y, args.batch_size, False, args.num_workers, pin_memory, args.prefetch_factor, spec.task)
    model_params = count_params(model)
    best_state = None
    best_val = math.inf
    best_epoch = 0
    best_train_examples_seen = 0
    best_optimizer_steps = 0
    wait = 0
    curves = []
    train_examples_seen = 0
    optimizer_steps = 0
    validation_checks = 0
    next_eval_examples = args.eval_every_examples
    interval_losses = []
    stop_training = False
    last_epoch = 0

    print(f"Training {dataset_slug} {encoding} seed {seed} with {model_params:,} parameters.")
    epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"{dataset_slug} {encoding} s{seed}", unit="epoch", leave=False)
    for epoch in epoch_bar:
        last_epoch = int(epoch)
        model.train()
        for batch in tqdm(train_loader, desc=f"train {encoding} e{epoch}", unit="batch", leave=False):
            inputs = move_inputs(batch[:-1], device)
            target = batch[-1].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_model(model, inputs)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            interval_losses.append(loss_value)
            batch_size = int(target.shape[0])
            train_examples_seen += batch_size
            optimizer_steps += 1

            if train_examples_seen >= next_eval_examples:
                validation_checks += 1
                val_loss = evaluate_loss(model, val_loader, loss_fn, device, f"validate {encoding} check {validation_checks}")
                train_loss = float(np.mean(interval_losses))
                interval_losses = []
                curves.append(
                    {
                        "dataset": dataset_slug,
                        "encoding": encoding,
                        "seed": int(seed),
                        "evaluation": int(validation_checks),
                        "epoch": int(epoch),
                        "optimizer_steps": int(optimizer_steps),
                        "train_examples_seen": int(train_examples_seen),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                )
                epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", best=f"{best_val:.4f}")
                tqdm.write(
                    f"{dataset_slug} {encoding} seed {seed} check {validation_checks}: "
                    f"examples={train_examples_seen:,}, val_loss={val_loss:.5f}"
                )
                if val_loss < best_val - args.min_delta:
                    best_val = val_loss
                    best_epoch = int(epoch)
                    best_train_examples_seen = int(train_examples_seen)
                    best_optimizer_steps = int(optimizer_steps)
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                next_eval_examples += args.eval_every_examples
                if wait >= args.patience:
                    tqdm.write(f"Early stopping {dataset_slug} {encoding} seed {seed} after {validation_checks} validation checks.")
                    stop_training = True
                    break
        if stop_training:
            break

    if best_state is None:
        assert interval_losses
        validation_checks += 1
        val_loss = evaluate_loss(model, val_loader, loss_fn, device, f"validate {encoding} final")
        train_loss = float(np.mean(interval_losses))
        curves.append(
            {
                "dataset": dataset_slug,
                "encoding": encoding,
                "seed": int(seed),
                "evaluation": int(validation_checks),
                "epoch": int(last_epoch),
                "optimizer_steps": int(optimizer_steps),
                "train_examples_seen": int(train_examples_seen),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        best_val = val_loss
        best_epoch = int(last_epoch)
        best_train_examples_seen = int(train_examples_seen)
        best_optimizer_steps = int(optimizer_steps)
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    logits = predict(model, test_inputs, args, device, f"predict {encoding} seed {seed}", spec.task)
    metrics, prediction_payload = compute_test_metrics(logits, test_y_eval, spec, class_names)
    run_seconds = time.perf_counter() - run_start
    throughput = float(train_examples_seen / max(run_seconds, 1e-9))
    result = {
        "dataset": dataset_slug,
        "encoding": encoding,
        "seed": int(seed),
        "task": spec.task,
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
        "best_train_examples_seen": int(best_train_examples_seen),
        "best_optimizer_steps": int(best_optimizer_steps),
        "validation_checks": int(validation_checks),
        "train_examples_seen": int(train_examples_seen),
        "epochs": int(last_epoch),
        "params": int(model_params),
        "run_seconds": float(run_seconds),
        "train_examples_per_second": throughput,
        "curves": curves,
        "prediction_payload": prediction_payload,
    }
    for key, value in metrics.items():
        result[key] = value
    print(f"Finished {dataset_slug} {encoding} seed {seed}: primary={primary_metric_value(result, spec):.5f}, seconds={run_seconds:.1f}")
    return result


def evaluate_loss(model, loader, loss_fn, device, desc):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
            inputs = move_inputs(batch[:-1], device)
            target = batch[-1].to(device, non_blocking=True)
            logits = forward_model(model, inputs)
            losses.append(float(loss_fn(logits, target).detach().cpu()))
    assert losses
    return float(np.mean(losses))


def predict(model, inputs, args, device, desc, task):
    zeros = np.zeros((len(inputs[0]),), dtype=np.int64) if task == "multiclass" else np.zeros((len(inputs[0]), 1), dtype=np.float32)
    pin_memory = device.type == "cuda"
    loader = make_loader(inputs, zeros, args.batch_size, False, args.num_workers, pin_memory, args.prefetch_factor, task)
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
            features = move_inputs(batch[:-1], device)
            logits = forward_model(model, features)
            chunks.append(logits.detach().cpu().numpy())
    assert chunks
    return np.concatenate(chunks, axis=0)


def compute_test_metrics(logits, y_eval, spec, class_names):
    if spec.task == "regression":
        pred_model = logits.reshape(-1).astype(np.float64)
        pred = inverse_regression_target(pred_model, spec)
        true = np.asarray(y_eval, dtype=np.float64)
        rmse = float(mean_squared_error(true, pred) ** 0.5)
        mae = float(mean_absolute_error(true, pred))
        r2 = float(r2_score(true, pred))
        pearson = float(pearsonr(true, pred)[0])
        spearman = float(spearmanr(true, pred)[0])
        metrics = {
            "test_rmse": rmse,
            "test_mae": mae,
            "test_r2": r2,
            "test_pearson": pearson,
            "test_spearman": spearman,
            "test_primary_metric": -rmse,
        }
        payload = {"actual": true, "predicted": pred}
        return metrics, payload

    if spec.task == "binary":
        true = np.asarray(y_eval, dtype=np.int64)
        prob = expit(logits.reshape(-1)).astype(np.float64)
        pred = (prob >= 0.5).astype(np.int64)
        metrics = {
            "test_auc": float(roc_auc_score(true, prob)),
            "test_average_precision": float(average_precision_score(true, prob)),
            "test_accuracy": float(accuracy_score(true, pred)),
            "test_f1": float(f1_score(true, pred)),
            "test_log_loss": float(log_loss(true, prob, labels=[0, 1])),
            "test_primary_metric": float(roc_auc_score(true, prob)),
        }
        payload = {"actual": true, "predicted": pred, "probability": prob}
        return metrics, payload

    true = np.asarray(y_eval, dtype=np.int64)
    probs = softmax(logits, axis=1)
    pred = np.argmax(probs, axis=1)
    labels = list(range(len(class_names)))
    metrics = {
        "test_accuracy": float(accuracy_score(true, pred)),
        "test_f1_macro": float(f1_score(true, pred, average="macro")),
        "test_log_loss": float(log_loss(true, probs, labels=labels)),
        "test_primary_metric": float(accuracy_score(true, pred)),
    }
    payload = {"actual": true, "predicted": pred, "probability": probs}
    return metrics, payload


def primary_metric_name(spec):
    if spec.task == "regression":
        return "test_rmse"
    if spec.task == "binary":
        return "test_auc"
    return "test_accuracy"


def primary_metric_higher_is_better(spec):
    return spec.task != "regression"


def primary_metric_value(row, spec):
    metric = primary_metric_name(spec)
    return float(row[metric])


def embedding_dims_for_args(args, vocab_sizes):
    return [int(args.embedding_size) for _ in vocab_sizes]


def embedding_size_payload(value):
    return int(value)


def build_model(encoding, num_numeric, vocab_sizes, args, output_dim):
    if encoding == "integer IDs":
        return DenseModel(num_numeric + len(vocab_sizes), args.hidden_sizes, args.dropout, output_dim)
    if encoding == "learned embeddings":
        return EmbeddingModel(num_numeric, vocab_sizes, embedding_dims_for_args(args, vocab_sizes), args.hidden_sizes, args.dropout, output_dim)
    raise AssertionError(f"Unknown encoding: {encoding}")


def inputs_for_encoding(encoding, train_num, train_cat, train_int, val_num, val_cat, val_int, test_num, test_cat, test_int):
    if encoding == "integer IDs":
        return (train_int,), (val_int,), (test_int,)
    return (train_num, train_cat), (val_num, val_cat), (test_num, test_cat)


def serializable_result(row):
    skip = {"curves", "prediction_payload", "embedding_weights"}
    payload = {}
    for key, value in row.items():
        if key not in skip:
            payload[key] = value
    return payload


def summarize_dataset(results, spec):
    rows = []
    for encoding in ENCODINGS:
        subset = [row for row in results if row["encoding"] == encoding]
        assert subset
        metric = primary_metric_name(spec)
        values = [row[metric] for row in subset]
        rows.append(
            {
                "dataset": spec.slug,
                "encoding": encoding,
                "task": spec.task,
                "primary_metric": metric,
                "primary_higher_is_better": primary_metric_higher_is_better(spec),
                "primary_mean": float(np.mean(values)),
                "primary_std": float(np.std(values, ddof=1)),
                "best_val_loss_mean": float(np.mean([row["best_val_loss"] for row in subset])),
                "params_mean": float(np.mean([row["params"] for row in subset])),
                "run_seconds_mean": float(np.mean([row["run_seconds"] for row in subset])),
                "throughput_mean": float(np.mean([row["train_examples_per_second"] for row in subset])),
            }
        )
    return rows


def build_curve_rows(results):
    rows = []
    for row in results:
        rows.extend(row["curves"])
    return rows


def build_prediction_rows(results, sample_count):
    rows = []
    for row in results:
        payload = row["prediction_payload"]
        actual = payload["actual"]
        predicted = payload["predicted"]
        count = min(sample_count, len(actual))
        for idx in range(count):
            next_row = {
                "dataset": row["dataset"],
                "encoding": row["encoding"],
                "seed": row["seed"],
                "sample_index": int(idx),
                "actual": float(actual[idx]),
                "predicted": float(predicted[idx]),
            }
            if "probability" in payload:
                probability = payload["probability"]
                if probability.ndim == 1:
                    next_row["probability"] = float(probability[idx])
                else:
                    next_row["probability_max"] = float(np.max(probability[idx]))
            rows.append(next_row)
    return rows


def bootstrap_ci(values, rng, iterations=2000):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    means = []
    for _ in range(iterations):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_comparisons(results, specs_by_slug, scope):
    rows = []
    rng = np.random.default_rng(20260625)
    dataset_slugs = sorted({row["dataset"] for row in results})
    for dataset_slug in dataset_slugs:
        spec = specs_by_slug[dataset_slug]
        metric = primary_metric_name(spec)
        higher = primary_metric_higher_is_better(spec)
        for first, second, label in METHOD_PAIRS:
            first_rows = {row["seed"]: row for row in results if row["dataset"] == dataset_slug and row["encoding"] == first}
            second_rows = {row["seed"]: row for row in results if row["dataset"] == dataset_slug and row["encoding"] == second}
            seeds = sorted(set(first_rows.keys()).intersection(set(second_rows.keys())))
            if len(seeds) < 2:
                continue
            first_values = np.array([first_rows[seed][metric] for seed in seeds], dtype=np.float64)
            second_values = np.array([second_rows[seed][metric] for seed in seeds], dtype=np.float64)
            raw_delta = first_values - second_values
            improvements = raw_delta if higher else -raw_delta
            ci_low, ci_high = bootstrap_ci(improvements, rng)
            t_result = ttest_rel(first_values, second_values)
            try:
                w_result = wilcoxon(improvements)
                wilcoxon_p = float(w_result[1])
            except ValueError:
                wilcoxon_p = 1.0
            rows.append(
                {
                    "scope": scope,
                    "dataset": dataset_slug,
                    "task": spec.task,
                    "metric": metric,
                    "higher_is_better": higher,
                    "first_encoding": first,
                    "second_encoding": second,
                    "pair_label": label,
                    "seed_count": int(len(seeds)),
                    "mean_paired_improvement": float(np.mean(improvements)),
                    "median_paired_improvement": float(np.median(improvements)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "win_count": int(np.sum(improvements > 0)),
                    "loss_count": int(np.sum(improvements < 0)),
                    "tie_count": int(np.sum(improvements == 0)),
                    "paired_t_p_value": float(t_result[1]),
                    "wilcoxon_p_value": wilcoxon_p,
                }
            )
    return rows


def aggregate_comparisons(results, specs_by_slug):
    rows = []
    rng = np.random.default_rng(20260626)
    for task in ["binary", "multiclass", "regression"]:
        task_results = [row for row in results if specs_by_slug[row["dataset"]].task == task]
        if not task_results:
            continue
        for first, second, label in METHOD_PAIRS:
            improvements = []
            for dataset_slug in sorted({row["dataset"] for row in task_results}):
                spec = specs_by_slug[dataset_slug]
                metric = primary_metric_name(spec)
                higher = primary_metric_higher_is_better(spec)
                first_rows = {row["seed"]: row for row in task_results if row["dataset"] == dataset_slug and row["encoding"] == first}
                second_rows = {row["seed"]: row for row in task_results if row["dataset"] == dataset_slug and row["encoding"] == second}
                seeds = sorted(set(first_rows.keys()).intersection(set(second_rows.keys())))
                for seed in seeds:
                    raw = float(first_rows[seed][metric] - second_rows[seed][metric])
                    improvements.append(raw if higher else -raw)
            if len(improvements) < 2:
                continue
            ci_low, ci_high = bootstrap_ci(improvements, rng)
            t_result = ttest_rel(np.asarray(improvements), np.zeros(len(improvements)))
            try:
                w_result = wilcoxon(improvements)
                wilcoxon_p = float(w_result[1])
            except ValueError:
                wilcoxon_p = 1.0
            rows.append(
                {
                    "scope": f"aggregate-{task}",
                    "dataset": "ALL",
                    "task": task,
                    "metric": "task_primary_metric",
                    "higher_is_better": True,
                    "first_encoding": first,
                    "second_encoding": second,
                    "pair_label": label,
                    "seed_count": int(len(improvements)),
                    "mean_paired_improvement": float(np.mean(improvements)),
                    "median_paired_improvement": float(np.median(improvements)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "win_count": int(np.sum(np.asarray(improvements) > 0)),
                    "loss_count": int(np.sum(np.asarray(improvements) < 0)),
                    "tie_count": int(np.sum(np.asarray(improvements) == 0)),
                    "paired_t_p_value": float(t_result[1]),
                    "wilcoxon_p_value": wilcoxon_p,
                }
            )
    return rows


def rank_table(summary_rows):
    rows = []
    for dataset_slug in sorted({row["dataset"] for row in summary_rows}):
        subset = [row for row in summary_rows if row["dataset"] == dataset_slug]
        higher = bool(subset[0]["primary_higher_is_better"])
        ordered = sorted(subset, key=lambda row: row["primary_mean"], reverse=higher)
        for rank, row in enumerate(ordered, start=1):
            rows.append(
                {
                    "dataset": dataset_slug,
                    "encoding": row["encoding"],
                    "rank": int(rank),
                    "primary_metric": row["primary_metric"],
                    "primary_mean": row["primary_mean"],
                }
            )
    return rows


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E2DACD", linewidth=0.8, alpha=0.7)


def plot_dataset_summary(summary_rows, path):
    labels = [row["encoding"] for row in summary_rows]
    values = [row["primary_mean"] for row in summary_rows]
    errors = [row["primary_std"] for row in summary_rows]
    colors = [PLOT_COLORS[label] for label in labels]
    metric = summary_rows[0]["primary_metric"]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, values, yerr=errors, color=colors, capsize=5)
    axis.set_title(f"{summary_rows[0]['dataset']} primary metric: {metric}")
    axis.tick_params(axis="x", rotation=18)
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(curve_rows, path):
    if not curve_rows:
        return
    frame = pd.DataFrame(curve_rows)
    fig, axis = plt.subplots(figsize=(10, 6))
    for encoding in ENCODINGS:
        subset = frame[frame["encoding"] == encoding]
        if len(subset) == 0:
            continue
        for seed in sorted(subset["seed"].unique()):
            seed_subset = subset[subset["seed"] == seed]
            axis.plot(seed_subset["train_examples_seen"], seed_subset["val_loss"], color=PLOT_COLORS[encoding], alpha=0.2)
        mean_subset = subset.groupby("train_examples_seen", as_index=False)["val_loss"].mean()
        axis.plot(mean_subset["train_examples_seen"], mean_subset["val_loss"], color=PLOT_COLORS[encoding], linewidth=2.5, label=encoding)
    axis.set_title("Validation loss by training examples seen")
    axis.set_xlabel("Training examples seen")
    axis.set_ylabel("Validation loss")
    axis.legend()
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(results, spec, path):
    fig, axes = plt.subplots(1, len(ENCODINGS), figsize=(5 * len(ENCODINGS), 5), squeeze=False)
    for axis, encoding in zip(axes[0], ENCODINGS):
        subset = [row for row in results if row["encoding"] == encoding]
        if not subset:
            axis.set_axis_off()
            continue
        best = sorted(subset, key=lambda row: row["best_val_loss"])[0]
        payload = best["prediction_payload"]
        actual = payload["actual"]
        predicted = payload["predicted"]
        count = min(2000, len(actual))
        if spec.task == "regression":
            axis.scatter(actual[:count], predicted[:count], s=10, alpha=0.45, color=PLOT_COLORS[encoding])
            low = float(min(np.min(actual[:count]), np.min(predicted[:count])))
            high = float(max(np.max(actual[:count]), np.max(predicted[:count])))
            axis.plot([low, high], [low, high], color="#17201C", linewidth=1)
            axis.set_xlabel("Actual")
            axis.set_ylabel("Predicted")
        elif spec.task == "binary":
            prob = payload["probability"]
            bins = np.linspace(0, 1, 11)
            centers = 0.5 * (bins[:-1] + bins[1:])
            observed = []
            for left, right in zip(bins[:-1], bins[1:]):
                mask = (prob >= left) & (prob < right)
                if np.any(mask):
                    observed.append(float(np.mean(actual[mask])))
                else:
                    observed.append(float("nan"))
            axis.plot(centers, observed, marker="o", color=PLOT_COLORS[encoding])
            axis.plot([0, 1], [0, 1], color="#17201C", linewidth=1)
            axis.set_xlabel("Predicted probability")
            axis.set_ylabel("Observed rate")
        else:
            accuracy = np.mean(actual == predicted)
            axis.bar(["correct", "wrong"], [accuracy, 1 - accuracy], color=[PLOT_COLORS[encoding], "#B83B2E"])
            axis.set_ylim(0, 1)
            axis.set_ylabel("Fraction")
        axis.set_title(f"{encoding}, seed {best['seed']}")
        style_axis(axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_embedding_projection(best_embedding_result, inv_maps, categorical, path):
    fig, axis = plt.subplots(figsize=(9, 7))
    if "embedding_weights" not in best_embedding_result:
        axis.set_axis_off()
    else:
        weights_list = best_embedding_result["embedding_weights"]
        sizes = [weights.shape[0] for weights in weights_list]
        feature_index = int(np.argmax(sizes))
        weights = weights_list[feature_index]
        labels = []
        inverse = inv_maps[categorical[feature_index]]
        for idx in range(1, min(weights.shape[0], 41)):
            labels.append(inverse[idx])
        if len(labels) >= 3:
            coords = PCA(n_components=2, random_state=20260624).fit_transform(weights[1 : len(labels) + 1])
            axis.scatter(coords[:, 0], coords[:, 1], color=PLOT_COLORS["learned embeddings"], s=40)
            for idx, label in enumerate(labels):
                axis.annotate(label, (coords[idx, 0], coords[idx, 1]), xytext=(4, 3), textcoords="offset points", fontsize=8)
            axis.set_title(f"Embedding PCA: {categorical[feature_index]}")
            axis.set_xlabel("PC1")
            axis.set_ylabel("PC2")
            style_axis(axis)
        else:
            axis.text(0.5, 0.5, "Not enough labels for PCA", ha="center", va="center")
            axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def color_for_label(label):
    text = str(label).lower()
    color_rules = [
        ("black", "#151515"),
        ("white", "#F2F0E8"),
        ("gray", "#8E969B"),
        ("grey", "#8E969B"),
        ("silver", "#BFC4C7"),
        ("blue", "#2F6FDB"),
        ("red", "#B83B2E"),
        ("green", "#3D7D4D"),
        ("brown", "#7A4E2C"),
        ("tan", "#C8A46A"),
        ("beige", "#D7C59D"),
        ("gold", "#DCA62B"),
        ("yellow", "#E3C64B"),
        ("orange", "#D8782C"),
        ("purple", "#7651A4"),
    ]
    for token, color in color_rules:
        if token in text:
            return color
    return "#5D6670"


def annotation_offset(index, coords):
    x_value = float(coords[index, 0])
    median_x = float(np.median(coords[:, 0]))
    dx = 9 if x_value <= median_x else -9
    dy_pattern = [7, -9, 19, -21, 31, -33]
    close_prior = 0
    for prior in range(index):
        distance = float(np.linalg.norm(coords[index] - coords[prior]))
        if distance < 0.55:
            close_prior += 1
    dy = dy_pattern[close_prior % len(dy_pattern)]
    horizontal_alignment = "left" if dx > 0 else "right"
    return dx, dy, horizontal_alignment


def plot_color_embedding_projection(best_embedding_result, inv_maps, categorical, path):
    color_features = [feature for feature in ("exterior_color", "interior_color") if feature in categorical]
    if not color_features or "embedding_weights" not in best_embedding_result:
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.text(0.5, 0.5, "No color embedding features available", ha="center", va="center")
        axis.set_axis_off()
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, len(color_features), figsize=(7.5 * len(color_features), 6.2), squeeze=False)
    weights_list = best_embedding_result["embedding_weights"]
    for axis, feature in zip(axes[0], color_features):
        feature_index = categorical.index(feature)
        weights = weights_list[feature_index]
        inverse = inv_maps[feature]
        labels = []
        for idx in range(1, min(weights.shape[0], 41)):
            labels.append(inverse[idx])
        if len(labels) < 3:
            axis.text(0.5, 0.5, "Not enough colors for PCA", ha="center", va="center")
            axis.set_axis_off()
            continue
        coords = PCA(n_components=2, random_state=20260624).fit_transform(weights[1 : len(labels) + 1])
        colors = [color_for_label(label) for label in labels]
        axis.scatter(coords[:, 0], coords[:, 1], color=colors, edgecolor="#17201C", linewidth=0.7, s=78, alpha=0.95, zorder=3)
        for idx, label in enumerate(labels):
            dx, dy, horizontal_alignment = annotation_offset(idx, coords)
            axis.annotate(
                label,
                (coords[idx, 0], coords[idx, 1]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9,
                ha=horizontal_alignment,
                va="center",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "#FAF8F2", "edgecolor": "#D5CCBD", "linewidth": 0.6},
                arrowprops={"arrowstyle": "-", "color": "#8A8174", "linewidth": 0.6, "shrinkA": 0, "shrinkB": 6},
                zorder=4,
            )
        axis.set_title(f"{feature.replace('_', ' ').title()} Embeddings")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        style_axis(axis)
    fig.suptitle("Learned Car Color Embeddings PCA", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_comparisons(comparison_rows, path, title):
    if not comparison_rows:
        return
    labels = [f"{row['dataset']}\n{row['pair_label']}" for row in comparison_rows]
    means = np.array([row["mean_paired_improvement"] for row in comparison_rows], dtype=np.float64)
    lows = np.array([row["ci95_low"] for row in comparison_rows], dtype=np.float64)
    highs = np.array([row["ci95_high"] for row in comparison_rows], dtype=np.float64)
    positions = np.arange(len(labels))
    fig_height = max(5, 0.35 * len(labels))
    fig, axis = plt.subplots(figsize=(11, fig_height))
    axis.errorbar(means, positions, xerr=np.vstack([means - lows, highs - means]), fmt="o", color="#17201C", ecolor="#2F6FDB", capsize=4)
    axis.axvline(0, color="#B83B2E", linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Mean paired improvement, positive favors first method")
    axis.set_title(title)
    axis.grid(axis="x", color="#E2DACD", linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_heatmap(summary_rows, path):
    if not summary_rows:
        return
    frame = pd.DataFrame(summary_rows)
    pivot = frame.pivot(index="dataset", columns="encoding", values="primary_mean")
    fig, axis = plt.subplots(figsize=(9, max(4, 0.5 * len(pivot))))
    values = pivot.to_numpy(dtype=np.float64)
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)))
    axis.set_yticklabels(pivot.index)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            axis.text(col_idx, row_idx, f"{values[row_idx, col_idx]:.3f}", ha="center", va="center", color="white", fontsize=8)
    axis.set_title("Primary metric by dataset and method")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def output_paths(base_dir, slug):
    dataset_dir = base_dir / slug
    plot_dir = dataset_dir / "plots"
    return {
        "dataset_dir": dataset_dir,
        "plot_dir": plot_dir,
        "metrics_json": dataset_dir / "metrics.json",
        "summary_csv": dataset_dir / "summary.csv",
        "runs_csv": dataset_dir / "runs.csv",
        "curves_csv": dataset_dir / "learning_curves.csv",
        "predictions_csv": dataset_dir / "prediction_samples.csv",
        "comparisons_csv": dataset_dir / "statistical_comparisons.csv",
        "summary_plot": plot_dir / "summary.png",
        "curves_plot": plot_dir / "learning_curves.png",
        "prediction_plot": plot_dir / "predictions.png",
        "embedding_plot": plot_dir / "embedding_projection.png",
        "color_embedding_plot": plot_dir / "color_embedding_projection.png",
        "comparison_plot": plot_dir / "method_comparisons.png",
    }


def dataframe_records(path):
    return pd.read_csv(path).to_dict("records")


def cached_result_complete(paths, args):
    required = ["metrics_json", "summary_csv", "runs_csv", "comparisons_csv"]
    for key in required:
        if not paths[key].is_file():
            return False
    try:
        payload = json.loads(paths["metrics_json"].read_text(encoding="utf-8"))
        runs = payload["runs"]
    except (json.JSONDecodeError, KeyError):
        return False
    expected_runs = len(ENCODINGS) * len(args.seeds)
    if len(runs) != expected_runs:
        return False
    observed = {(row["encoding"], int(row["seed"])) for row in runs}
    expected = {(encoding, int(seed)) for encoding in ENCODINGS for seed in args.seeds}
    return observed == expected


def load_cached_dataset_outputs(spec, paths):
    print(f"Using cached completed results for {spec.slug}: {paths['dataset_dir']}")
    payload = json.loads(paths["metrics_json"].read_text(encoding="utf-8"))
    results = payload["runs"]
    summaries = dataframe_records(paths["summary_csv"])
    comparisons = dataframe_records(paths["comparisons_csv"])
    return results, summaries, comparisons


def write_dataset_outputs(spec, payload, summary_rows, results, comparison_rows, inv_maps, categorical, paths, args):
    paths["dataset_dir"].mkdir(parents=True, exist_ok=True)
    paths["plot_dir"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(paths["summary_csv"], index=False)
    pd.DataFrame([serializable_result(row) for row in results]).to_csv(paths["runs_csv"], index=False)
    pd.DataFrame(build_curve_rows(results)).to_csv(paths["curves_csv"], index=False)
    pd.DataFrame(build_prediction_rows(results, args.sample_predictions)).to_csv(paths["predictions_csv"], index=False)
    pd.DataFrame(comparison_rows).to_csv(paths["comparisons_csv"], index=False)
    plot_dataset_summary(summary_rows, paths["summary_plot"])
    plot_learning_curves(build_curve_rows(results), paths["curves_plot"])
    plot_predictions(results, spec, paths["prediction_plot"])
    best_embedding = sorted([row for row in results if row["encoding"] == "learned embeddings"], key=lambda row: row["best_val_loss"])[0]
    plot_embedding_projection(best_embedding, inv_maps, categorical, paths["embedding_plot"])
    plot_color_embedding_projection(best_embedding, inv_maps, categorical, paths["color_embedding_plot"])
    plot_comparisons(comparison_rows, paths["comparison_plot"], f"{spec.slug} paired method comparisons")
    paths["metrics_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_aggregate_outputs(base_dir, all_results, all_summaries, all_comparisons, specs_by_slug):
    aggregate_dir = base_dir / "_aggregate"
    plot_dir = aggregate_dir / "plots"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    aggregate_comparison_rows = aggregate_comparisons(all_results, specs_by_slug)
    all_comparison_rows = all_comparisons + aggregate_comparison_rows
    ranks = rank_table(all_summaries)
    pd.DataFrame([serializable_result(row) for row in all_results]).to_csv(aggregate_dir / "master_runs.csv", index=False)
    pd.DataFrame(all_summaries).to_csv(aggregate_dir / "master_summary.csv", index=False)
    pd.DataFrame(ranks).to_csv(aggregate_dir / "method_ranks.csv", index=False)
    pd.DataFrame(all_comparison_rows).to_csv(aggregate_dir / "statistical_comparisons.csv", index=False)
    plot_aggregate_heatmap(all_summaries, plot_dir / "dataset_method_heatmap.png")
    plot_comparisons(all_comparison_rows, plot_dir / "aggregate_forest_plot.png", "Paired method comparisons")
    report_lines = [
        "# Categorical Benchmark Report",
        "",
        "## Datasets",
    ]
    for slug in sorted(specs_by_slug.keys()):
        spec = specs_by_slug[slug]
        report_lines.append(f"- {slug}: {spec.name}, task={spec.task}, source={spec.citation_url}")
    report_lines.extend(["", "## Mean Ranks"])
    rank_frame = pd.DataFrame(ranks)
    if len(rank_frame):
        rank_summary = rank_frame.groupby("encoding", as_index=False)["rank"].mean().sort_values("rank")
        for row in rank_summary.to_dict("records"):
            report_lines.append(f"- {row['encoding']}: mean rank {row['rank']:.2f}")
    report_lines.extend(["", "## Outputs", "- `master_runs.csv`", "- `master_summary.csv`", "- `statistical_comparisons.csv`", "- `plots/`"])
    (aggregate_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


def dataset_config_payload(spec, args, device, numeric, categorical, vocab_sizes, split_counts):
    backend, cudf_module = resolve_preprocess_backend(args)
    return {
        "dataset": spec.slug,
        "name": spec.name,
        "task": spec.task,
        "source_kind": spec.source_kind,
        "citation_url": spec.citation_url,
        "target": spec.target,
        "numeric_features": list(numeric),
        "categorical_features": list(categorical),
        "vocab_sizes": {categorical[idx]: int(vocab_sizes[idx]) for idx in range(len(categorical))},
        "split_counts": split_counts,
        "config": {
            "full_dataset": bool(args.full_dataset),
            "rows": int(args.rows),
            "validation_rows": int(args.validation_rows),
            "test_rows": int(args.test_rows),
            "eval_every_examples": int(args.eval_every_examples),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "optimizer": "AdamW",
            "embedding_size": embedding_size_payload(args.embedding_size),
            "embedding_dims": embedding_dims_for_args(args, vocab_sizes),
            "dropout": float(args.dropout),
            "hidden_sizes": [int(value) for value in args.hidden_sizes],
            "seeds": [int(seed) for seed in args.seeds],
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "category_min_count": int(args.category_min_count),
            "max_total_vocab": int(args.max_total_vocab),
            "max_vocab_per_column": int(args.max_vocab_per_column),
            "device": str(device),
            "precision": "float32",
            "feature_workers": int(args.feature_workers),
            "preprocess_backend": backend,
            "resolved_feature_workers": int(1 if backend == "cudf" else resolve_feature_workers(args, len(categorical))),
            "torch_version": torch.__version__,
        },
    }


def run_dataset(spec, args, device):
    dataset_start = time.perf_counter()
    cache_dir = args.output_dir / "_cache" / spec.slug
    paths = output_paths(args.output_dir, spec.slug)
    print(f"\n=== Dataset: {spec.slug} ===")
    if not args.loader_only and not args.rerun_completed and cached_result_complete(paths, args):
        return load_cached_dataset_outputs(spec, paths)
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
    payload_base = dataset_config_payload(spec, args, device, numeric, categorical, vocab_sizes, split_counts)
    if args.loader_only:
        payload = payload_base
        payload["loader_only"] = True
        payload["seconds"] = round(float(time.perf_counter() - dataset_start), 3)
        paths["dataset_dir"].mkdir(parents=True, exist_ok=True)
        paths["metrics_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return [], [], []

    output_dim = output_dim_for_task(spec.task, class_names)
    results = []
    for seed in tqdm(args.seeds, desc=f"{spec.slug} seeds", unit="seed"):
        for encoding in tqdm(ENCODINGS, desc=f"{spec.slug} encodings", unit="method", leave=False):
            set_seed(seed)
            model = build_model(encoding, train_num.shape[1], vocab_sizes, args, output_dim).to(device)
            train_inputs, val_inputs, test_inputs = inputs_for_encoding(
                encoding,
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
                encoding,
                model,
                seed,
                train_inputs,
                val_inputs,
                test_inputs,
                train_y,
                val_y,
                test_eval,
                spec,
                class_names,
                args,
                device,
            )
            if encoding == "learned embeddings":
                result["embedding_weights"] = [table.weight.detach().cpu().numpy() for table in model.embeddings]
            results.append(result)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary_rows = summarize_dataset(results, spec)
    comparison_rows = paired_comparisons(results, {spec.slug: spec}, "dataset")
    payload = payload_base
    payload["summary"] = summary_rows
    payload["runs"] = [serializable_result(row) for row in results]
    payload["statistical_comparisons"] = comparison_rows
    payload["seconds"] = round(float(time.perf_counter() - dataset_start), 3)
    write_dataset_outputs(spec, payload, summary_rows, results, comparison_rows, inv_maps, categorical, paths, args)
    return results, summary_rows, comparison_rows


def main():
    args = parse_args()
    validate_args(args)
    configure_torch(args)
    device = resolve_device(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slugs = selected_dataset_slugs(args)
    specs = [DATASET_REGISTRY[slug] for slug in slugs]
    specs_by_slug = {spec.slug: spec for spec in specs}
    print(f"Running datasets: {', '.join(slugs)}")
    print(f"Device: {device}; precision=float32; batch_size={args.batch_size}; workers={args.num_workers}")
    all_results = []
    all_summaries = []
    all_comparisons = []
    for spec in specs:
        results, summaries, comparisons = run_dataset(spec, args, device)
        all_results.extend(results)
        all_summaries.extend(summaries)
        all_comparisons.extend(comparisons)
    if not args.loader_only and all_results:
        write_aggregate_outputs(args.output_dir, all_results, all_summaries, all_comparisons, specs_by_slug)
    print("Categorical benchmark complete.")


if __name__ == "__main__":
    main()

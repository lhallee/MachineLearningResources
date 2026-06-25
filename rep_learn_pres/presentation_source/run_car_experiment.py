import sys

from run_categorical_benchmark import main


def has_dataset_selection(argv):
    for arg in argv:
        if arg == "--all-datasets":
            return True
        if arg == "--datasets":
            return True
        if arg.startswith("--datasets="):
            return True
    return False


if __name__ == "__main__":
    if not has_dataset_selection(sys.argv[1:]):
        sys.argv.extend(["--datasets", "car-price"])
    main()

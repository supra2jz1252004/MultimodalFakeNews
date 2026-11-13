"""Run the controlled Fakeddit experiment from the project root."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)

    commands.add_parser("audit")
    commands.add_parser("download")

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--train", type=int, default=50000)
    prepare.add_argument("--validation", type=int, default=2500)
    prepare.add_argument("--test", type=int, default=2500)
    prepare.add_argument("--workers", type=int, default=24)
    prepare.add_argument(
        "--image-cache-manifest",
        type=Path,
        action="append",
        default=[],
    )

    extract = commands.add_parser("extract-features")
    extract.add_argument("--smoke", action="store_true")
    extract.add_argument("--batch-size", type=int, default=16)
    extract.add_argument("--threads", type=int, default=6)

    commands.add_parser("smoke")
    commands.add_parser("search")

    analysis = commands.add_parser("analyze-similarity")
    analysis.add_argument("--smoke", action="store_true")

    commands.add_parser("evaluate")
    commands.add_parser("plots")

    # Manual prediction on a new unseen text-image pair.
    predict = commands.add_parser("predict")
    predict.add_argument("--text", required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument(
        "--seed",
        type=int,
        choices=[42, 123, 2026],
        default=42,
    )

    complete = commands.add_parser("all")
    complete.add_argument("--train", type=int, default=50000)
    complete.add_argument("--validation", type=int, default=2500)
    complete.add_argument("--test", type=int, default=2500)

    training = commands.add_parser("train")
    training.add_argument(
        "--model",
        choices=["all", "text", "image", "baseline", "proposed"],
        default="all",
    )

    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    from src.data import metadata_audit, prepare as prepare_data

    if args.stage == "audit":
        from src.data import audit_acquired

        metadata = metadata_audit(root)
        audit_acquired(root, metadata)

    elif args.stage == "download":
        from src.data import download_metadata

        download_metadata(root)

    elif args.stage == "prepare":
        prepare_data(
            root,
            {
                "train": args.train,
                "validation": args.validation,
                "test": args.test,
            },
            args.workers,
            args.image_cache_manifest,
        )

    elif args.stage == "extract-features":
        from src.feature_extractors import extract

        extract(
            root,
            args.smoke,
            args.batch_size,
            args.threads,
        )

    elif args.stage == "smoke":
        from src.train import smoke_test

        smoke_test(root)

    elif args.stage == "search":
        from src.train import search

        search(root)

    elif args.stage == "train":
        from src.train import train

        train(root, args.model)

    elif args.stage == "analyze-similarity":
        from src.similarity import analyze

        analyze(root, args.smoke)

    elif args.stage == "evaluate":
        from src.evaluate import evaluate

        evaluate(root)

    elif args.stage == "plots":
        from src.plots import plots

        plots(root)

    elif args.stage == "predict":
        from src.predict import predict_pair

        predict_pair(
            root,
            args.text,
            args.image,
            args.seed,
        )

    elif args.stage == "all":
        from src.data import download_metadata, audit_acquired
        from src.feature_extractors import extract
        from src.similarity import analyze
        from src.train import search, train, smoke_test
        from src.evaluate import evaluate
        from src.plots import plots
        from src.utils import file_hash

        download_metadata(root)

        if not (root / "results/metrics/smoke.json").exists():
            prepare_data(
                root,
                {
                    "train": 300,
                    "validation": 50,
                    "test": 50,
                },
            )

            shutil.copy2(
                root / "data/processed/manifest.csv",
                root / "data/processed/smoke_manifest.csv",
            )

            shutil.copy2(
                root / "data/processed/selection.json",
                root / "data/processed/smoke_selection.json",
            )

            extract(root, smoke=True)
            analyze(root, smoke=True)
            smoke_test(root)

        counts = {
            "train": args.train,
            "validation": args.validation,
            "test": args.test,
        }

        selection_path = root / "data/processed/selection.json"

        selection = (
            json.loads(selection_path.read_text())
            if selection_path.exists()
            else {}
        )

        manifest_path = root / "data/processed/manifest.csv"

        if (
            selection.get("counts") != counts
            or not manifest_path.is_file()
            or selection.get("manifest_sha256")
            != file_hash(manifest_path)
        ):
            prepare_data(root, counts)

        audit_acquired(
            root,
            metadata_audit(root),
        )

        extract(root)
        analyze(root)

        if not (root / "models/locked_config.json").exists():
            search(root)

        train(root)
        evaluate(root)
        plots(root)


if __name__ == "__main__":
    main()
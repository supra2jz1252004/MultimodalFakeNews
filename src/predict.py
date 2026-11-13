"""Run frozen Fakeddit A/B heads on a supplied text–image pair."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50
from transformers import (
    AutoTokenizer,
    BertModel,
    CLIPModel,
    CLIPProcessor,
)

from .feature_extractors import (
    BERT,
    BERT_REVISION,
    CLIP,
    CLIP_REVISION,
    encode,
)
from .models import Classifier
from .utils import file_hash


def predict_pair(
    root: Path,
    text: str,
    image_path: Path,
    seed: int = 42,
) -> dict:
    """Return A/B labels, softmax scores and the raw CLIP cosine.

    Reuses the saved encoder revisions, training scaler and checkpoint contract.
    The caller supplies the pair; this function does not establish that it was
    absent from training or that its claim is factually correct.
    """

    text = text.strip()

    if not text:
        raise ValueError("Text cannot be empty.")

    image_path = Path(image_path).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    lock_path = root / "models" / "locked_config.json"
    normalization_path = root / "models" / "normalization.npz"
    extraction_path = root / "data" / "features" / "extraction.json"

    if not lock_path.is_file():
        raise FileNotFoundError("models/locked_config.json is missing.")

    if not normalization_path.is_file():
        raise FileNotFoundError("models/normalization.npz is missing.")

    if not extraction_path.is_file():
        raise FileNotFoundError("data/features/extraction.json is missing.")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))

    if seed not in lock["seeds"]:
        raise ValueError(
            f"Seed {seed} is unavailable. "
            f"Available seeds: {lock['seeds']}"
        )

    max_length = int(extraction["contract"]["bert_max_length"])

    torch.set_num_threads(lock["budget"]["threads"])

    # Reuse the original encoder path with a one-row batch.
    sample = pd.DataFrame(
        [
            {
                "text": text,
                "image_path": str(image_path),
            }
        ]
    )

    print("Loading BERT...")
    tokenizer = AutoTokenizer.from_pretrained(
        BERT,
        revision=BERT_REVISION,
    )

    bert = BertModel.from_pretrained(
        BERT,
        revision=BERT_REVISION,
    )

    bert.eval().requires_grad_(False)

    text_feature = encode(
        "text",
        sample,
        root,
        tokenizer,
        bert,
        max_length,
    )

    del bert
    del tokenizer

    print("Loading ResNet50...")
    weights = ResNet50_Weights.IMAGENET1K_V2

    image_transform = weights.transforms()

    resnet = resnet50(weights=weights)
    resnet.fc = torch.nn.Identity()

    resnet.eval().requires_grad_(False)

    image_feature = encode(
        "image",
        sample,
        root,
        image_transform,
        resnet,
        max_length,
    )

    del resnet

    print("Loading CLIP...")
    clip_processor = CLIPProcessor.from_pretrained(
        CLIP,
        revision=CLIP_REVISION,
        use_fast=False,
    )

    clip_model = CLIPModel.from_pretrained(
        CLIP,
        revision=CLIP_REVISION,
    )

    clip_model.eval().requires_grad_(False)

    similarity_feature = encode(
        "similarity",
        sample,
        root,
        clip_processor,
        clip_model,
        max_length,
    )

    del clip_model
    del clip_processor

    raw_similarity = float(similarity_feature[0, 0])

    raw_features = {
        "text": text_feature,
        "image": image_feature,
        "similarity": similarity_feature,
    }

    # Apply saved training moments; never fit a scaler to this input.
    normalized = {}

    with np.load(normalization_path, allow_pickle=False) as saved:
        for kind, values in raw_features.items():
            mean = saved[f"{kind}_mean"]
            std = saved[f"{kind}_std"]

            transformed = (values - mean) / std

            if not np.isfinite(transformed).all():
                raise RuntimeError(
                    f"Non-finite values found in {kind} features."
                )

            normalized[kind] = torch.from_numpy(
                transformed.astype(np.float32)
            )

    results = {}

    for kind in ("baseline", "proposed"):
        checkpoint_path = (
            root / "models" / "final" / f"{kind}_{seed}.pt"
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint missing: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        # Check checkpoint identity and its saved input contracts.
        if checkpoint["model"] != kind:
            raise RuntimeError("Checkpoint model type mismatch.")

        if checkpoint["seed"] != seed:
            raise RuntimeError("Checkpoint seed mismatch.")

        if checkpoint["config"] != lock["config"]:
            raise RuntimeError("Checkpoint configuration mismatch.")

        if checkpoint["manifest_sha256"] != file_hash(
            root / "data" / "processed" / "manifest.csv"
        ):
            raise RuntimeError("Dataset manifest does not match checkpoint.")

        if checkpoint["normalization_sha256"] != file_hash(
            normalization_path
        ):
            raise RuntimeError(
                "Normalization file does not match checkpoint."
            )

        if checkpoint["feature_contract"] != extraction["contract"]:
            raise RuntimeError(
                "Feature extraction contract does not match checkpoint."
            )

        model = Classifier(
            kind,
            lock["config"]["dimension"],
            lock["config"]["dropout"],
        )

        model.load_state_dict(
            checkpoint["state_dict"],
            strict=True,
        )

        model.eval()

        with torch.inference_mode():
            logits = model(
                normalized["text"],
                normalized["image"],
                normalized["similarity"],
            )

            probabilities = F.softmax(logits, dim=1)[0]

        fake_probability = float(probabilities[0])
        real_probability = float(probabilities[1])

        predicted_label = int(torch.argmax(probabilities))

        results[kind] = {
            "label": predicted_label,
            "prediction": "FAKE" if predicted_label == 0 else "REAL",
            "probability_fake": fake_probability,
            "probability_real": real_probability,
            "confidence": max(fake_probability, real_probability),
        }

    print()
    print("=" * 60)
    print("MANUAL UNSEEN NEWS PREDICTION")
    print("=" * 60)

    print(f"Text: {text}")
    print(f"Image: {image_path}")

    print()
    print(f"CLIP cosine similarity: {raw_similarity:.6f}")

    print()
    print("BASELINE — BERT + ResNet50")
    print(
        f"Prediction: {results['baseline']['prediction']}"
    )
    print(
        f"Fake probability: "
        f"{results['baseline']['probability_fake']:.4f}"
    )
    print(
        f"Real probability: "
        f"{results['baseline']['probability_real']:.4f}"
    )

    print()
    print("PROPOSED — BERT + ResNet50 + CLIP similarity")
    print(
        f"Prediction: {results['proposed']['prediction']}"
    )
    print(
        f"Fake probability: "
        f"{results['proposed']['probability_fake']:.4f}"
    )
    print(
        f"Real probability: "
        f"{results['proposed']['probability_real']:.4f}"
    )

    print()
    print(
        "Note: probabilities are model softmax scores, "
        "not verified factual truth probabilities."
    )
    print("=" * 60)

    return {
        "text": text,
        "image": str(image_path),
        "clip_similarity": raw_similarity,
        "baseline": results["baseline"],
        "proposed": results["proposed"],
    }
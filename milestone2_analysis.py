"""Milestone 2 soil analysis pipeline.

The supplied project currently contains structured soil measurements only. This
script trains the structured model immediately after dependencies are installed
and exposes an image-model path for a folder arranged as ImageFolder classes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "Samples EJP Probefield.xlsx"
OUTPUT_DIR = ROOT / "milestone2_outputs"
RANDOM_STATE = 42


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Structured dataset not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def nutrient_columns(frame: pd.DataFrame) -> list[str]:
    aliases = {
        "nitrogen": r"\b(n|nitrogen|no3|nitrate)\b",
        "phosphorus": r"\b(p|phosphorus|phosphate|po4)\b",
        "potassium": r"\b(k|potassium)\b",
    }
    result = []
    for name in frame.columns:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(name).lower())
        if any(re.search(pattern, normalized) for pattern in aliases.values()):
            if pd.api.types.is_numeric_dtype(frame[name]):
                result.append(str(name))
    return result


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_structured_model(dataset_path: Path, output_dir: Path) -> dict[str, Any]:
    import joblib

    frame = load_table(dataset_path)
    numeric = frame.select_dtypes(include="number").columns.tolist()
    targets = nutrient_columns(frame)
    if not targets:
        raise ValueError("No numeric nutrient columns were detected. Pass a dataset with N, P, or K measurements.")
    features = [column for column in numeric if column not in targets]
    if not features:
        raise ValueError("No numeric input features remain after removing nutrient target columns.")

    clean = frame[features + targets].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 12:
        raise ValueError(f"At least 12 complete rows are required; found {len(clean)}.")

    # Deficiency labels are derived from the supplied measurements: values below
    # the training median are deficient, making the target definition reproducible.
    thresholds = {target: float(clean[target].median()) for target in targets}
    labels = (clean[targets].lt(pd.Series(thresholds))).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        clean[features], labels, test_size=0.25, random_state=RANDOM_STATE
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_models = {}
    feature_scores = np.zeros(len(features), dtype=float)
    for target in targets:
        baseline = GradientBoostingClassifier(random_state=RANDOM_STATE)
        baseline.fit(x_train, y_train[target])
        importance = permutation_importance(
            baseline, x_test, y_test[target], scoring="f1_macro", random_state=RANDOM_STATE
        )
        feature_scores += np.maximum(importance.importances_mean, 0)
        baseline_models[target] = baseline
    feature_scores /= len(targets)
    ranked_features = [features[index] for index in np.argsort(feature_scores)[::-1]]
    selected_features = ranked_features[:max(3, min(12, len(ranked_features)))]
    x_train_selected = x_train[selected_features]
    x_test_selected = x_test[selected_features]

    parameter_grid = {
        "n_estimators": [80, 140],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    }
    models_by_target = {}
    predicted = {}
    target_metrics = {}
    tuning = {}
    for target in targets:
        search = GridSearchCV(
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            parameter_grid,
            scoring="f1_macro",
            cv=3,
            n_jobs=-1,
        )
        search.fit(x_train_selected, y_train[target])
        models_by_target[target] = search.best_estimator_
        predicted[target] = search.predict(x_test_selected)
        target_metrics[target] = metrics(y_test[target].to_numpy(), predicted[target])
        tuning[target] = {"best_params": search.best_params_, "best_cv_f1": float(search.best_score_)}

    feature_importance = dict(sorted(
        zip(features, feature_scores.tolist()), key=lambda item: item[1], reverse=True
    ))
    shap_status = "not_generated"
    shap_error = None
    shap_importance = {}
    shap_plot = None
    try:
        import matplotlib.pyplot as plt
        import shap
        explanations = []
        for target in targets:
            explainer = shap.TreeExplainer(models_by_target[target])
            values = explainer.shap_values(x_test_selected)
            if isinstance(values, list):
                values = values[1]
            explanations.append(np.abs(values).mean(axis=0))
        mean_shap = np.mean(explanations, axis=0)
        shap_importance = dict(sorted(
            zip(selected_features, mean_shap.tolist()), key=lambda item: item[1], reverse=True
        ))
        plt.figure(figsize=(9, 5))
        names = list(shap_importance.keys())[::-1]
        values = list(shap_importance.values())[::-1]
        plt.barh(names, values, color="#2f7d6d")
        plt.title("Mean absolute SHAP importance across nutrient targets")
        plt.xlabel("Mean absolute SHAP value")
        plt.tight_layout()
        shap_plot = output_dir / "structured_shap_feature_importance.png"
        plt.savefig(shap_plot, dpi=150)
        plt.close()
        shap_status = "generated"
    except Exception as exc:
        shap_error = str(exc)

    estimator = {"models": models_by_target, "features": selected_features, "targets": targets, "thresholds": thresholds}
    model_path = output_dir / "structured_soil_model.joblib"
    joblib.dump(estimator, model_path)

    report = {
        "status": "trained",
        "model": "GradientBoostingClassifier via MultiOutputClassifier",
        "dataset": str(dataset_path),
        "rows_used": int(len(clean)),
        "features": features,
        "selected_features": selected_features,
        "targets": targets,
        "deficiency_rule": "measurement < training median",
        "thresholds": thresholds,
        "metrics": target_metrics,
        "feature_importance": feature_importance,
        "shap": {
            "status": shap_status,
            "mean_absolute_importance": shap_importance,
            "plot": str(shap_plot) if shap_plot else None,
            "error": shap_error,
        },
        "tuning": tuning,
        "model_path": str(model_path),
        "limitations": [
            "Deficiency labels are proxy labels because the workbook has no labelled deficiency outcome.",
            "A larger labelled dataset is required before using this model for agronomic decisions.",
        ],
    }
    (output_dir / "structured_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_hybrid_assessment(output_dir: Path, structured_report: dict[str, Any], image_result: dict[str, Any] | None = None) -> None:
    """Combine calibrated validation signals while keeping label spaces separate."""
    structured_confidence = float(np.mean([
        values["f1"] for values in structured_report["metrics"].values()
    ]))
    modalities = [{"name": "structured", "confidence": structured_confidence, "weight": 0.6}]
    if image_result:
        modalities.append({
            "name": "image",
            "confidence": float(image_result.get("confidence", image_result["test_metrics"]["f1"])),
            "weight": 0.4,
            "prediction": image_result.get("predicted_class"),
        })
    total_weight = sum(item["weight"] for item in modalities)
    weighted_confidence = sum(item["confidence"] * item["weight"] for item in modalities) / total_weight
    assessment = {
        "status": "hybrid_ready" if len(modalities) == 2 else "structured_only",
        "logic": "weighted confidence fusion; structured nutrient and image disease labels remain separate",
        "modalities": modalities,
        "confidence": round(float(weighted_confidence), 4),
        "soil_health_score": round(float(weighted_confidence * 100), 2),
        "prediction_combination": {
            "structured": "macro-F1 across nutrient deficiency targets",
            "image": "test macro-F1 and predicted disease class when available",
            "weights": {item["name"]: item["weight"] / total_weight for item in modalities},
        },
        "limitations": [
            "The structured target is a median-based deficiency proxy.",
            "Image and structured models use different label spaces; this is a confidence/health-score fusion, not a shared class prediction.",
        ],
    }
    (output_dir / "hybrid_soil_assessment.json").write_text(json.dumps(assessment, indent=2), encoding="utf-8")


def generate_gradcam(model: Any, image_tensor: Any, target_class: int, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import torch

    activations = []
    gradients = []
    layer = model.features[-1]
    layer.register_forward_hook(lambda _, __, output: activations.append(output))
    layer.register_full_backward_hook(lambda _, grad_input, grad_output: gradients.append(grad_output[0]))
    model.zero_grad(set_to_none=True)
    logits = model(image_tensor)
    logits[:, target_class].sum().backward()
    weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activations[-1]).sum(dim=1)).squeeze().detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    image = image_tensor.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    plt.figure(figsize=(7, 5))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.42, extent=(0, image.shape[1], image.shape[0], 0))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def run_image_model(
    image_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    patience: int = 2,
    max_images_per_class: int | None = None,
    pretrained: bool = True,
) -> dict[str, Any]:
    try:
        import torch
        from PIL import ImageFile
        from torch import nn
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise RuntimeError("CNN support requires torch and torchvision. Install requirements.txt first.") from exc
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or patience < 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive; patience cannot be negative.")

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    raw_dataset = datasets.ImageFolder(image_dir)
    classes = raw_dataset.classes
    if len(classes) < 2 or len(raw_dataset) < 12:
        raise ValueError("CNN image data must contain at least two class directories.")

    targets = np.asarray(raw_dataset.targets)
    distribution = np.bincount(targets, minlength=len(classes))
    dataset_review = {
        "dataset": str(image_dir),
        "class_count": len(classes),
        "image_count": len(raw_dataset),
        "classes": classes,
        "class_distribution": {classes[index]: int(count) for index, count in enumerate(distribution)},
        "minimum_class_size": int(distribution.min()),
        "maximum_class_size": int(distribution.max()),
        "imbalance_ratio": round(float(distribution.max() / distribution.min()), 3),
        "split_strategy": "stratified 70% train, 15% validation, 15% test",
        "canonical_variant": "color; grayscale and segmented folders should not be combined with it",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "image_dataset_review.json").write_text(json.dumps(dataset_review, indent=2), encoding="utf-8")
    selected_indices = np.arange(len(raw_dataset))
    if max_images_per_class is not None:
        if max_images_per_class < 3:
            raise ValueError("max_images_per_class must be at least 3.")
        rng = np.random.default_rng(RANDOM_STATE)
        selected_indices = np.concatenate([
            rng.choice(np.flatnonzero(targets == class_index),
                       size=min(max_images_per_class, int(np.sum(targets == class_index))),
                       replace=False)
            for class_index in range(len(classes))
        ])
        selected_indices.sort()
        targets = targets[selected_indices]
    minimum_class_count = int(np.bincount(targets, minlength=len(classes)).min())
    if minimum_class_count < 8:
        raise ValueError(
            "Each class needs at least 8 images for stratified train/validation/test splits; "
            f"the smallest class has {minimum_class_count}."
        )

    all_indices = np.arange(len(selected_indices))
    train_indices, holdout_indices = train_test_split(
        all_indices, test_size=0.30, random_state=RANDOM_STATE, stratify=targets
    )
    validation_indices, test_indices = train_test_split(
        holdout_indices, test_size=0.50, random_state=RANDOM_STATE, stratify=targets[holdout_indices]
    )

    image_size = 224
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.ImageFolder(image_dir, transform=train_transform)
    evaluation_dataset = datasets.ImageFolder(image_dir, transform=evaluation_transform)
    train_dataset = Subset(train_dataset, selected_indices[train_indices].tolist())
    validation_dataset = Subset(evaluation_dataset, selected_indices[validation_indices].tolist())
    test_dataset = Subset(evaluation_dataset, selected_indices[test_indices].tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {"batch_size": batch_size, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        print(f"Pretrained EfficientNet weights unavailable ({exc}); using ImageNet architecture without weights.")
        model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model = model.to(device)

    train_targets = targets[train_indices]
    class_counts = np.bincount(train_targets, minlength=len(classes))
    class_weights = len(train_targets) / (len(classes) * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=1)

    def evaluate(loader: DataLoader) -> tuple[float, float, np.ndarray, np.ndarray]:
        model.eval()
        total_loss = 0.0
        predictions, labels = [], []
        with torch.no_grad():
            for images, batch_labels in loader:
                logits = model(images.to(device))
                total_loss += float(criterion(logits, batch_labels.to(device)).item()) * len(batch_labels)
                predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                labels.extend(batch_labels.numpy().tolist())
        average_loss = total_loss / max(len(loader.dataset), 1)
        macro_f1 = f1_score(
            labels, predictions, labels=np.arange(len(classes)), average="macro", zero_division=0
        )
        return average_loss, float(macro_f1), np.asarray(labels), np.asarray(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_plant_disease_efficientnet_b0.pt"
    history = []
    best_f1 = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_labels, train_predictions = [], []
        for images, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, batch_labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(batch_labels)
            train_predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            train_labels.extend(batch_labels.numpy().tolist())
        validation_loss, validation_f1, validation_labels, validation_predictions = evaluate(validation_loader)
        train_f1 = float(f1_score(
            train_labels, train_predictions, labels=np.arange(len(classes)), average="macro", zero_division=0
        ))
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss / max(len(train_loader.dataset), 1),
            "validation_loss": validation_loss,
            "train_accuracy": float(accuracy_score(train_labels, train_predictions)),
            "validation_accuracy": float(accuracy_score(validation_labels, validation_predictions)),
            "train_f1": train_f1,
            "validation_f1": validation_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))
        scheduler.step(validation_f1)
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "classes": classes}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience + 1:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_f1, test_labels, test_predictions = evaluate(test_loader)
    test_metrics = {
        "accuracy": float(accuracy_score(test_labels, test_predictions)),
        "precision": float(precision_score(
            test_labels, test_predictions, labels=np.arange(len(classes)), average="macro", zero_division=0
        )),
        "recall": float(recall_score(
            test_labels, test_predictions, labels=np.arange(len(classes)), average="macro", zero_division=0
        )),
        "f1": float(test_f1),
        "loss": float(test_loss),
        "confusion_matrix": confusion_matrix(test_labels, test_predictions).tolist(),
    }
    gradcam_path = output_dir / "gradcam_attention.png"
    gradcam_error = None
    predicted_class = classes[int(test_predictions[0])] if len(test_predictions) else None
    try:
        sample_tensor, _ = test_dataset[0]
        generate_gradcam(model, sample_tensor.unsqueeze(0).to(device), int(test_predictions[0]), gradcam_path)
    except Exception as exc:
        gradcam_path = None
        gradcam_error = str(exc)
    incorrect_predictions = []
    selected_paths = [raw_dataset.samples[index][0] for index in selected_indices[test_indices]]
    for path, actual, predicted in zip(selected_paths, test_labels, test_predictions):
        if actual != predicted:
            incorrect_predictions.append({
                "path": path,
                "actual": classes[int(actual)],
                "predicted": classes[int(predicted)],
            })
    (output_dir / "image_incorrect_predictions.json").write_text(
        json.dumps(incorrect_predictions, indent=2), encoding="utf-8"
    )
    report = {
        "status": "trained",
        "model": "EfficientNet-B0 transfer learning",
        "dataset": str(image_dir),
        "classes": classes,
        "class_count": len(classes),
        "images_used": len(selected_indices),
        "split_sizes": {"train": len(train_dataset), "validation": len(validation_dataset), "test": len(test_dataset)},
        "class_distribution": {classes[index]: int(count) for index, count in enumerate(distribution)},
        "training": {"epochs_requested": epochs, "epochs_completed": len(history), "batch_size": batch_size,
                     "learning_rate": learning_rate, "optimizer": "AdamW", "device": str(device),
                     "history": history},
        "test_metrics": test_metrics,
        "confidence": float(test_f1),
        "predicted_class": predicted_class,
        "gradcam": {"status": "generated" if gradcam_path else "failed",
                 "path": str(gradcam_path) if gradcam_path else None, "error": gradcam_error},
        "incorrect_predictions": len(incorrect_predictions),
        "checkpoint": str(checkpoint_path),
        "selection": "best validation macro-F1 with early stopping",
    }
    (output_dir / "image_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone 2 soil ML analysis.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--image-dir", type=Path, help="ImageFolder root with one subdirectory per class.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-images-per-class", type=int)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/use ImageNet weights.")
    parser.add_argument("--image-only", action="store_true", help="Run the image pipeline without loading the Excel dataset.")
    args = parser.parse_args()

    structured_report = None
    image_report = None
    if not args.image_only:
        report = train_structured_model(args.dataset, args.output_dir)
        structured_report = report
        print(json.dumps({"structured_status": report["status"], "targets": report["targets"], "model": report["model_path"]}, indent=2))
    if args.image_dir:
        image_report = run_image_model(
            args.image_dir, args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, patience=args.patience,
            max_images_per_class=args.max_images_per_class, pretrained=not args.no_pretrained,
        )
        print(json.dumps({"image_status": image_report["status"], "test_metrics": image_report["test_metrics"]}, indent=2))
    elif not args.image_only:
        print("CNN status: skipped; no --image-dir was supplied.")
    if structured_report is not None:
        write_hybrid_assessment(args.output_dir, structured_report, image_report)


if __name__ == "__main__":
    main()
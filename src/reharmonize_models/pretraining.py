from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MaskedChordModel(nn.Module):
    """Predict a hidden chord from melody and surrounding harmony."""

    def __init__(
        self,
        quality_classes: int,
        sequence_length: int,
        *,
        hidden_size: int = 96,
        attention_heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.root_embedding = nn.Embedding(15, hidden_size)
        self.quality_embedding = nn.Embedding(quality_classes + 2, hidden_size)
        self.melody_projection = nn.Linear(12, hidden_size)
        self.scalar_projection = nn.Linear(2, hidden_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size, attention_heads, hidden_size * 4, dropout,
            batch_first=True, norm_first=False, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.normalization = nn.LayerNorm(hidden_size)
        self.root_head = nn.Linear(hidden_size, 13)
        self.quality_head = nn.Linear(hidden_size, quality_classes)

    def forward(
        self,
        root_input: torch.Tensor,
        quality_input: torch.Tensor,
        melody: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = (
            self.root_embedding(root_input)
            + self.quality_embedding(quality_input)
            + self.melody_projection(melody)
            + self.scalar_projection(scalars)
            + self.position_embedding
        )
        encoded = self.encoder(encoded)
        center = self.normalization(encoded[:, self.sequence_length // 2])
        return self.root_head(center), self.quality_head(center)


def _load_split(path: Path) -> TensorDataset:
    with np.load(path) as values:
        arrays = [
            values["root_input"], values["quality_input"], values["melody"],
            values["scalars"], values["target_root"], values["target_quality"],
            values["sample_weight"],
        ]
    return TensorDataset(
        torch.from_numpy(arrays[0]).long(),
        torch.from_numpy(arrays[1]).long(),
        torch.from_numpy(arrays[2]).float(),
        torch.from_numpy(arrays[3]).float(),
        torch.from_numpy(arrays[4]).long(),
        torch.from_numpy(arrays[5]).long(),
        torch.from_numpy(arrays[6]).float(),
    )


def _weighted_prediction_metrics(
    predicted_root: np.ndarray,
    predicted_quality: np.ndarray,
    target_root: np.ndarray,
    target_quality: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    denominator = max(float(weight.sum()), 1e-8)
    root_correct = predicted_root == target_root
    quality_correct = predicted_quality == target_quality
    return {
        "root_accuracy": float((root_correct * weight).sum() / denominator),
        "quality_accuracy": float((quality_correct * weight).sum() / denominator),
        "chord_accuracy": float(((root_correct & quality_correct) * weight).sum() / denominator),
    }


def calculate_pretraining_baselines(
    dataset_root: str | Path, *, split: str = "test",
) -> dict[str, object]:
    """Measure simple work-balanced baselines for masked-chord accuracy."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    vocabulary = json.loads((dataset_root / "vocabulary.json").read_text())
    quality_classes = len(vocabulary["qualities"])
    with np.load(dataset_root / "train.npz") as train:
        train_root = train["target_root"]
        train_quality = train["target_quality"]
        train_weight = train["sample_weight"]
    with np.load(dataset_root / f"{split}.npz") as evaluation:
        root_input = evaluation["root_input"]
        quality_input = evaluation["quality_input"]
        target_root = evaluation["target_root"]
        target_quality = evaluation["target_quality"]
        weight = evaluation["sample_weight"]

    root_counts = np.bincount(train_root, weights=train_weight, minlength=13)
    quality_counts = np.bincount(
        train_quality, weights=train_weight, minlength=quality_classes
    )
    joint_counts = np.bincount(
        train_root * quality_classes + train_quality,
        weights=train_weight,
        minlength=13 * quality_classes,
    )
    majority_root = int(root_counts.argmax())
    majority_quality = int(quality_counts.argmax())
    majority_joint = int(joint_counts.argmax())
    majority_chord_root = majority_joint // quality_classes
    majority_chord_quality = majority_joint % quality_classes
    center = root_input.shape[1] // 2
    examples = len(target_root)

    def constant(value: int) -> np.ndarray:
        return np.full(examples, value, dtype=np.int64)

    return {
        "split": split,
        "examples": examples,
        "independent_majority": _weighted_prediction_metrics(
            constant(majority_root), constant(majority_quality),
            target_root, target_quality, weight,
        ),
        "majority_chord": _weighted_prediction_metrics(
            constant(majority_chord_root), constant(majority_chord_quality),
            target_root, target_quality, weight,
        ),
        "copy_previous": _weighted_prediction_metrics(
            root_input[:, center - 1], quality_input[:, center - 1],
            target_root, target_quality, weight,
        ),
        "copy_next": _weighted_prediction_metrics(
            root_input[:, center + 1], quality_input[:, center + 1],
            target_root, target_quality, weight,
        ),
        "majority_labels": {
            "root": majority_root,
            "quality": vocabulary["qualities"][majority_quality],
            "chord_root": majority_chord_root,
            "chord_quality": vocabulary["qualities"][majority_chord_quality],
        },
    }


def transpose_batch(
    root_input: torch.Tensor,
    melody: torch.Tensor,
    target_root: torch.Tensor,
    shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transpose pitched inputs independently while preserving special root tokens."""
    shifted_roots = torch.where(
        root_input < 12,
        (root_input + shifts[:, None]) % 12,
        root_input,
    )
    shifted_target = torch.where(target_root < 12, (target_root + shifts) % 12, target_root)
    indexes = (torch.arange(12, device=melody.device)[None, :] - shifts[:, None]) % 12
    indexes = indexes[:, None, :].expand(-1, melody.shape[1], -1)
    shifted_melody = torch.gather(melody, 2, indexes)
    return shifted_roots, shifted_melody, shifted_target


def _evaluate(
    model: MaskedChordModel, loader: DataLoader, device: torch.device,
) -> dict[str, float]:
    model.eval()
    weight_total = root_correct = quality_correct = chord_correct = 0.0
    root_loss_total = quality_loss_total = 0.0
    criterion = nn.CrossEntropyLoss(reduction="none")
    with torch.inference_mode():
        for root, quality, melody, scalars, target_root, target_quality, weight in loader:
            root, quality, melody, scalars = (
                root.to(device), quality.to(device), melody.to(device), scalars.to(device)
            )
            target_root = target_root.to(device)
            target_quality = target_quality.to(device)
            weight = weight.to(device)
            root_logits, quality_logits = model(root, quality, melody, scalars)
            root_prediction = root_logits.argmax(dim=1)
            quality_prediction = quality_logits.argmax(dim=1)
            weight_total += float(weight.sum())
            root_correct += float(((root_prediction == target_root) * weight).sum())
            quality_correct += float(((quality_prediction == target_quality) * weight).sum())
            chord_correct += float(
                ((root_prediction == target_root) & (quality_prediction == target_quality)).float()
                .mul(weight).sum()
            )
            root_loss_total += float((criterion(root_logits, target_root) * weight).sum())
            quality_loss_total += float((criterion(quality_logits, target_quality) * weight).sum())
    return {
        "loss": (root_loss_total + quality_loss_total) / max(weight_total, 1e-8),
        "root_accuracy": root_correct / max(weight_total, 1e-8),
        "quality_accuracy": quality_correct / max(weight_total, 1e-8),
        "chord_accuracy": chord_correct / max(weight_total, 1e-8),
    }


def train_masked_chord_model(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str = "auto",
    resume_from: str | Path | None = None,
) -> dict[str, object]:
    """Train a transpose-augmented masked chord model and retain the best validation state."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    vocabulary = json.loads((dataset_root / "vocabulary.json").read_text())
    train_data = _load_split(dataset_root / "train.npz")
    validation_path = dataset_root / "validation.npz"
    validation_data = _load_split(validation_path if validation_path.exists() else dataset_root / "train.npz")
    test_path = dataset_root / "test.npz"

    if device == "auto":
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    torch_device = torch.device(device)

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    model = MaskedChordModel(
        len(vocabulary["qualities"]), int(vocabulary["sequence_length"])
    ).to(torch_device)
    resumed_checkpoint: str | None = None
    previous_best_loss = float("inf")
    if resume_from is not None:
        resume_path = Path(resume_from).expanduser().resolve()
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("vocabulary") != vocabulary:
            raise ValueError("Resume checkpoint vocabulary does not match this dataset")
        model.load_state_dict(checkpoint["model"])
        previous_best_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        resumed_checkpoint = str(resume_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(reduction="none")
    history: list[dict[str, float | int]] = []
    best_loss = previous_best_loss
    best_state: dict[str, torch.Tensor] | None = (
        {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if resume_from is not None else None
    )

    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss = weight_total = 0.0
        for root, quality, melody, scalars, target_root, target_quality, weight in train_loader:
            root, quality, melody, scalars = (
                root.to(torch_device), quality.to(torch_device), melody.to(torch_device),
                scalars.to(torch_device),
            )
            target_root = target_root.to(torch_device)
            target_quality = target_quality.to(torch_device)
            weight = weight.to(torch_device)
            shifts = torch.randint(0, 12, (len(root),), generator=generator).to(torch_device)
            root, melody, target_root = transpose_batch(root, melody, target_root, shifts)
            optimizer.zero_grad()
            root_logits, quality_logits = model(root, quality, melody, scalars)
            losses = criterion(root_logits, target_root) + criterion(quality_logits, target_quality)
            loss = (losses * weight).sum() / weight.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weighted_loss += float((losses.detach() * weight).sum())
            weight_total += float(weight.sum())
        validation = _evaluate(model, validation_loader, torch_device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": weighted_loss / max(weight_total, 1e-8),
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_metrics = (
        _evaluate(
            model, DataLoader(_load_split(test_path), batch_size=batch_size), torch_device
        )
        if test_path.exists() else {}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_type": "masked_chord_transformer",
        "vocabulary": vocabulary,
        "best_validation_loss": best_loss,
        "test_metrics": test_metrics,
        "feature_version": 1,
        "epochs_completed": epochs,
        "resumed_from": resumed_checkpoint,
    }, output_path)
    metrics: dict[str, object] = {
        "epochs": epochs,
        "training_examples": len(train_data),
        "validation_examples": len(validation_data),
        "device": str(torch_device),
        "resumed_from": resumed_checkpoint,
        "best_validation_loss": best_loss,
        "test": test_metrics,
        "checkpoint": str(output_path),
        "history": history,
    }
    (output_path.parent / f"{output_path.stem}-metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-supervise a masked chord model")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/pretraining"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/masked-chord-model.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    metrics = train_masked_chord_model(
        args.dataset_root, args.output, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, device=args.device,
        resume_from=args.resume,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

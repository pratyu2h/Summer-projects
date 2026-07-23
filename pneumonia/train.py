"""
Training loop for the multi-label CXR classifier.

Metrics: per-class AUROC + mean AUROC (mAUC), which is the standard
metric for NIH ChestX-14 (used in the original CheXNet paper) -- accuracy
is meaningless here since most labels are 0 for most classes.

Usage (in Colab, after downloading the dataset -- see README.md):
    python train.py --data-root /content/nih_data --epochs 10 --batch-size 32
"""
import argparse
import copy

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import NIHChestXrayDataset, get_transforms
from losses import MultiLabelFocalLoss, compute_alpha_from_freq
from model import MultiLabelCXRModel, NIH_CLASSES


def compute_auroc(y_true: torch.Tensor, y_prob: torch.Tensor):
    """
    Per-class AUROC, skipping any class with no positive examples in this
    batch/split (roc_auc_score is undefined there -- common for rare
    findings like Hernia on small subsets). Returns (per_class_dict, mean_auc).
    """
    y_true_np = y_true.cpu().numpy()
    y_prob_np = y_prob.cpu().numpy()
    per_class = {}
    for i, cls in enumerate(NIH_CLASSES):
        col = y_true_np[:, i]
        if col.min() == col.max():
            continue  # only one class present, AUROC undefined
        per_class[cls] = roc_auc_score(col, y_prob_np[:, i])
    mean_auc = sum(per_class.values()) / len(per_class) if per_class else float("nan")
    return per_class, mean_auc


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n_batches = 0.0, 0
    all_probs, all_targets = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = loss_fn(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_targets.append(targets.detach().cpu())

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    per_class_auc, mean_auc = compute_auroc(all_targets, all_probs)
    return total_loss / max(n_batches, 1), mean_auc, per_class_auc


def train(data_root, epochs=10, batch_size=32, lr=1e-4, image_size=224,
          freeze_backbone=True, checkpoint_path="best_model.pth", device=None,
          num_workers=2, pretrained=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = NIHChestXrayDataset(data_root, "train_val_list.txt", transform=get_transforms(True, image_size))
    val_ds = NIHChestXrayDataset(data_root, "test_list.txt", transform=get_transforms(False, image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    alpha = compute_alpha_from_freq(train_ds.labels)
    loss_fn = MultiLabelFocalLoss(alpha=alpha, gamma=2.0)

    model = MultiLabelCXRModel(pretrained=pretrained, freeze_backbone=freeze_backbone).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_auc, best_state = -1.0, None
    history = []

    for epoch in range(1, epochs + 1):
        # progressive unfreezing: after 1/3 of training, start fine-tuning layer3+
        if freeze_backbone and epoch == max(1, epochs // 3):
            model.unfreeze_backbone("layer3")
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr / 10, weight_decay=1e-5)
            print(f"[epoch {epoch}] unfroze layer3+, lr -> {lr/10:.2e}")

        train_loss, train_auc, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_auc, val_per_class = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        scheduler.step(val_auc)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc,
                         "val_loss": val_loss, "val_auc": val_auc})
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f} train_mAUC={train_auc:.4f}  "
              f"val_loss={val_loss:.4f} val_mAUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            print(f"  -> new best (val_mAUC={best_auc:.4f}), saved to {checkpoint_path}")

    model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--checkpoint", default="best_model.pth")
    args = parser.parse_args()

    train(args.data_root, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, image_size=args.image_size, checkpoint_path=args.checkpoint)

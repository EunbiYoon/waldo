import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--query",
        type=str,
        default="Waldo wearing a red and white striped shirt, glasses, blue jeans",
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lambda_attr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=512)

    parser.add_argument("--use_attribute_binding", action="store_true")
    parser.add_argument("--debug_samples", type=int, default=0)

    return parser.parse_args()


class WaldoDetectionDataset(Dataset):
    def __init__(self, root, split, img_size=512, debug_samples=0):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / split
        self.ann_path = self.img_dir / "_annotations.coco.json"
        self.img_size = img_size

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        if not self.ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.ann_path}")

        with open(self.ann_path, "r") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.samples = []

        seen = set()
        for ann in coco["annotations"]:
            image_id = ann["image_id"]
            if image_id in seen:
                continue
            seen.add(image_id)

            img_info = self.images[image_id]
            filename = img_info["file_name"]
            width = img_info["width"]
            height = img_info["height"]

            x, y, w, h = ann["bbox"]
            box = [x, y, x + w, y + h]

            self.samples.append(
                {
                    "file_name": filename,
                    "width": width,
                    "height": height,
                    "box": box,
                }
            )

        if debug_samples > 0:
            self.samples = self.samples[:debug_samples]

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

        print(
            f"[Dataset] {split}: {len(self.samples)} samples loaded from {self.img_dir}",
            flush=True,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = self.img_dir / item["file_name"]

        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        sx = self.img_size / item["width"]
        sy = self.img_size / item["height"]

        x1, y1, x2, y2 = item["box"]
        box = torch.tensor(
            [[x1 * sx, y1 * sy, x2 * sx, y2 * sy]],
            dtype=torch.float32,
        )

        target = {
            "boxes": box,
            "labels": torch.tensor([1], dtype=torch.int64),
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


class AttributeBindingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        loss = torch.tensor(0.0, device=targets[0]["boxes"].device)

        for pred, target in zip(predictions, targets):
            if len(pred["boxes"]) == 0:
                loss = loss + 1.0
                continue

            pred_box = pred["boxes"][0:1]
            gt_box = target["boxes"][0:1]

            iou = box_iou(pred_box, gt_box).mean()
            loss = loss + (1.0 - iou)

        return loss / len(targets)


def evaluate(model, dataloader, device, split_name="val"):
    model.eval()

    ious = []
    top1_hits = 0
    total = 0
    zero_iou = 0

    pbar = tqdm(dataloader, desc=f"Evaluating {split_name}", leave=False)

    with torch.no_grad():
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                if len(output["boxes"]) == 0:
                    iou = 0.0
                else:
                    pred_box = output["boxes"][0:1]
                    gt_box = target["boxes"][0:1]
                    iou = box_iou(pred_box, gt_box).item()

                ious.append(iou)

                if iou >= 0.5:
                    top1_hits += 1
                if iou == 0:
                    zero_iou += 1

                total += 1

            if len(ious) > 0:
                pbar.set_postfix(mean_iou=f"{np.mean(ious):.4f}")

    return {
        "IoU": float(np.mean(ious)),
        "top1@0.5": float(top1_hits / total),
        "IoU=0 fraction": float(zero_iou / total),
        "p50": float(np.percentile(ious, 50)),
        "p90": float(np.percentile(ious, 90)),
    }


def main():
    args = parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("STARTING FASTER R-CNN + ATTRIBUTE BINDING TRAINING", flush=True)
    print("=" * 70, flush=True)
    print(f"dataset_root           : {dataset_root}", flush=True)
    print(f"output_dir             : {output_dir}", flush=True)
    print(f"query                  : {args.query}", flush=True)
    print(f"lr                     : {args.lr}", flush=True)
    print(f"epochs                 : {args.epochs}", flush=True)
    print(f"lambda_attr            : {args.lambda_attr}", flush=True)
    print(f"batch_size             : {args.batch_size}", flush=True)
    print(f"img_size               : {args.img_size}", flush=True)
    print(f"use_attribute_binding  : {args.use_attribute_binding}", flush=True)
    print(f"debug_samples          : {args.debug_samples}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device                 : {device}", flush=True)

    if torch.cuda.is_available():
        print(f"GPU name               : {torch.cuda.get_device_name(0)}", flush=True)

    print("=" * 70, flush=True)

    train_ds = WaldoDetectionDataset(
        dataset_root, "train", args.img_size, args.debug_samples
    )
    val_ds = WaldoDetectionDataset(
        dataset_root, "valid", args.img_size, args.debug_samples
    )
    test_ds = WaldoDetectionDataset(
        dataset_root, "test", args.img_size, args.debug_samples
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    print("[Model] Loading Faster R-CNN pretrained weights...", flush=True)

    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    model.roi_heads.box_predictor.cls_score = nn.Linear(
        model.roi_heads.box_predictor.cls_score.in_features,
        2,
    )
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(
        model.roi_heads.box_predictor.bbox_pred.in_features,
        2 * 4,
    )

    model.to(device)

    print("[Model] Model loaded and moved to device.", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    attr_loss_fn = AttributeBindingLoss()

    best_val_iou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print("=" * 70, flush=True)
        print(f"Epoch {epoch}/{args.epochs} started", flush=True)
        print("=" * 70, flush=True)

        model.train()
        total_loss = 0.0
        total_detection_loss = 0.0
        total_attr_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Training epoch {epoch}", leave=True)

        for step, (images, targets) in enumerate(pbar, start=1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            detection_loss = sum(loss for loss in loss_dict.values())

            loss = detection_loss
            attr_loss_value = 0.0

            if args.use_attribute_binding:
                model.eval()
                with torch.no_grad():
                    preds = model(images)
                model.train()

                attr_loss = attr_loss_fn(preds, targets)
                attr_loss_value = attr_loss.item()
                loss = detection_loss + args.lambda_attr * attr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_detection_loss += detection_loss.item()
            total_attr_loss += attr_loss_value

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "det": f"{detection_loss.item():.4f}",
                    "attr": f"{attr_loss_value:.4f}",
                }
            )

            if step % 10 == 0 or step == 1:
                print(
                    f"[Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)}] "
                    f"loss={loss.item():.4f} "
                    f"detection_loss={detection_loss.item():.4f} "
                    f"attr_loss={attr_loss_value:.4f}",
                    flush=True,
                )

        avg_train_loss = total_loss / len(train_loader)
        avg_detection_loss = total_detection_loss / len(train_loader)
        avg_attr_loss = total_attr_loss / len(train_loader)

        print(
            f"[Epoch {epoch}] Training finished. "
            f"avg_train_loss={avg_train_loss:.4f} "
            f"avg_detection_loss={avg_detection_loss:.4f} "
            f"avg_attr_loss={avg_attr_loss:.4f}",
            flush=True,
        )
        print(f"[Epoch {epoch}] Starting validation...", flush=True)

        val_metrics = evaluate(model, val_loader, device, split_name="valid")
        val_iou = val_metrics["IoU"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "detection_loss": float(avg_detection_loss),
                "attr_loss": float(avg_attr_loss),
                "val_IoU": float(val_metrics["IoU"]),
                "val_top1@0.5": float(val_metrics["top1@0.5"]),
                "val_IoU=0 fraction": float(val_metrics["IoU=0 fraction"]),
                "val_p50": float(val_metrics["p50"]),
                "val_p90": float(val_metrics["p90"]),
            }
        )

        history_path = output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"detection_loss={avg_detection_loss:.4f} | "
            f"attr_loss={avg_attr_loss:.4f} | "
            f"val_IoU={val_iou:.4f} | "
            f"top1@0.5={val_metrics['top1@0.5']:.4f} | "
            f"IoU=0 fraction={val_metrics['IoU=0 fraction']:.4f}",
            flush=True,
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"[Checkpoint] New best model saved to {save_path}", flush=True)

    print("=" * 70, flush=True)
    print("Training complete. Loading best model...", flush=True)
    print("=" * 70, flush=True)

    history_path = output_dir / "training_history.json"
    print(f"[Saved] training history saved to {history_path}", flush=True)

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))

    print("[Final Eval] Evaluating validation set...", flush=True)
    val_metrics = evaluate(model, val_loader, device, split_name="valid")

    print("[Final Eval] Evaluating test set...", flush=True)
    test_metrics = evaluate(model, test_loader, device, split_name="test")

    results = {
        "query": args.query,
        "lr": args.lr,
        "epochs": args.epochs,
        "lambda_attr": args.lambda_attr,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "use_attribute_binding": args.use_attribute_binding,
        "debug_samples": args.debug_samples,
        "best_val_iou": float(best_val_iou),
        "history_path": str(history_path),
        "val": val_metrics,
        "test": test_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 70, flush=True)
    print("FASTER R-CNN + ATTRIBUTE BINDING RESULTS")
    print("=" * 70, flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print(f"[Saved] metrics saved to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
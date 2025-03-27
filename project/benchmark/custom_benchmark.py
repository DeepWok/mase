import argparse
import time
import torch
from ultralytics import YOLO

def main(args):
    # Load the YOLO model with the specified task (detect, segment, classify)
    print(f"Loading model: {args.model_path} with task: {args.task}")
    model = YOLO(args.model_path, task=args.task)

    # Run validation on the provided dataset configuration
    print("Starting validation...")
    start_val = time.time()
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=True
    )
    total_val_time = time.time() - start_val
    print(f"Validation completed in {total_val_time:.2f} seconds.")

    # Extract and print mAP metrics if available
    try:
        mAP50 = metrics.box.map50
    except Exception:
        mAP50 = "N/A"
    try:
        mAP50_95 = metrics.box.map
    except Exception:
        mAP50_95 = "N/A"
    print(f"mAP50: {mAP50}")
    print(f"mAP50-95: {mAP50_95}")

    # Benchmark prediction speed: measure the time taken for a number of inference iterations
    print("Running inference benchmark to compute FPS...")
    dummy_input = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=args.device)
    n_iter = args.iterations
    start_pred = time.time()
    for _ in range(n_iter):
        _ = model(dummy_input)
    total_pred_time = time.time() - start_pred
    avg_time_per_batch = total_pred_time / n_iter
    fps = args.batch / avg_time_per_batch
    print(f"Average prediction time per batch: {avg_time_per_batch:.4f} seconds")
    print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom YOLO benchmark")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the YOLO model file (e.g., yolov8n.pt or yolov8n.yaml)")
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment", "classify"],
                        help="Task type for the model")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for validation/inference")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for validation/inference")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (e.g., cpu, cuda:0)")
    parser.add_argument("--data", type=str, default="coco.yaml", help="Dataset configuration file for validation")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations for FPS benchmark")
    args = parser.parse_args()
    main(args)

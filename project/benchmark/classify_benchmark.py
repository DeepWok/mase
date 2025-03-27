import argparse
import torch
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

def main(args):
    # Load the classifier model and its weights
    model = YOLO("yolov8n-cls.pt", task=args.task)
    state_dict = torch.load("classifier.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    
    # Validate the model on Caltech101 dataset
    metrics = model.val(data="caltech101", imgsz=640, batch=args.batch, device="1", verbose=True)
    
    # Run benchmark on the model using the engine format on Imagenet10
    benchmark(model="yolov8n-cls.yaml", data="imagenet10", format='engine', imgsz=640, device='2', verbose=True)
    
    # Predict on a sample image and print results
    results = model("https://ultralytics.com/images/bus.jpg")
    for result in results:
        print(f"Predicted Class: {result.names[result.probs.top1]}")
        print(f"Confidence Score: {result.probs.top1conf:.2f}")
    
    # Extract and print mAP metrics if available
    try:
        mAP50 = metrics.box.map50
    except Exception:
        mAP50 = "N/A"
    try:
        mAP50_95 = metrics.box.map
    except Exception:
        mAP50_95 = "N/A"
    
    print(f"Batch size: {args.batch}")
    if isinstance(mAP50, (float, int)):
        print(f"mAPval50 (B): {mAP50:.6f}")
    else:
        print(f"mAPval50 (B): {mAP50}")
    if isinstance(mAP50_95, (float, int)):
        print(f"mAPval50-95 (B): {mAP50_95:.6f}")
    else:
        print(f"mAPval50-95 (B): {mAP50_95}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO classifier model and print benchmark results"
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument(
        "--task",
        type=str,
        default="classify",
        choices=['detect', 'segment', 'classify', 'pose', 'obb'],
        help="Task type for the model"
    )
    args = parser.parse_args()
    main(args)

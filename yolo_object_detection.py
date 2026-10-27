import os
import sys
import time
import csv
from datetime import datetime
import cv2
import numpy as np

# Suppress Qt font warnings on OpenCV
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

try:
    import torch
    from ultralytics import YOLO
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"
    print(f"AI Backend: PyTorch ({torch.__version__}) | Device: {DEVICE.upper()}")
    if CUDA_AVAILABLE:
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    
    # Load YOLOv8 Nano Segmentation model
    model = YOLO("yolov8n-seg.pt")
    if CUDA_AVAILABLE:
        model.to(DEVICE).half()
except ImportError:
    print("ERROR: 'torch' or 'ultralytics' not installed in current environment.")
    sys.exit(1)

def main():
    print("\n--- AI Video & Screen Stream Tester (With Performance Logging) ---")
    print("Select input source:")
    print("  [1] Use default Webcam / Virtual Camera")
    print("  [2] Provide path to a local video file (.mp4, .avi)")
    print("  [3] Capture Desktop Screen (requires ximagesrc / ffmpeg)")
    
    choice = input("Enter choice [1/2/3] (default 1): ").strip()
    if choice == "2":
        source = input("Enter path to video file: ").strip()
    elif choice == "3":
        source = "ximagesrc use-damage=0 ! videoconvert ! appsink"
    else:
        source = 0  # Default webcam index

    print(f"\nOpening source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source {source}")
        sys.exit(1)

    window_name = "AI Video Stream Test (RTX 3050 Ti)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 1024, 768)

    is_fullscreen = False
    print("\nControls:")
    print(" -> Press 'f' to toggle Fullscreen")
    print(" -> Press 'q' or ESC to quit")

    # Initialize CSV Performance Log
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"ai_stream_metrics_{timestamp_str}.csv"
    
    csv_file = open(log_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "timestamp", "fps", "inference_ms", "vram_allocated_mb", "device"])

    frame_count = 0
    start_time = time.time()
    fps_smooth = 0.0

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("End of stream or failed to grab frame.")
                break

            # Run YOLOv8 instance segmentation with latency timer
            inf_start = time.time()
            results = model.predict(
                source=frame,
                device=DEVICE,
                conf=0.45,
                verbose=False,
                retina_masks=True
            )
            inf_time = (time.time() - inf_start) * 1000.0  # in milliseconds
            res = results[0]

            # Annotate frame with segmentation masks and bounding boxes
            annotated_frame = res.plot(conf=True, labels=True, boxes=True, masks=True)

            # Calculate instantaneous and smoothed FPS
            frame_count += 1
            elapsed_total = time.time() - start_time
            current_fps = frame_count / elapsed_total if elapsed_total > 0 else 0.0
            fps_smooth = (0.9 * fps_smooth) + (0.1 * current_fps) if frame_count > 1 else current_fps

            # Query live GPU memory usage if on CUDA
            vram_mb = 0.0
            if CUDA_AVAILABLE:
                vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)

            # Write performance metrics to CSV log
            csv_writer.writerow([
                frame_count,
                datetime.now().isoformat(),
                f"{fps_smooth:.2f}",
                f"{inf_time:.2f}",
                f"{vram_mb:.1f}",
                DEVICE.upper()
            ])

            # Add HUD info overlay to the video feed
            h, w, _ = annotated_frame.shape
            hud_text1 = f"Device: {DEVICE.upper()} | FPS: {fps_smooth:.1f}"
            hud_text2 = f"Inference: {inf_time:.1f}ms | VRAM: {vram_mb:.1f}MB"
            
            cv2.putText(annotated_frame, hud_text1, (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, hud_text2, (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27] or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                )

    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print(f"\nStream closed cleanly. Performance metrics saved to: {log_filename}")

if __name__ == "__main__":
    main()
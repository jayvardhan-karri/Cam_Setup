import os
import sys
import pyrealsense2 as rs
import numpy as np
import cv2

try:
    import torch
    from ultralytics import YOLO
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"
    print(f"AI Backend: PyTorch ({torch.__version__}) | Device: {DEVICE.upper()}")
    if CUDA_AVAILABLE:
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    
    # Using YOLOv8 Small Segmentation model (yolov8s-seg.pt) for high-accuracy robotics perception
    model = YOLO("yolov8s-seg.pt")
    if CUDA_AVAILABLE:
        model.to(DEVICE).half()
        
except ImportError:
    print("ERROR: 'torch' or 'ultralytics' not installed. Please check your venv.")
    sys.exit(1)

# Suppress Qt font warnings on OpenCV
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("ERROR: No RealSense device found.")
    sys.exit(1)

dev = devices[0]
usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
print(f"Connected: {dev.get_info(rs.camera_info.name)} | USB Mode: {usb_type}")

pipeline = rs.pipeline()
config = rs.config()

if "2." in usb_type:
    print("\nConfiguring safe USB 2.1 profile (424x240 @ 6 FPS)...")
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)
else:
    print("\nConfiguring full USB 3.2 profile (640x480 @ 30 FPS)...")
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

align = rs.align(rs.stream.color)

try:
    profile = pipeline.start(config)
    print("SUCCESS: Pipeline started!")
except Exception as e:
    print(f"Fallback: Starting with default profile ({e})")
    config.disable_all_streams()
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)
    profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

spatial_filter = rs.spatial_filter()
spatial_filter.set_option(rs.option.filter_magnitude, 2)
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
spatial_filter.set_option(rs.option.holes_fill, 3)

hole_filling_filter = rs.hole_filling_filter()

# COCO Class IDs of interest (39 = 'bottle', 41 = 'cup')
TARGET_CLASSES = [39, 41]

window_name = "RealSense D455 - 3D Dimensions & Segmentation"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(window_name, 1280, 480)

is_fullscreen = False
print("\nControls:")
print(" -> Press 'f' to toggle Fullscreen")
print(" -> Press 'q' or ESC to quit.")

try:
    while True:
        frames = pipeline.poll_for_frames()
        if not frames:
            key = cv2.waitKey(15) & 0xFF
            if key in [ord('q'), 27] or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        filtered_depth_frame = spatial_filter.process(depth_frame)
        filtered_depth_frame = hole_filling_filter.process(filtered_depth_frame)

        intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        fx, fy = intrinsics.fx, intrinsics.fy

        depth_image = np.asanyarray(filtered_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if color_frame.profile.format() == rs.format.rgb8:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        if depth_colormap.shape[:2] != color_image.shape[:2]:
            depth_colormap = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))

        h, w, _ = color_image.shape
        depth_image_aligned = cv2.resize(depth_image, (w, h), interpolation=cv2.INTER_NEAREST) if depth_image.shape[:2] != (h, w) else depth_image

        if model is not None:
            results = model.predict(
                source=color_image,
                device=DEVICE,
                classes=TARGET_CLASSES,
                conf=0.45,
                verbose=False,
                retina_masks=True
            )
            res = results[0]

            if res.boxes is not None and len(res.boxes) > 0:
                for i, box in enumerate(res.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    label_name = model.names[cls_id]

                    u = int((x1 + x2) / 2)
                    v = int((y1 + y2) / 2)

                    if res.masks is not None and i < len(res.masks.data):
                        mask = res.masks.data[i].cpu().numpy().astype(bool)
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        obj_depths = depth_image_aligned[mask] * depth_scale
                        color_image[mask] = cv2.addWeighted(color_image[mask], 0.6, np.full_like(color_image[mask], (0, 255, 0)), 0.4, 0)
                    else:
                        pad_x, pad_y = max(1, (x2 - x1) // 4), max(1, (y2 - y1) // 4)
                        roi_depth = depth_image_aligned[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x]
                        obj_depths = roi_depth * depth_scale

                    valid_depths = obj_depths[(obj_depths > 0.15) & (obj_depths < 4.0)]

                    if len(valid_depths) > 0:
                        z_min = float(np.min(valid_depths))
                        z_max = float(np.max(valid_depths))
                        z_dist = float(np.percentile(valid_depths, 35))

                        # Compute physical width and height using camera intrinsics and median depth
                        box_width_px = x2 - x1
                        box_height_px = y2 - y1
                        real_width = (box_width_px * z_dist) / fx
                        real_height = (box_height_px * z_dist) / fy
                        real_length = z_max - z_min  # Thickness along camera optical axis

                        X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], z_dist)

                        # Draw bounding boxes and overlays
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)

                        info_top = f"{label_name.upper()} ({conf:.0%}): {Z:.2f}m"
                        info_dim = f"Dim [WxHxL]: {real_width*100:.1f}x{real_height*100:.1f}x{real_length*100:.1f}cm"

                        cv2.putText(color_image, info_top, (x1, max(20, y1 - 25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(color_image, info_dim, (x1, max(42, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

        images = np.hstack((color_image, depth_colormap))
        cv2.imshow(window_name, images)

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

except KeyboardInterrupt:
    print("\nStopped via Ctrl+C.")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
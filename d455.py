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
    
    # Upgraded to YOLOv8 Small Segmentation model (yolov8s-seg.pt) for superior accuracy on robotics objects
    model = YOLO("yolov8s-seg.pt")
except ImportError:
    print("WARNING: 'torch' or 'ultralytics' not installed yet. Running without AI detection.")
    model = None
    DEVICE = "cpu"

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

# Print exact valid profiles reported by the camera hardware
print("\n--- Available Stream Profiles on Current USB Connection ---")
for sensor in dev.query_sensors():
    print(f"Sensor: {sensor.get_info(rs.camera_info.name)}")
    for p in sensor.get_stream_profiles():
        if not p.is_video_stream_profile():
            continue
        vp = p.as_video_stream_profile()
        if vp.format() in [rs.format.z16, rs.format.bgr8, rs.format.rgb8]:
            print(f"  -> {vp.stream_type()}: {vp.width()}x{vp.height()} @ {vp.fps()} FPS ({vp.format()})")
            break  # Print first valid match per sensor to keep output clean

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
    print(f"Fallback: Starting with default auto-negotiated camera profile ({e})")
    config.disable_all_streams()
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)
    profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# RealSense hardware post-processing filters to fix specular reflections on metal bottles
spatial_filter = rs.spatial_filter()
spatial_filter.set_option(rs.option.filter_magnitude, 2)
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
spatial_filter.set_option(rs.option.holes_fill, 3)  # Fill zero-depth specular holes

hole_filling_filter = rs.hole_filling_filter()

# COCO Class IDs of interest (39 = 'bottle', 41 = 'cup')
TARGET_CLASSES = [39, 41]

window_name = "RealSense D455 Feed (YOLOv8s-seg)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(window_name, 1280, 480)

is_fullscreen = False
flip_mode = None

print("\nControls:")
print(" -> Press 'f' to toggle Fullscreen")
print(" -> Press 'r' to flip image 180° or 'm' to mirror")
print(" -> Press 'q', ESC, or click 'X' to quit.")

try:
    while True:
        frames = pipeline.poll_for_frames()
        if not frames:
            key = cv2.waitKey(15) & 0xFF
            if key in [ord('q'), 27] or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\nClosing stream cleanly...")
                break
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                )
            elif key == ord('r'):
                flip_mode = -1 if flip_mode != -1 else None
            elif key == ord('m'):
                flip_mode = 1 if flip_mode != 1 else None
            continue

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        filtered_depth_frame = spatial_filter.process(depth_frame)
        filtered_depth_frame = hole_filling_filter.process(filtered_depth_frame)

        intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()

        depth_image = np.asanyarray(filtered_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if color_frame.profile.format() == rs.format.rgb8:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        if flip_mode is not None:
            color_image = cv2.flip(color_image, flip_mode)
            depth_image = cv2.flip(depth_image, flip_mode)

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        if depth_colormap.shape[:2] != color_image.shape[:2]:
            depth_colormap = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))

        h, w, _ = color_image.shape
        if depth_image.shape[:2] != (h, w):
            depth_image_aligned = cv2.resize(depth_image, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            depth_image_aligned = depth_image

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
                        z_dist = float(np.percentile(valid_depths, 35))
                        X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], z_dist)

                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)

                        info_top = f"{label_name.upper()} ({conf:.0%}): {Z:.2f}m"
                        info_xyz = f"XYZ: [{X:+.2f}, {Y:+.2f}, {Z:.2f}]m"

                        cv2.putText(color_image, info_top, (x1, max(20, y1 - 22)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                        cv2.putText(color_image, info_xyz, (x1, max(40, y1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        center_dist = depth_image_aligned[h // 2, w // 2] * depth_scale
        cv2.circle(color_image, (w // 2, h // 2), 3, (255, 200, 0), -1)
        cv2.putText(color_image, f"Center: {center_dist:.2f}m", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

        images = np.hstack((color_image, depth_colormap))
        cv2.imshow(window_name, images)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27] or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("\nClosing stream cleanly...")
            break
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            )

except KeyboardInterrupt:
    print("\nStopped via Ctrl+C. Closing stream cleanly...")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
import os
import sys
import pyrealsense2 as rs
import numpy as np
import cv2

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

window_name = "RealSense D455 Feed"
# WINDOW_NORMAL + WINDOW_KEEPRATIO enables the OS maximize button and manual resizing
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(window_name, 1280, 480)

is_fullscreen = False
# Set flip_mode:
#   None = normal orientation
#  -1    = flip upside down (180 deg, if camera is mounted inverted)
#   1    = mirror horizontally (selfie/webcam view)
#   0    = flip vertically only
flip_mode = None

print("\nControls:")
print(" -> Press 'f' to toggle Fullscreen")
print(" -> Press 'r' to flip image 180° (if camera is upside-down) or 'm' to mirror")
print(" -> Press 'q', ESC, Ctrl+C in terminal, or click the window 'X' button to quit.")

try:
    while True:
        # Use poll_for_frames + short sleep instead of blocking wait_for_frames
        # so Ctrl+C and window 'X' button respond instantaneously even at 5 FPS
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
                print(f"180° Flip: {'ON' if flip_mode == -1 else 'OFF'}")
            elif key == ord('m'):
                flip_mode = 1 if flip_mode != 1 else None
                print(f"Horizontal Mirror: {'ON' if flip_mode == 1 else 'OFF'}")
            continue

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Fix inverted color channels (Auto-negotiation defaults to RGB8 instead of BGR8)
        if color_frame.profile.format() == rs.format.rgb8:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Apply optional orientation flip (before drawing text so text stays readable!)
        if flip_mode is not None:
            color_image = cv2.flip(color_image, flip_mode)
            depth_image = cv2.flip(depth_image, flip_mode)

        # Resize depth colormap to match color_image dimensions if auto-negotiated sizes differ
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        if depth_colormap.shape[:2] != color_image.shape[:2]:
            depth_colormap = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))

        h, w, _ = color_image.shape
        center_dist = depth_frame.get_distance(w // 2, h // 2)

        cv2.circle(color_image, (w // 2, h // 2), 4, (0, 255, 0), -1)
        cv2.putText(color_image, f"{center_dist:.2f}m", (w // 2 + 10, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
        elif key == ord('r'):
            flip_mode = -1 if flip_mode != -1 else None
            print(f"180° Flip: {'ON' if flip_mode == -1 else 'OFF'}")
        elif key == ord('m'):
            flip_mode = 1 if flip_mode != 1 else None
            print(f"Horizontal Mirror: {'ON' if flip_mode == 1 else 'OFF'}")

except KeyboardInterrupt:
    print("\nStopped via Ctrl+C. Closing stream cleanly...")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
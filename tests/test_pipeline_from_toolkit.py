# test_camera_pipeline.py                                                                                                                                                                  
import time                                                                                                                                                                              
from pathlib import Path                                                                                                                                                                   
                                                                                                                                                                                            
from stereo_depth import CameraStreamer, RollingBuffer, StereoPipeline
from stereo_depth.adapters.camera.uvc_source import UvcSource

# --- Build pipeline from your best calib ---
pipeline = StereoPipeline.from_yaml(
    "/home/kevin/projects/stereo-depth-toolkit/outputs/calib/calib.yaml"
)

# --- Open camera ---
source = UvcSource(device_index=0, width=2560, height=720, fps=30)
buffer = RollingBuffer(maxlen=15)

with CameraStreamer(source, buffer, pipeline) as streamer:
    # streamer.preview()
    print("Waiting for buffer to fill...")
    timeout = time.monotonic() + 10.0
    while not streamer.is_ready:
        if time.monotonic() > timeout:
            raise RuntimeError("Buffer did not fill in 10s — check camera connection")
        time.sleep(0.05)

    print(f"Ready. FPS: {streamer.fps:.1f}")

    result = streamer.snapshot()

    print(f"frame_index   : {result.frame_index}")
    print(f"stable_depth  : shape={result.stable_depth.shape}  dtype={result.stable_depth.dtype}")
    print(f"rgb_snapshot  : shape={result.rgb_snapshot.shape}  dtype={result.rgb_snapshot.dtype}")
    print(f"process_time  : {result.process_time_s*1000:.1f} ms")

    import numpy as np
    valid = result.stable_depth[~np.isnan(result.stable_depth)]
    if valid.size:
        print(f"depth range   : {valid.min():.3f} m – {valid.max():.3f} m")
        print(f"depth median  : {np.median(valid):.3f} m")
    else:
        print("depth range   : all NaN (check calib or lighting)")

print("Done.")
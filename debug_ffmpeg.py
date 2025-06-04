import subprocess
import time
import os

# Input video paths
videos = [
    "composite_videos/BKO_E1_D1_02_Maraka/beat_1/temp_resized/video_mix_concat.mp4",
    "composite_videos/BKO_E1_D1_02_Maraka/beat_1/temp_resized/front_view_concat.mp4",
    "composite_videos/BKO_E1_D1_02_Maraka/beat_1/temp_resized/video_mix_concat.mp4",
    "composite_videos/BKO_E1_D1_02_Maraka/beat_1/temp_resized/front_view_concat.mp4",
]

# Verify input files exist
print("Checking input files:")
for video in videos:
    if os.path.exists(video):
        print(f"✓ {video} exists")
    else:
        print(f"✗ {video} does not exist")

# Layout positions (2x2 grid, 960x540 each)
layout = "0_0|960_0|0_540|960_540"

# Output path
output = "final_output.mp4"

# Build ffmpeg command
cmd = [
    "ffmpeg", "-y",
    "-hwaccel", "cuda",
    *[arg for v in videos for arg in ("-i", v)],
    "-filter_complex", f"xstack=inputs=4:layout={layout}:fill=black[out]",
    "-map", "[out]",
    "-c:v", "h264_nvenc",
    "-preset", "p1",
    "-rc", "constqp",
    "-qp", "23",
    # "-r", "24",
    output
]

# Print the command for verification
print("\nExecuting command:")
print(" ".join(cmd))
print("\nStarting video processing...")

# Run command with timing
start_time = time.time()
try:
    # Live output: do not capture output, let ffmpeg print to terminal
    subprocess.run(cmd, check=True)
    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
except subprocess.CalledProcessError as e:
    end_time = time.time()
    print(f"\nError occurred after {end_time - start_time:.2f} seconds") 
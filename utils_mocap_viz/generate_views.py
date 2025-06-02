import os
from datetime import datetime
import subprocess
from .mocap_visualizer_F_view import MocapVisualizerFront
from .mocap_visualizer_RS_view import MocapVisualizerRightSide
from .mocap_visualizer_LS_view import MocapVisualizerLeftSide
from .mocap_visualizer_T_view import MocapVisualizerTop
from .kinematic_visualizer import visualize_joint_position

def get_output_dir(bvh_file, start_time, end_time):
    """Create and return output directory path based on filename and time range"""
    filename = os.path.splitext(os.path.basename(bvh_file))[0]
    dir_name = f"{filename}_{start_time:.1f}_{end_time:.1f}"
    output_dir = os.path.join("output", dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def check_existing_videos(output_dir, views_to_generate):
    """Check if videos for specified views already exist"""
    existing_videos = {}
    for view in views_to_generate:
        video_path = os.path.join(output_dir, f"{view}_view.mp4")
        if os.path.exists(video_path):
            existing_videos[view] = video_path
    return existing_videos

def generate_individual_videos(bvh_file, start_time, end_time, output_fps, video_size, views_to_generate=None):
    """Generate videos for specified views
    
    Args:
        bvh_file: Path to BVH file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_fps: Output video FPS
        video_size: Output video size (width, height)
        views_to_generate: List of views to generate ('front', 'right', 'left', 'top')
                          If None, generates all views
    """
    # Create output directory
    output_dir = get_output_dir(bvh_file, start_time, end_time)
    
    # Default to all views if none specified
    if views_to_generate is None:
        views_to_generate = ['front', 'right']  # 'left', 'top'
    
    # Check for existing videos
    existing_videos = check_existing_videos(output_dir, views_to_generate)
    
    # Map view names to their visualizer classes
    view_classes = {
        'front': (MocapVisualizerFront, 'front_view.mp4'),
        'right': (MocapVisualizerRightSide, 'right_view.mp4'),
        'left': (MocapVisualizerLeftSide, 'left_view.mp4'),
        'top': (MocapVisualizerTop, 'top_view.mp4')
    }
    
    # Generate only missing views
    for view in views_to_generate:
        if view not in existing_videos:
            if view in view_classes:
                visualizer_class, output_filename = view_classes[view]
                print(f"\nGenerating {view} view...")
                visualizer = visualizer_class(bvh_file, debug=True)
                visualizer.generate_video(
                    output_file=os.path.join(output_dir, output_filename),
                    start_time=start_time,
                    end_time=end_time,
                    output_fps=output_fps,
                    video_size=video_size,
                    show_info=(view == 'front')  # Show frame info only in front view
                )
    
    return output_dir

def trim_video(input_file, output_file, start_time, end_time, target_fps=24):
    """Trim a video using FFmpeg and convert to target frame rate, also extract audio separately"""
    # Trim video
    video_command = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-i', input_file,
        '-c:v', 'libx264',
        '-r', str(target_fps),  # Set target frame rate
        '-filter:v', f'fps={target_fps},setpts=PTS-STARTPTS',  # Ensure consistent frame rate and reset timestamps
        '-pix_fmt', 'yuv420p',
        output_file
    ]
    # subprocess.run(video_command, check= True)
    print(f"\nExecuting video processing command:")
    print(" ".join(video_command))
    try:
        result = subprocess.run(video_command, check=True, capture_output=True, text=True)
        print("Video processing completed successfully")
        print(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video processing: {e}")
        print(f"Error output: {e.stderr}")
        raise
    
    # Extract audio
    audio_output = os.path.splitext(output_file)[0] + '_audio.mp3'
    audio_command = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-to', str(end_time),
        '-i', input_file,
        '-q:a', '0',  # High quality audio
        '-map', 'a',
        audio_output
    ]
    # subprocess.run(audio_command, check= True)
    print(f"\nExecuting audio processing command:")
    print(" ".join(audio_command))
    try:
        result = subprocess.run(audio_command, check=True, capture_output=True, text=True)
        print("Audio processing completed successfully")
        print(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio processing: {e}")
        print(f"Error output: {e.stderr}")
        raise

def prepare_videos(
    filename,
    start_time,
    end_time,
    video_path=None,
    video_size=(1280, 720),
    fps=24
):
    """Prepare all required videos for combining
    
    Args:
        filename: Base filename for output
        start_time: Start time in seconds
        end_time: End time in seconds
        video_path: Path to video file for replacement views
        video_size: Overall output video size (width, height)
        fps: Target frame rate for output video
        
    Returns:
        tuple: (output_dir, view_videos)
    """
    # Create output directory
    output_dir = get_output_dir(filename, start_time, end_time)
    
    # Generate all required views
    view_videos = {}
    
    # Generate motion capture views
    for view in ['front', 'right', 'left', 'top']:
        view_path = os.path.join(output_dir, f"{view}_view.mp4")
        if not os.path.exists(view_path):
            generate_individual_videos(
                bvh_file=filename + ".bvh",
                start_time=start_time,
                end_time=end_time,
                output_fps=fps,
                video_size=video_size,
                views_to_generate=[view]
            )
        view_videos[view] = view_path
    
    # video trimming
    if video_path:
        trimmed_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_trimmed.mp4")
        if not os.path.exists(trimmed_path):
            trim_video(video_path, trimmed_path, start_time, end_time, fps)
        view_videos[os.path.basename(video_path)] = trimmed_path
    
    return output_dir, view_videos

def main():
    start_t = datetime.now()
    
    # Configuration
    filename = "BKO_E1_D5_01_Maraka"
    bvh_file = filename + "_T"
    start_time = 120.0
    end_time = 125.0
    output_fps = 24
    video_size = (1280, 720)
    
    joint_name = "LeftAnkle"  
    axis = 'y'
    
    # Generate phase analysis videos
    print("\nGenerating phase analysis videos...")
    from animated_phase_analysis import generate_all_animations
    from animated_merged_phase_analysis import animate_merged_phase_analysis
    
    # Generate individual phase analysis videos
    cycles_csv_path = f"virtual_cycles/{filename}_C.csv"
    onsets_csv_path = f"drum_onsets/{filename}.csv"
    
    # Get output directory for phase analysis videos
    output_dir = get_output_dir(bvh_file, start_time, end_time)
    
    # Generate all individual phase analysis videos
    generate_all_animations(
        filename, start_time, end_time,
        cycles_csv_path, onsets_csv_path,
        save_dir=output_dir,
        figsize=(10, 3), dpi=100
    )
    
    # Generate merged phase analysis video
    # merged_save_path = f"{filename}_{start_time:.1f}_{end_time:.1f}_merged_phase_analysis.mp4"
    merged_save_path = "combined.mp4"
    animate_merged_phase_analysis(
        filename, start_time, end_time,
        cycles_csv_path, onsets_csv_path,
        figsize=(10, 4), dpi=100,
        save_path=merged_save_path,
        save_dir=output_dir
    )
    
    # 1. Prepare all videos first
    output_dir, view_videos = prepare_videos(
        filename=bvh_file,
        start_time=start_time,
        end_time=end_time,
        video_path="BKO_E1_D5_01_Maraka_pre_R_Mix.mp4",
        video_size=video_size,
        fps=output_fps
    )
    
    # Generate joint position visualization
    joint_video = os.path.join(output_dir, f"{joint_name}_{axis}_position.mp4")
    if not os.path.exists(joint_video):
        visualize_joint_position(
            bvh_file=bvh_file + ".bvh",
            joint_name=joint_name,
            axis=axis,
            start_time=start_time,
            end_time=end_time,
            output_fps=output_fps,
            output_dir=output_dir,
            video_size=(video_size[0], video_size[1] // 2)  # Half height for joint visualization
        )
    view_videos['joint_position'] = joint_video
    
    end_t = datetime.now()
    print(f"Time taken: {end_t - start_t}")

if __name__ == "__main__":
    main() 
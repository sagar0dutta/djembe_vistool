import os
import subprocess

def get_layout_config(layout_name='L2'):
    """Get layout configuration by name"""
    layouts = {
        'L1': [
            {'view': 'front', 'x': 0, 'y': 0, 'width': 640, 'height': 360},
            {'view': 'BKO_E1_D5_01_Maraka_pre_R_Mix.mp4', 'x': 640, 'y': 0, 'width': 640, 'height': 360},
            {'view': 'joint_position', 'x': 0, 'y': 360, 'width': 1280, 'height': 180}
        ],
        'L2': [
            {'view': 'front', 'x': 0, 'y': 0, 'width': 640, 'height': 360},
            {'view': 'right', 'x': 640, 'y': 0, 'width': 640, 'height': 360},
            {'view': 'joint_position', 'x': 0, 'y': 360, 'width': 1280, 'height': 360}
        ]
    }
    return layouts.get(layout_name, layouts['L2'])  # Default to L2 if layout not found

def get_video_paths(output_dir, filename, joint_name="LeftAnkle", axis='y'):
    """Get dictionary mapping view names to their expected video paths"""
    return {
        'front': os.path.join(output_dir, "front_view.mp4"),
        'right': os.path.join(output_dir, "right_view.mp4"),
        # 'left': os.path.join(output_dir, "left_view.mp4"),
        # 'top': os.path.join(output_dir, "top_view.mp4"),
        
        'dundun': os.path.join(output_dir, "Dun.mp4"),
        'J1': os.path.join(output_dir, "J1.mp4"),
        'J2': os.path.join(output_dir, "J2.mp4"),
        'combined': os.path.join(output_dir, "combined.mp4"),
        
        'joint_pos': os.path.join(output_dir, f"{joint_name}_{axis}_position.mp4"),
        'video_mix': os.path.join(output_dir, f"{filename}_pre_R_Mix_trimmed.mp4"),
        'audio': os.path.join(output_dir, f"{filename}_pre_R_Mix_trimmed_audio.mp3")
        
    }

def combine_views(
    filename,
    start_time,
    end_time,
    output_dir,
    view_videos,
    video_path=None,
    layout_config=None,
    video_size=(1280, 720),
    fps=24
):
    """Combine multiple views into a single video with custom layout
    
    Args:
        filename: Base filename for output
        start_time: Start time in seconds
        end_time: End time in seconds
        view_videos: Dictionary mapping view names to their video paths
        video_path: Path to video file for replacement views
        layout_config: List of dictionaries defining the layout structure, where each dict contains:
                      - view: Name of the view/video
                      - x: X coordinate position
                      - y: Y coordinate position
                      - width: Width of the video
                      - height: Height of the video
        video_size: Overall output video size (width, height)
        fps: Target frame rate for output video
    """
    # Get output directory
    # output_dir = os.path.join("output", f"{filename}_{start_time:.1f}_{end_time:.1f}")
    # os.makedirs(output_dir, exist_ok=True)
    
    # Build ffmpeg filter complex for layout
    filter_complex = []
    input_files = []
    input_count = 0
    
    # Create a black background layer
    filter_complex.append(
        f'color=c=black:s={video_size[0]}x{video_size[1]}:d={end_time-start_time}[base];'
    )
    
    # Process each video in the layout
    last_output = 'base'
    for item in layout_config:
        view_name = item['view']
        x = item['x']
        y = item['y']
        width = item['width']
        height = item['height']
        
        # Get video path from view_videos dictionary
        video_path = view_videos[view_name]
        input_files.extend(['-i', video_path])
        
        # Scale video to specified dimensions and ensure frame rate and timing
        filter_complex.append(
            f'[{input_count}:v]'
            f'fps={fps},'  # First ensure consistent frame rate
            f'setpts=PTS-STARTPTS,'  # Reset timestamps
            f'scale={width}:{height}'  # Then scale to desired size
            f'[v{input_count}];'
        )
        
        # Add view name only if it's not the joint position visualization
        if view_name != 'joint_position':
            filter_complex.append(
                f'[v{input_count}]drawtext=text=\'{view_name.title()} View\':'
                f'fontcolor=white:fontsize=24:'
                f'x=(w-text_w)/2:y=20[v{input_count}t];'
            )
        else:
            # For joint position, just pass through without adding text
            filter_complex.append(f'[v{input_count}]copy[v{input_count}t];')
        
        # Overlay this video on top of the previous result
        current_output = f'overlay{input_count}'
        filter_complex.append(
            f'[{last_output}][v{input_count}t]overlay=x={x}:y={y}[{current_output}];'
        )
        last_output = current_output
        input_count += 1
    
    # Output file path
    output_file = os.path.join(output_dir, f"combined_{os.path.splitext(os.path.basename(filename))[0]}.mp4")
    
    # Add audio input if available
    if 'audio' in view_videos and os.path.exists(view_videos['audio']):
        input_files.extend(['-i', view_videos['audio']])
        print(f"Adding audio from: {view_videos['audio']}")
    else:
        print("No audio file found in view_videos")
    
    # Use ffmpeg to create the final video
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        *input_files,
        '-filter_complex', ''.join(filter_complex)[:-1],  # Remove last semicolon
        '-map', f'[{last_output}]',
        *(['-map', f'{input_count}:a'] if 'audio' in view_videos and os.path.exists(view_videos['audio']) else []),  # Add audio mapping if available
        '-c:v', 'libx264',
        '-c:a', 'aac',  # Add audio codec
        '-preset', 'slow',  # Use slower preset for better quality
        '-crf', '18',  # Use higher quality (lower value = higher quality, 18-28 is good range)
        '-pix_fmt', 'yuv420p',
        '-r', str(fps),
        '-t', str(end_time - start_time),
        output_file
    ]
    
    print("\nFFmpeg command:")
    print(' '.join(ffmpeg_cmd))
    
    subprocess.run(ffmpeg_cmd)
    print(f"Combined video saved to {output_file}")
    return output_file

def main():
    # Configuration
    filename = "BKO_E1_D5_01_Maraka"
    bvh_file = filename + "_T"
    start_time = 120.0
    end_time = 150.0
    video_size = (1280, 720)
    fps = 24
    joint_name = "LeftAnkle"
    axis = 'y'
    
    # Get output directory
    output_dir = os.path.join("output", f"{bvh_file}_{start_time:.1f}_{end_time:.1f}")
    
    # Choose layout
    layout = {
        'L1': [
            {'view': 'front', 'x': 0, 'y': 0, 'width': 320, 'height': 360},
            {'view': 'right', 'x': 320, 'y': 0, 'width': 320, 'height': 360},
            {'view': 'video_mix', 'x': 640, 'y': 0, 'width': 640, 'height': 360},
            {'view': 'combined', 'x': 0, 'y': 360, 'width': 1280, 'height': 360}
        ],
        'L2': [
            {'view': 'front', 'x': 0, 'y': 0, 'width': 640, 'height': 360},
            {'view': 'right', 'x': 640, 'y': 0, 'width': 640, 'height': 360},
            {'view': 'joint_position', 'x': 0, 'y': 360, 'width': 1280, 'height': 360}
        ]
    }
    # layout_name = 'L1'  # Change this to use a different layout
    layout_config = layout['L1']
    
    # Get video paths
    view_videos = get_video_paths(output_dir, filename, joint_name, axis)
    
    # Check if we have all required videos for the layout
    required_views = {item['view'] for item in layout_config}
    missing_views = required_views - set(view_videos.keys())
    
    if missing_views:
        print(f"Warning: Missing videos for views: {missing_views}")
        print("Please run combine_views.py first to generate the videos")
        return
    
    # Combine the videos
    combine_views(
        filename=bvh_file,
        start_time=start_time,
        end_time=end_time,
        output_dir=output_dir,
        view_videos=view_videos,
        layout_config=layout_config,
        video_size=video_size,
        fps=fps
    )

if __name__ == "__main__":
    main() 
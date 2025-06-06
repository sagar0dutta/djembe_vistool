import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D


def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'


def extract_kinematic_cycle_plots(
    file_name: str,
    windows: list,  # List of (win_start, win_end, t_poi) tuples
    joint_name: str,
    axis: str = 'y',
    base_path_logs: str = "data/logs_v2_may",
    frame_rate: float = 240,  # Trajectory data frame rate
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    nn: int = 3,
    output_dir2: str = None,
    figsize: tuple = (10, 3),
    dpi: int = 200,
    legend_flag: bool = True,
):
    """
    Create trajectory animations for windows around points of interest (beats or subdivisions).
    Each plot shows [-cycle, 0-cycle, +cycle] around the POI.
    """
    # Create save directory if not provided
    # if output_dir2 is None:
    #     output_dir2 = os.path.join("cycle_plots", file_name, window_key, joint_name)
    #     os.makedirs(output_dir2, exist_ok=True)
    
    bvh_to_mvnx = {
    'x': 'y',  # BVH side → MVNX side
    'y': 'z',  # BVH vertical → MVNX vertical
    'z': 'x',  # BVH forward → MVNX forward
    }
    
    
    # Load joint position data
    dir_csv = "extracted_mocap_csv"
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    worldpos_file = os.path.join(dir_csv, f"{base_name}_T_worldpos.csv")
    
    try:
        world_positions = pd.read_csv(worldpos_file)
        print(f"Successfully loaded CSV with {len(world_positions)} rows")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise
    
    # Get time column and position data
    time_column = world_positions.columns[0]  # First column is time
    times = world_positions[time_column].values
    positions = world_positions[f"{joint_name}.{axis.upper()}"].values
    
    print(f"\nProcessing {len(windows)} windows")
    # print(f"Total frames in trajectory data: {len(times)}")
    # print(f"Time range in trajectory data: {times[0]:.3f} to {times[-1]:.3f}")
    
    # Process each window or cycle
    for i, (win_start, win_end, _) in enumerate(windows):  # Removed t_poi
        print(f"\nProcessing window {i+1}:")
        print(f"  Window time range: {win_start:.3f} to {win_end:.3f}")
        
        # Calculate segment times
        start_time = win_start
        end_time = win_end
        duration = end_time - start_time
        
        # Calculate window parameters
        beat_len = duration / n_beats_per_cycle
        subdiv_len = beat_len / n_subdiv_per_beat
        half_win = subdiv_len * nn
        
        # Calculate frame numbers for trajectory (240fps)
        traj_start_frame = int(start_time * frame_rate)
        traj_end_frame = int(end_time * frame_rate)
        traj_n_frames = traj_end_frame - traj_start_frame
        
        print(f"  Trajectory frames: {traj_start_frame} to {traj_end_frame} (240fps)")
        
        # Check if we have valid frame numbers
        if traj_start_frame >= traj_end_frame:
            print(f"  Skipping window {i+1}: Invalid frame range (start >= end)")
            continue
        if traj_start_frame < 0:
            print(f"  Skipping window {i+1}: Start frame < 0")
            continue
        if traj_end_frame > len(positions):
            print(f"  Skipping window {i+1}: End frame > total frames")
            continue
        
        # Trim trajectory data using frame numbers at 240fps
        pos_win = positions[traj_start_frame:traj_end_frame]
        t_win = times[traj_start_frame:traj_end_frame]
        
        # Check if we have valid trajectory data
        if len(pos_win) == 0:
            print(f"  Skipping window {i+1}: No trajectory data")
            continue
        
        print(f"  Trajectory data points: {len(pos_win)}")
        
        # Create figure and axis ------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.tight_layout(pad=3.0) 
        
        # Calculate all subdivision times for the window
        all_subdiv_times = []
        for beat_idx in range(0, n_beats_per_cycle + 1):  # Changed: now starts from 0
            beat_time = start_time + beat_idx * beat_len  # Changed: use start_time instead of downbeat
            for subdiv_idx in range(n_subdiv_per_beat):
                subdiv_time = beat_time + subdiv_idx * subdiv_len
                if start_time <= subdiv_time <= end_time:
                    all_subdiv_times.append((subdiv_time, beat_idx * n_subdiv_per_beat + subdiv_idx + 1))

        # Plot subdivision lines with appropriate colors
        for subdiv_time, subdiv_num in all_subdiv_times:
            color = get_subdiv_color(subdiv_num)
            if subdiv_num in [1, 4, 7, 10, 13]:
                ax.axvline(subdiv_time, color=color, linestyle='-', linewidth=2, alpha=0.7) #beat color
            else:
                ax.axvline(subdiv_time, color=color, linestyle='--', linewidth=1, alpha=0.3) #subdivision color
        
        # Plot trajectory
        ax.plot(t_win, pos_win, '-', color='green', alpha=0.5, label=f'{joint_name} {axis.upper()}', linewidth=3.5)
        
        # Set y-axis limits with safety checks
        try:
            y_min = pos_win.min()
            y_max = pos_win.max()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        except ValueError as e:
            print(f"  Warning: Could not set y-axis limits: {e}")
            ax.set_ylim(-1, 1)
        
        # Create vertical playhead
        v_playhead, = ax.plot([start_time, start_time], 
                            [y_min - 0.1*y_range, y_max + 0.1*y_range],
                            lw=1.5, alpha=0.9, color='orange')
        
        # Set up the plot with scaled x-axis
        ax.set_xlabel(f'Beat span')
        ax.set_ylabel(f'Pelvis') 
        # ax.set_ylabel(f'{joint_name} {bvh_to_mvnx[axis.lower()]} Position')      # {axis.upper()} y is vertical in bvh files, Z is vertical in mocap
        ax.set_title(f'{file_name} | Window:{start_time:.2f}s - {end_time:.2f}| Time: {start_time:.2f}s')
        ax.grid(True, alpha=0.3)
        
        # Scale x-axis to show beats instead of cycles
        x_ticks = np.arange(1, n_beats_per_cycle + 2)  # Changed: now 1 to 5
        x_tick_positions = start_time + (x_ticks - 1) * beat_len  # Changed: use start_time and adjust for 1-based indexing
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_ticks)
        ax.set_xlim(start_time, end_time)
        
        # Add legend
        custom = [
            Line2D([0],[0], color='green', lw=1.5),
            Line2D([0],[0], color='black', lw=1),
            Line2D([0],[0], color='green', lw=1, linestyle='--'),
            Line2D([0],[0], color='red', lw=1, linestyle='--'),
        ]
        labels = [
            f"{joint_name} {axis.upper()}", 
            "Subdiv-1 (1,4,7,10)", 
            "Subdiv-2 (2,5,8,11)", 
            "Subdiv-3 (3,6,9,12)"
        ]
        
        if legend_flag:
            ax.legend(custom, labels, loc='upper left', framealpha=0.3, fontsize=6)
        
        def update(frame):
            v_playhead.set_xdata([frame, frame])
            ax.set_title(f'{file_name} | Window:{start_time:.2f}s - {end_time:.2f}s| Time: {frame:.2f}s')
            return v_playhead,
        
        # Create animation frames at 24fps
        # n_frames = int(duration * 24)           # 
        # frames = np.linspace(start_time, end_time, n_frames)
        
        frames = np.arange(start_time, end_time, 1/24)      # New 06 June 2025
        anim = animation.FuncAnimation(
            fig, update, frames=frames,
            interval=1000/24,  # 24fps
            blit=True
        )
        
        # Save animation
        plot_output_path = os.path.join(output_dir2, f"{file_name}_window_{i+1:03d}_{start_time:.2f}_{end_time:.2f}.mp4")
        writer = animation.FFMpegWriter(fps= 24, 
                                        bitrate=2000,
                                        codec='libx264',  # Specify codec
                                        # extra_args=['-preset', 'ultrafast']
                                        )  # 24fps
        anim.save(plot_output_path, writer=writer)
        plt.close(fig)
        
        print(f"Plot saved: {plot_output_path}")
        print(f"Plot duration: {len(frames)/24:.3f}s")
    
    print("\nProcessing complete!")


#----------------------------- Concatenate videos ---------------------------------
#----------------------------------------------------------------------------------

def extract_category(filename):
    """
    Given a filename like "front_view_56.7_61.2.mp4" or
    "BKO_E1_D1_02_Maraka_pre_R_Mix_trimmed_56.7_61.2.mp4",
    return the category portion before the last two underscore-separated tokens.
    """
    name, _ = os.path.splitext(filename)     # strip .mp4
    parts = name.split('_')
    # Last two parts are start and end times, so category is everything before them
    if len(parts) > 2:
        return "_".join(parts[:-2])
    return name   # fallback if unexpected format

def write_all_categories(files, output_dir, video_dir):
    """
    From a list of filenames, group by category (as defined by extract_category),
    and write each group into its own .txt file in output_dir.
    """
    # os.makedirs(output_dir, exist_ok=True)

    # Group filenames by category
    categories = {}
    for fname in files:
        cat = extract_category(fname)
        categories.setdefault(cat, []).append(fname)

    # Write each category's filenames to a separate text file
    for cat, fnames in categories.items():
        txt_path = os.path.join(output_dir, f"{cat}.txt")
        with open(txt_path, "w") as fw:
            for f in fnames:
                if video_dir:
                    rel_path = os.path.relpath(os.path.join(video_dir, f), os.path.dirname(txt_path))
                    fw.write(f"file '{rel_path}'\n")
                else:
                    fw.write(f + "\n")  
                

def create_concat_file(video_dir, output_file, prefix):
    """Create a text file listing all videos in order for concatenation"""
    with open(output_file, 'w') as f:
        # Get all video files and sort them
        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
        # Write each file path - use relative path from the text file location
        for video in video_files:
            # Get relative path from output_file to video_dir
            rel_path = os.path.relpath(os.path.join(video_dir, video), os.path.dirname(output_file))
            f.write(f"file '{rel_path}'\n") 
            
def concatenate_and_overlay_videos(file_name, joint_name,  save_dir, views_to_generate):
    """Concatenate cycle videos and plot videos, then overlay them"""
    video_dir = os.path.join(save_dir, "videos")
    plot_dir = os.path.join(save_dir, "plots")
    joint_dir = os.path.join(save_dir, joint_name)
    vid_skel_dir = os.path.join(save_dir, "video_skeleton")
    drum_dot_dir = os.path.join(save_dir, "drum_dot_merged")
    dance_dot_dir = os.path.join(save_dir, "dance_dot")

    # Create text files for concatenation
    video_list = os.path.join(save_dir, "video_list.txt")
    plot_list = os.path.join(save_dir, "plot_list.txt")
    joint_list = os.path.join(save_dir, "joint_list.txt")
    
    drum_dot_list = os.path.join(save_dir, "drum_dot_list.txt")
    dance_dot_list = os.path.join(save_dir, "dance_dot_list.txt")
    
    # 'front', 'right' 'left', 'top'
    if 'front' in views_to_generate:
        front_view_list = os.path.join(save_dir, "front_view.txt")
    if 'left' in views_to_generate:
        left_view_list = os.path.join(save_dir, "left_view.txt")
    if 'right' in views_to_generate:
        right_view_list = os.path.join(save_dir, "right_view.txt")
    if 'top' in views_to_generate:
        top_view_list = os.path.join(save_dir, "top_view.txt")
    
    
    mp4_file_list = [f for f in os.listdir(vid_skel_dir) if f.lower().endswith(".mp4")]
    write_all_categories(mp4_file_list, save_dir, video_dir = vid_skel_dir)
    
    
    # Check if directories exist
    if not os.path.exists(video_dir):
        print(f"Video directory not found: {video_dir}")
        return
    if not os.path.exists(plot_dir):
        print(f"Plot directory not found: {plot_dir}")
        return
        
    # Check if text files already exist
    if os.path.exists(video_list) and os.path.exists(plot_list):
        print("Concatenation files already exist, skipping creation")
    else:
        print("Creating concatenation files...")
        create_concat_file(video_dir, video_list, f"{file_name}_cycle_")
        create_concat_file(plot_dir, plot_list, f"{file_name}_cycle_")
        create_concat_file(joint_dir, joint_list, f"{file_name}_cycle_")
        create_concat_file(drum_dot_dir, drum_dot_list, f"{file_name}_cycle_")
        create_concat_file(dance_dot_dir, dance_dot_list, f"{file_name}_cycle_")
    
    def concatenate_videos(video_list, save_dir, save_name):
        
        concat_video = os.path.join(save_dir, f"{save_name}.mp4")
        try:
            result = subprocess.run([
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', video_list,
                '-c', 'copy',
                concat_video
            ], capture_output=True, text=True)
            if result.returncode != 0:
                print("Error concatenating videos:", result.stderr)
                return
        except Exception as e:
            print("Error running ffmpeg:", str(e))
            return
    
    concatenate_videos(video_list, save_dir, f"video_mix_concat")
    concatenate_videos(plot_list, save_dir, f"plot_concat")
    concatenate_videos(joint_list, save_dir, f"joint_{joint_name}_concat")
    
    concatenate_videos(drum_dot_list, save_dir, f"drum_dot_concat")
    concatenate_videos(dance_dot_list, save_dir, f"dance_dot_concat")
    
    if 'front' in views_to_generate:    
        concatenate_videos(front_view_list, save_dir, f"front_view_concat")
    if 'left' in views_to_generate:
        concatenate_videos(left_view_list, save_dir, f"left_view_concat")
    if 'right' in views_to_generate:
        concatenate_videos(right_view_list, save_dir, f"right_view_concat")
    if 'top' in views_to_generate:
        concatenate_videos(top_view_list, save_dir, f"top_view_concat") 

    
    print(f"Concatenation complete: {save_dir}")



#----------------------------- Resize video ----------------------------------------
#----------------------------------------------------------------------------------
def resize_video(video_path, width, height, save_dir):
    """
    Resize a video to the specified width and height using ffmpeg,
    with debug‐level output.

    Parameters:
    - video_path: str, path to the input video file
    - width: int, target width in pixels
    - height: int, target height in pixels
    - save_dir: str, directory where the resized video will be saved

    The output filename will be: <original_basename>_<width>x<height>.mp4
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Derive output filename from input basename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{base_name}.mp4"   # f"{base_name}_{width}x{height}.mp4"
    output_path = os.path.join(save_dir, output_filename)

    # Build ffmpeg command with debug-level logging
    cmd = [
        "ffmpeg",
        "-y",                   # overwrite output if it exists
        "-loglevel", "debug",   # show full debug output
        "-i", video_path,
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        output_path
    ]

    # Run ffmpeg and capture stdout/stderr
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print debug output
    # print("=== ffmpeg stdout ===")
    # print(result.stdout)
    # print("=== ffmpeg stderr ===")
    # print(result.stderr)

    # Check return code and report
    if result.returncode == 0:
        print(f"Resizing succeeded, output saved to: {output_path}")
    else:
        print(f"ffmpeg failed with return code {result.returncode}")

    return output_path if result.returncode == 0 else None


#----------------------------- Create composite video -----------------------------
#----------------------------------------------------------------------------------

def create_composite_video(composite_video_elements, final_out):

    video_positions = []
    for element in composite_video_elements:
        video_positions.append({
            'path': element['vid_path'],
            'x': element['x_pos_pxl'],
            'y': element['y_pos_pxl']
        })

    # Build the ffmpeg command
    ffmpeg_inputs = []
    for pos in video_positions:
        ffmpeg_inputs.extend(['-i', pos['path']])

    # Create the xstack layout string
    # Format: xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0
    layout = []
    for pos in video_positions:
        layout.append(f"{pos['x']}_{pos['y']}")

    xstack_layout = "|".join(layout)

    # final_out = os.path.join(base_output_dir, f"{file_name}_{start_time:.2f}_{end_time:.2f}.mp4")
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        *ffmpeg_inputs,
        # '-filter_complex', f'xstack=inputs={len(video_positions)}:layout={xstack_layout}[v]:fill=black[v]',
        '-filter_complex', f'xstack=inputs={len(video_positions)}:layout={xstack_layout}:fill=black[v]',
        '-map', '[v]',
        '-map', '0:a?', '-c:a', 'aac', '-b:a', '192k',
        '-c:v', 'libx264',   #'libx264',
        '-crf', '23',
        '-preset', 'ultrafast',
        final_out
    ]

    # Execute the command

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video successfully created as {final_out}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")

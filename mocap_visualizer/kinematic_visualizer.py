import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation

def visualize_joint_position(
    bvh_file,
    joint_name,
    start_time,
    end_time,
    axis='x',  # 'x', 'y', or 'z'
    output_fps=24,
    output_dir=None,
    fig_size=(1280, 360),
    dpi=200,
):
    """
    Visualize joint position data for a single axis with a play head
    
    Args:
        bvh_file: Path to BVH file
        joint_name: Name of joint to visualize (e.g., 'Hips', 'RightHand')
        axis: Which axis to plot ('x', 'y', or 'z')
        start_time: Start time in seconds
        end_time: End time in seconds
        output_fps: Output video FPS
        output_dir: Directory to save output files
        fig_size: Output figure size (width, height)
    """
    # 1. Get output directory
    if output_dir is None:
        filename = os.path.splitext(os.path.basename(bvh_file))[0]
        dir_name = f"{filename}_{start_time:.1f}_{end_time:.1f}"
        output_dir = os.path.join("output", dir_name)
        os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load CSV file from the same directory as BVH file
    base_name = os.path.splitext(os.path.basename(bvh_file))[0]
    worldpos_file = os.path.join(os.path.dirname(bvh_file), f"{base_name}_worldpos.csv")
    world_positions = pd.read_csv(worldpos_file)
    
    # 3. Get time column and find frame indices
    time_column = world_positions.columns[0]  # First column is time
    mask = (world_positions[time_column] >= start_time) & (world_positions[time_column] <= end_time)
    filtered_data = world_positions[mask]
    
    # Get position data for specified axis
    time_data = filtered_data[time_column]
    axis_data = filtered_data[f"{joint_name}.{axis.upper()}"]
    
    # 4. Create the plot
    fig, ax = plt.subplots(figsize= fig_size, dpi=dpi)
    
    # Plot position data
    ax.plot(time_data, axis_data, 'b-', label=f'{axis.upper()}-position')
    
    # Add play head line
    play_head = ax.axvline(x=start_time, color='r', linestyle='--')
    
    # Add title and labels
    ax.set_title(f'Position of {joint_name} along {axis.upper()}-axis', fontsize=12, pad=10)
    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel(f'{axis.upper()}-position', fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 5. Create animation
    def update(frame):
        current_time = start_time + (frame / output_fps)
        play_head.set_xdata([current_time, current_time])
        return play_head,
    
    # 6. Save animation
    anim = FuncAnimation(fig, update, frames=int((end_time - start_time) * output_fps),
                        interval=1000/output_fps, blit=True)
    
    # 7. Save video
    output_file = os.path.join(output_dir, f"{joint_name}_{axis}_position.mp4")
    anim.save(output_file, writer='ffmpeg', fps=output_fps)
    
    plt.close()
    return output_file 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils_subdivision.gen_distribution_single_plots import find_cycle_phases, kde_estimate

def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'

def plot_foot_onsets_stacked(file_name,
                             dance_mode,
                             cycles_csv_path,
                             left_onsets,
                             right_onsets,
                             dance_mode_time_segments,
                             figsize=(10, 3),
                             dpi=200,
                             use_window=True):
    """Plot left and right foot onsets with stacked scatter and combined KDE, using robust phase and KDE calculation."""
    
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cycles = pd.read_csv(cycles_csv_path)["Virtual Onset"].values

    vertical_ranges = {
        'left': (1, 6),
        'right': (8, 13),                    # 'right': (1, 6),       # 'right': (8, 13)
    }

    combined_phases = []
    dance_phases_kde = {"phases": {}, "y_scaled": {}, "kde_h": {}, "kde_xx": {} }
    
    for foot_type, onsets, color in [('left', left_onsets, '#1f77b4'), ('right', right_onsets, '#d62728')]:
        # Filter onsets by time segments
        if use_window:
            window_mask = np.zeros(len(onsets), dtype=bool)
            for W_start, W_end in dance_mode_time_segments:
                segment_mask = (onsets >= W_start) & (onsets <= W_end)
                window_mask |= segment_mask
            filtered_onsets = onsets[window_mask]
        else:
            filtered_onsets = onsets

        if len(filtered_onsets) == 0:
            continue

        # Use robust phase calculation
        cycle_indices, phases, valid_onsets = find_cycle_phases(filtered_onsets, cycles)
        if len(phases) == 0:
            continue

        # Collect for combined KDE
        combined_phases.extend(phases)

        # Calculate window positions --------------------------------------------------------
        window_positions = []
        if use_window:
            for onset in valid_onsets:
                for seg_idx, (W_start, W_end) in enumerate(dance_mode_time_segments):
                    if W_start <= onset <= W_end:
                        segment_duration = W_end - W_start
                        relative_pos = (onset - W_start) / segment_duration
                        window_pos = seg_idx + relative_pos
                        window_positions.append(window_pos)
                        break
        else:
            window_positions = np.zeros_like(valid_onsets)

        window_positions = np.array(window_positions)
        y0, y1 = vertical_ranges[foot_type]
        y_scaled = y0 + (window_positions * (y1 - y0))

        # Plot scatter
        ax.scatter(phases * 400, y_scaled, s=5, alpha=0.6, color=color, label=f'{foot_type.capitalize()} Foot')

        # Collect for combined KDE
        dance_phases_kde["phases"][foot_type] = phases
        dance_phases_kde["y_scaled"][foot_type] = y_scaled
    
        

    # Combined KDE at bottom using kde_estimate ----------------------------------------------
    if len(combined_phases) > 0:
        
        kde_xx, kde_h = kde_estimate(np.array(combined_phases), SIG=0.01)
        
        # Only plot the region that maps to the x-axis
        mask = (kde_xx * 400 >= -33) & (kde_xx * 400 <= 400)
        kde_xx_plot = kde_xx[mask]
        kde_h_plot = kde_h[mask]
        
        if np.max(kde_h_plot) > 0:
            kde_scaled = -5 + (5 * kde_h_plot / np.max(kde_h_plot))
            ax.fill_between(kde_xx_plot * 400, -5, kde_scaled, alpha=0.3, color='purple', label='Combined KDE')

        # Collect for combined KDE
        dance_phases_kde["kde_h"] = kde_h_plot
        dance_phases_kde["kde_xx"] = kde_xx_plot
    
    
    # Subdivision lines --------------------------------------------------------------------
    for subdiv in range(1, 13):
        color = get_subdiv_color(subdiv)
        x_pos = ((subdiv-1) * 400) / 12
        
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(x_pos, -5.5, 20.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    # Styling ------------------------------------------------------------------------------
    # xtick = [0, 33, 67, 100, 133, 167, 200, 233, 267, 300, 333, 367, 400]
    xtick = [0, 100, 200, 300, 400]
    xtick_labels = [1, 2, 3, 4, 5]
    
    
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-33, 400)
    ax.set_xlabel('Beat span')
    
    ax.set_ylim(-5.5, 13.5)
    ax.set_yticks([3, 10])
    ax.set_yticklabels(['LF', 'RF'])
    ax.set_ylabel('Foot')
    ax.grid(True, alpha=0.3)

    # Title & legend
    title = f'File: {file_name} | Dance Mode: {dance_mode}'
    title += f' | Segments: {len(dance_mode_time_segments)}' if use_window else ' | Full Recording'
    ax.set_title(title, pad=10)
    ax.legend(loc='upper left', framealpha=0.4, fontsize=6)

    return fig, ax, dance_phases_kde


def plot_combined_foot_stacked(piece_type, dance_mode, dance_phases_kde_all, figsize=(10, 3), dpi=200):
    """Create a single plot showing combined foot analysis for all pieces of a type"""
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    vertical_ranges = {
        'left': (1, 6),
        'right': (8, 13),
    }

    combined_phases = []
    
    # Combine data for each foot
    for foot_type, color in [('left', '#1f77b4'), ('right', '#d62728')]:
        # Combine phases and y_scaled from all pieces
        all_phases = []
        all_y_scaled = []
        segment_kde_h = None
        segment_kde_xx = None

        # Loop through all pieces' data
        for piece_data in dance_phases_kde_all:
            if foot_type in piece_data["phases"]:
                # Combine phases and y_scaled
                all_phases.extend(piece_data["phases"][foot_type])
                all_y_scaled.extend(piece_data["y_scaled"][foot_type])
                combined_phases.extend(piece_data["phases"][foot_type])

        if not all_phases:  # Skip if no data found
            continue

        # Convert lists to numpy arrays
        phases = np.array(all_phases)
        y_scaled = np.array(all_y_scaled)
            
        # Plot scatter with single color for each foot
        ax.scatter(phases * 400,
                   y_scaled,
                   s=5, alpha=0.6,
                   color=color,
                   label=f'{foot_type.capitalize()} Foot')

    # Combined KDE at bottom using kde_estimate
    if len(combined_phases) > 0:
        kde_xx, kde_h = kde_estimate(np.array(combined_phases), SIG=0.01)
        
        # Only plot the region that maps to the x-axis
        mask = (kde_xx * 400 >= -33) & (kde_xx * 400 <= 400)
        kde_xx_plot = kde_xx[mask]
        kde_h_plot = kde_h[mask]
        
        if np.max(kde_h_plot) > 0:
            kde_scaled = -5 + (5 * kde_h_plot / np.max(kde_h_plot))
            ax.fill_between(kde_xx_plot * 400, -5, kde_scaled, alpha=0.3, color='purple', label='Combined KDE')

    # Subdivision lines
    for subdiv in range(1, 13):
        color = get_subdiv_color(subdiv)
        x_pos = ((subdiv-1) * 400) / 12
        
        if subdiv in [1, 4, 7, 10]:
            ax.vlines(x_pos, -5.5, 13.5, color=color, linestyle='-', linewidth=1.5, alpha=0.7)
        else:
            ax.vlines(x_pos, -5.5, 13.5, color=color, linestyle='--', linewidth=1, alpha=0.3)

    # Styling
    xtick = [0, 100, 200, 300, 400]
    xtick_labels = [1, 2, 3, 4, 5]
    
    ax.set_xticks(xtick)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-33, 400)
    ax.set_xlabel('Beat span')
    
    ax.set_ylim(-5.5, 13.5)
    ax.set_yticks([3, 10])
    ax.set_yticklabels(['LF', 'RF'])
    ax.set_ylabel('Foot')
    ax.grid(True, alpha=0.3)

    # Title & legend
    title = f'Piece: {piece_type} | Dance Mode: {dance_mode}'
    title += f' | Combined from {len(dance_phases_kde_all)} pieces'
    ax.set_title(title, pad=10)
    ax.legend(loc='upper left', framealpha=0.4, fontsize=6)

    return fig, ax


def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from matplotlib.lines import Line2D
from scipy.interpolate import interp1d


def plot_all_cycles_trajectories(
    file_name: str,
    mode: str,
    base_path_cycles: str = "data/virtual_cycles",
    base_path_logs: str = "data/logs_v2_may",
    frame_rate: float = 240,
    time_segments: list = None,  # List of (start, end) tuples
    n_beats_per_cycle: int = 4,
    n_subdiv_per_beat: int = 3,
    figsize: tuple = (12, 6),
    dpi: int = 200,
    show_trajectories: bool = True,  # Control trajectory lines
    show_vlines: bool = True,        # Control vertical lines
    show_gray_plots: bool = True     # Control gray trajectory plots
):
    """
    Plot all foot trajectories in a single plot with grand average.
    Shows beat and subdivision lines with colors.
    X-axis shows beats 1-4 directly.
    """
    # Use default window if no segments provided
    if time_segments is None:
        time_segments = [(0, 10)]

    # build file paths
    cycles_csv = os.path.join(base_path_cycles, f"{file_name}_C.csv")
    logs_onset_dir = os.path.join(base_path_logs, f"{file_name}_T", "onset_info")
    left_onsets_csv  = os.path.join(logs_onset_dir, f"{file_name}_T_left_foot_onsets.csv")
    right_onsets_csv = os.path.join(logs_onset_dir, f"{file_name}_T_right_foot_onsets.csv")
    left_zpos_csv    = os.path.join(logs_onset_dir, f"{file_name}_T_left_foot_zpos.csv")
    right_zpos_csv   = os.path.join(logs_onset_dir, f"{file_name}_T_right_foot_zpos.csv")

    # load data
    Lz = pd.read_csv(left_zpos_csv)["zpos"].values
    Rz = pd.read_csv(right_zpos_csv)["zpos"].values
    n_frames = len(Lz)
    times = np.arange(n_frames) / frame_rate

    # interpolation functions
    L_interp = interp1d(times, Lz, bounds_error=False, fill_value="extrapolate")
    R_interp = interp1d(times, Rz, bounds_error=False, fill_value="extrapolate")

    # Get overall time range for color mapping
    total_start = min(seg[0] for seg in time_segments)
    total_end = max(seg[1] for seg in time_segments)
    t_range = total_end - total_start

    # Calculate average cycle duration from all segments
    all_onsets = []
    for seg_start, seg_end in time_segments:
        cyc_df = pd.read_csv(cycles_csv)
        cyc_df = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        if not cyc_df.empty:
            all_onsets.extend(cyc_df["Virtual Onset"].values[:-1])
    
    if not all_onsets:
        raise ValueError("No cycles found in any of the time segments")
    
    all_onsets = np.sort(all_onsets)
    
    # Calculate average cycle duration
    durations = np.diff(sorted(all_onsets))
    avg_cycle = durations.mean()

    # Calculate beat and subdivision lengths
    # beat_len = avg_cycle / n_beats_per_cycle
    # subdiv_len = beat_len / n_subdiv_per_beat

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.get_cmap('cool')

    # Define subdivision color mapping
    def get_subdiv_color(subdiv):
        total_subdiv = n_beats_per_cycle * n_subdiv_per_beat
        subdiv = ((subdiv - 1) % total_subdiv) + 1
        group = ((subdiv - 1) % 3) + 1
        if group == 1:
            return 'black'
        elif group == 2:
            return 'green'
        elif group == 3:
            return 'red'
        return 'gray'

    # Process each time segment
    all_L_trajectories = []
    all_R_trajectories = []
    all_times = []

    for seg_start, seg_end in time_segments:
        # trim to window
        win_mask = (times >= seg_start) & (times <= seg_end)
        t_win = times[win_mask]
        L_win = Lz[win_mask]
        R_win = Rz[win_mask]

        # cycles (downbeats)
        cyc_df = pd.read_csv(cycles_csv)
        cyc_df = cyc_df[(cyc_df["Virtual Onset"] >= seg_start) & (cyc_df["Virtual Onset"] <= seg_end)]
        onsets = cyc_df["Virtual Onset"].values[:-1]

        # foot onsets
        left_df  = pd.read_csv(left_onsets_csv)
        right_df = pd.read_csv(right_onsets_csv)
        left_times  = left_df[(left_df["time_sec"]>=seg_start)&(left_df["time_sec"]<=seg_end)]["time_sec"].values
        right_times = right_df[(right_df["time_sec"]>=seg_start)&(right_df["time_sec"]<=seg_end)]["time_sec"].values

        # Plot trajectories for each cycle
        # for c in onsets:
        for i, c in enumerate(onsets):
            # Convert time to beat position (1-4)
            # cycle_start = c
            # cycle_end = c + avg_cycle
            # m = (t_win >= cycle_start) & (t_win <= cycle_end)
            # tr = (t_win[m] - cycle_start) / beat_len  # Convert to beat positions 1-4
            
            cycle_start = c
            cycle_end = onsets[i + 1] if i < len(onsets) - 1 else c + avg_cycle  # fallback to avg only for last cycle
            cycle_duration = cycle_end - cycle_start
            beat_len = cycle_duration / n_beats_per_cycle
            
            # Convert time to beat position using actual cycle duration
            m = (t_win >= cycle_start) & (t_win <= cycle_end)
            tr = (t_win[m] - cycle_start) / (cycle_duration / n_beats_per_cycle)  # normalize to 0-4 beats
  
            
            if show_trajectories:
                col = cmap((c-total_start)/t_range)
                
                # # Check if this cycle has any onsets
                # has_left_onsets = any(cycle_start <= lt <= cycle_end for lt in left_times)
                # has_right_onsets = any(cycle_start <= rt <= cycle_end for rt in right_times)
                
                # # Plot gray trajectories only for cycles without onsets
                # if show_gray_plots:
                #     if not has_left_onsets:
                #         ax.plot(tr, L_win[m], '-', color='black', alpha=0.3)
                #     if not has_right_onsets:
                #         ax.plot(tr, R_win[m], '--', color='black', alpha=0.3)
                
                # # Plot colored trajectories for cycles with onsets
                # if has_left_onsets:
                #     ax.plot(tr, L_win[m], '-', color=col, alpha=0.3, label="Left Foot" if c==onsets[0] else "")
                # if has_right_onsets:
                #     ax.plot(tr, R_win[m], '--', color=col, alpha=0.3, label="Right Foot" if c==onsets[0] else "")
                
                # Check if this cycle has any onsets at all
                has_any_onsets = any(cycle_start <= t <= cycle_end for t in left_times) or any(cycle_start <= t <= cycle_end for t in right_times)
                
                # Plot gray trajectories only for cycles without any onsets
                if show_gray_plots and not has_any_onsets:
                    ax.plot(tr, L_win[m], '-', color='gray', alpha=0.9)
                    ax.plot(tr, R_win[m], '--', color='gray', alpha=0.9)
                
                # Plot colored trajectories for cycles with any onsets
                if has_any_onsets:
                    ax.plot(tr, L_win[m], '-', color=col, alpha=0.3, label="Left Foot" if c==onsets[0] else "")
                    ax.plot(tr, R_win[m], '--', color=col, alpha=0.3, label="Right Foot" if c==onsets[0] else "")
                            
                # Store ALL trajectories for grand average (removed the condition)
                all_L_trajectories.append(L_win[m])
                all_R_trajectories.append(R_win[m])
                all_times.append(tr)

                # Plot markers for foot onsets
                for lt in left_times:
                    if cycle_start <= lt <= cycle_end:
                        rel = (lt - cycle_start) / beat_len
                        if show_vlines:
                            ax.axvline(rel, color=col, linestyle='-', alpha=0.5)
                        ax.plot(rel, L_interp(lt), 'o', ms=8, markeredgecolor='k', 
                                markerfacecolor='blue', alpha=0.8)

                for rt in right_times:
                    if cycle_start <= rt <= cycle_end:
                        rel = (rt - cycle_start) / beat_len
                        if show_vlines:
                            ax.axvline(rel, color=col, linestyle='--', alpha=0.5)
                        ax.plot(rel, R_interp(rt), 'x', ms=8, markeredgecolor='red', 
                                color='red', alpha=0.8)

    # Calculate and plot grand average
    if all_L_trajectories and all_R_trajectories:
        # Interpolate all trajectories to the same time points
        common_times = np.linspace(0, n_beats_per_cycle, 100)
        
        L_avg = np.zeros(len(common_times))
        R_avg = np.zeros(len(common_times))
        count = 0
        
        for L_traj, R_traj, t_traj in zip(all_L_trajectories, all_R_trajectories, all_times):
            if len(t_traj) > 1:  # Only use trajectories with more than one point
                L_interp = interp1d(t_traj, L_traj, bounds_error=False, fill_value="extrapolate")
                R_interp = interp1d(t_traj, R_traj, bounds_error=False, fill_value="extrapolate")
                L_avg += L_interp(common_times)
                R_avg += R_interp(common_times)
                count += 1
        
        if count > 0:
            L_avg /= count
            R_avg /= count
            ax.plot(common_times, L_avg, '-', color='blue', linewidth=3, label='Left Foot Average')
            ax.plot(common_times, R_avg, '--', color='red', linewidth=3, label='Right Foot Average')

    # # Calculate and plot grand average combined left and right
    # if all_L_trajectories and all_R_trajectories:
    #     # Interpolate all trajectories to the same time points
    #     common_times = np.linspace(0, n_beats_per_cycle, 100)
        
    #     # Single array for combined average
    #     combined_avg = np.zeros(len(common_times))
    #     count = 0
        
    #     for L_traj, R_traj, t_traj in zip(all_L_trajectories, all_R_trajectories, all_times):
    #         if len(t_traj) > 1:  # Only use trajectories with more than one point
    #             L_interp = interp1d(t_traj, L_traj, bounds_error=False, fill_value="extrapolate")
    #             R_interp = interp1d(t_traj, R_traj, bounds_error=False, fill_value="extrapolate")
    #             # Combine left and right trajectories
    #             combined_avg += (L_interp(common_times) + R_interp(common_times)) / 2
    #             count += 1
        
    #     if count > 0:
    #         combined_avg /= count
    #         ax.plot(common_times, combined_avg, '-', color='purple', linewidth=3, label='Combined Foot Average')
        
    # Add vertical line at position 0 (will display as 1)
    ax.axvline(0, color='black', linewidth=2, alpha=0.8)
    
    # Add beat and subdivision lines
    for beat in range(1, n_beats_per_cycle + 1):
        ax.axvline(beat, color='black', linewidth=2, alpha=0.8)
        # Add subdivision lines
        for subdiv in range(1, n_subdiv_per_beat):
            subdiv_pos = beat - 1 + subdiv/n_subdiv_per_beat
            subdiv_num = (beat - 1) * n_subdiv_per_beat + subdiv + 1
            
            grid_color = get_subdiv_color(subdiv_num)
            ax.axvline(subdiv_pos, color=grid_color, alpha=0.8, linewidth=1.5)
            
            

    ax.set_xlabel("Cycle Span")
    ax.set_ylabel("Foot Y Position")
    ax.set_title(f"All Cycles Trajectories with Grand Average\n{file_name} | {mode}")
    # ax.grid(True, alpha=0.3)
    xticks = [0.0, 0.33, 0.67, 1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0]
    ax.set_xticks(xticks)
    ax.set_xticklabels([1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0, 3.33, 3.67, 4.0, 4.33, 4.67, 5.0])
    ax.set_xlim(0.0, 4.0)
    # Add all legends together outside the plot
    custom = [
        # Main trajectory and onset markers
        Line2D([0],[0],marker='o', color='w', markerfacecolor='blue', ms=8, markeredgecolor='k'),
        Line2D([0],[0],marker='x', color='red', ms=8),
        Line2D([0],[0],color='blue', lw=3),
        Line2D([0],[0],color='red', lw=3, linestyle='--'),
        # Line2D([0],[0],color='purple', lw=3),  # For combined average
        # Subdivision lines
        Line2D([0],[0],color='gray', lw=1.5),
        Line2D([0],[0],color='black', lw=1.5),
        Line2D([0],[0],color='green', lw=1.5),
        Line2D([0],[0],color='red', lw=1.5)
    ]
    labels = [
        "Left Onset", 
        "Right Onset", 
        "Left Foot Average", 
        "Right Foot Average", 
        # "Combined Average",
        "Undetected trajectory",
        "Subdivision 1 (1,4,7,10)", 
        "Subdivision 2 (2,5,8,11)", 
        "Subdivision 3 (3,6,9,12)"
    ]
    fig.legend(custom, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), framealpha=0.3)
    

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(total_start, total_end))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Time in recording (s)')

    plt.tight_layout()
    return fig, ax
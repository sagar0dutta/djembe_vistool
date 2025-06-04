import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def load_cycles(cycles_csv_path):
    """Load cycle onsets from a CSV file.
    
    Args:
        cycles_csv_path: Path to the CSV file containing cycle onsets
        
    Returns:
        numpy array: List of cycle locations in seconds
    """
    df = pd.read_csv(cycles_csv_path)
    return np.sort(np.array(df['Virtual Onset'].values))   # 'Selected Onset'

def load_onsets(onsets_csv_path, onset_type):
    """Load drum onsets from a CSV file.
    
    Args:
        onsets_csv_path: Path to the CSV file containing drum onsets
        onset_type: Type of onset to load ('Dun', 'J1', or 'J2')
        
    Returns:
        numpy array: List of onset locations in seconds
    """
    df = pd.read_csv(onsets_csv_path)
    return np.sort(np.array(df[onset_type].values))

def find_cycle_phases(onsets, cycles):
    """Find the cycle and phase for each onset, handling circular nature of metric cycle.
    
    Args:
        onsets: Array of onset times
        cycles: Array of cycle times
        
    Returns:
        tuple: (cycle_indices, phases, valid_onsets)
    """
    cycle_indices = []
    phases = []
    valid_onsets = []
    
    for onset in onsets:
        # Find which cycle contains this onset
        idx = np.searchsorted(cycles, onset)
        
        # Skip if onset is before first cycle or after last cycle
        if idx == 0 or idx >= len(cycles):
            continue
            
        # Initially assign to the cycle that contains the onset
        c = idx - 1
        
        # Compute initial phase within this cycle
        L_c = cycles[c]
        L_c1 = cycles[c + 1]
        
        # Skip if cycle duration is zero or invalid
        if L_c1 <= L_c:
            continue
            
        # Compute initial phase
        f = (onset - L_c) / (L_c1 - L_c)
        
        # Handle phases near cycle boundaries
        if f > 0.95:  # Close to cycle end
            # For downbeat (first subdivision), include negative phase
            if c == 0:  # First cycle
                f = f - 1.0  # Make it negative
            else:
                # For other cycles, assign to next cycle
                c = c + 1
                if c + 1 < len(cycles):
                    L_c = cycles[c]
                    L_c1 = cycles[c + 1]
                    if L_c1 <= L_c:
                        continue
                    f = (onset - L_c) / (L_c1 - L_c)
        
        cycle_indices.append(c)
        phases.append(f)
        valid_onsets.append(onset)
    
    return np.array(cycle_indices), np.array(phases), np.array(valid_onsets)

def compute_window_positions(onsets, W_start, W_end):
    """Compute relative positions within the window.
    
    Args:
        onsets: Array of onset times
        W_start: Start of window
        W_end: End of window
        
    Returns:
        tuple: (positions, mask) - Relative positions y(i) and boolean mask of valid onsets
    """
    # Skip if window duration is zero or invalid
    if W_end <= W_start:
        return np.zeros_like(onsets), np.zeros_like(onsets, dtype=bool)
    
    # Create mask for onsets within window
    mask = (onsets >= W_start) & (onsets <= W_end)
    
    # Compute positions only for valid onsets
    positions = np.zeros_like(onsets)
    positions[mask] = (onsets[mask] - W_start) / (W_end - W_start)
    
    return positions, mask

def kde_estimate(data, SIG=0.01):
    """Compute KDE estimate handling negative phases.
    
    Args:
        data: Array of phase values
        SIG: Standard deviation for Gaussian kernel
        
    Returns:
        tuple: (xx, h) - grid points and density estimates
    """
    # Filter out invalid data points
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return np.array([]), np.array([])
    
    # Extend grid to include negative values
    xx = np.arange(-0.2, 1.001, 0.001)  # Include negative values
    h = np.zeros_like(xx)
    
    for d in valid_data:
        # Compute Gaussian distribution for current onset
        p = norm.pdf(xx, d, SIG)
        p_sum = np.sum(p)
        if p_sum > 0:  # Only normalize if sum is not zero
            p = p / p_sum
        h = h + p
    
    h_sum = np.sum(h)
    if h_sum > 0:  # Only normalize if sum is not zero
        h = h / h_sum
    
    return xx, h

def plot_results(phases, window_positions, kde_xx, kde_h, file_name, onset_type, dance_mode=None, W_start=None, W_end=None, save_path=None, figsize=(12, 8), dpi=100, use_window=True):
    """Create a single plot with scatter above zero and KDE below zero, sharing x-axis.
    
    Args:
        phases: Array of phase values
        window_positions: Array of window positions
        kde_xx: Grid points for KDE
        kde_h: Density estimates
        file_name: Name of the file being analyzed
        onset_type: Type of onset being analyzed
        W_start: Start of analysis window (optional if use_window=False)
        W_end: End of analysis window (optional if use_window=False)
        save_path: Path to save the plot (if None, plot will be displayed)
        figsize: Tuple of (width, height) in inches
        dpi: Dots per inch for the figure
        use_window: Whether to filter by time window (True) or use all data (False)
    """
    # Skip plotting if no valid data
    if len(phases) == 0 or len(kde_xx) == 0:
        print("No valid data to plot")
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot scatter points (above zero)
    ax.scatter(phases, window_positions, alpha=0.5, color='blue', s=5)
    
    # Scale KDE to be between -0.5 and 0, starting from bottom
    kde_scaled = -0.5 + (0.5 * kde_h / np.max(kde_h))
    ax.fill_between(kde_xx, -0.5, kde_scaled, alpha=0.3, color='orange')
    
    # Set axis labels
    ax.set_xlabel('Normalized metric cycle')
    ax.set_ylabel('Relative Position in Window')
    
    # Add title
    if use_window:
        ax.set_title(f'File: {file_name} | {dance_mode} | Onset: {onset_type}')
    else:
        ax.set_title(f'File: {file_name} | Full Recording | Onset: {onset_type}')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    ax.set_ylim(-0.55, 1.0)
    
    # Set x-axis limits to show negative values
    ax.set_xlim(-0.1, 1.0)
    xticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(xticks)
    
    # Draw vertical lines at each subdivision i/12
    ymin, ymax = ax.get_ylim()
    for subdiv  in range(1, 13):
        xpos = (subdiv - 1) / 12    # subdiv 1 → 0.0, subdiv 4 → 0.25, etc.
        ax.vlines(xpos, ymin, ymax, color=get_subdiv_color(subdiv), linewidth=1)
    
    # Customize y-axis ticks to only show positive values
    yticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(yticks)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def analyze_phases(cycles_csv_path, onsets_csv_path, onset_type, dance_mode_time_segments=None, dance_mode=None, save_path=None, SIG=0.01, figsize=(12, 8), dpi=100, use_window=True):
    """Main function to perform phase analysis.
    
    Args:
        cycles_csv_path: Path to the CSV file containing cycle onsets
        onsets_csv_path: Path to the CSV file containing drum onsets
        onset_type: Type of onset to analyze ('Dun', 'J1', or 'J2')
        dance_mode_time_segments: List of tuples containing (start_time, end_time) for each segment
        save_path: Path to save the plot (if None, plot will be displayed)
        SIG: Standard deviation for KDE
        figsize: Tuple of (width, height) in inches
        dpi: Dots per inch for the figure
        use_window: Whether to filter by time segments (True) or use all data (False)
    """
    # Load data
    cycles = load_cycles(cycles_csv_path)
    all_onsets = load_onsets(onsets_csv_path, onset_type)
    
    # Filter onsets by time segments if use_window is True
    if use_window:
        if dance_mode_time_segments is None or len(dance_mode_time_segments) == 0:
            raise ValueError("dance_mode_time_segments must be provided when use_window=True")
        
        # Create a mask for all segments
        window_mask = np.zeros(len(all_onsets), dtype=bool)
        for W_start, W_end in dance_mode_time_segments:
            segment_mask = (all_onsets >= W_start) & (all_onsets <= W_end)
            window_mask |= segment_mask
        
        onsets = all_onsets[window_mask]
        print(f"Total onsets: {len(all_onsets)}")
        print(f"Onsets in segments: {len(onsets)}")
    else:
        onsets = all_onsets
        print(f"Total onsets: {len(onsets)}")
    
    # Validate input data
    if len(cycles) < 2:
        print("Error: Need at least 2 cycle points")
        return None, None, None, None
    
    if len(onsets) == 0:
        print("Error: No onset points found")
        return None, None, None, None
    
    # Step 1: Find cycles and compute phases for filtered onsets
    cycle_indices, phases, valid_onsets = find_cycle_phases(onsets, cycles)
    
    if len(phases) == 0:
        print("Error: No valid phases computed")
        return None, None, None, None
    
    # Step 2: Compute window positions
    if use_window:
        # Calculate relative positions within the total duration of all segments
        total_duration = sum(end - start for start, end in dance_mode_time_segments)
        window_positions = np.zeros_like(valid_onsets)
        
        current_position = 0
        for W_start, W_end in dance_mode_time_segments:
            segment_mask = (valid_onsets >= W_start) & (valid_onsets <= W_end)
            segment_duration = W_end - W_start
            relative_positions = (valid_onsets[segment_mask] - W_start) / segment_duration
            window_positions[segment_mask] = current_position + relative_positions * (segment_duration / total_duration)
            current_position += segment_duration / total_duration
    else:
        # When not using window, distribute points evenly
        window_positions = np.linspace(0, 1, len(valid_onsets))
    
    # Step 3: KDE estimation
    kde_xx, kde_h = kde_estimate(phases, SIG)
    
    # Step 4: Plot results
    file_name = os.path.basename(cycles_csv_path).replace('_C.csv', '')
    plot_results(phases, window_positions, kde_xx, kde_h, file_name, onset_type, dance_mode, 
                dance_mode_time_segments[0][0] if use_window else None, 
                dance_mode_time_segments[-1][1] if use_window else None, 
                save_path, figsize, dpi, use_window)
    
    return phases, window_positions, kde_xx, kde_h

def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'
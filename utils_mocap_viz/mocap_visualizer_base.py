import numpy as np
import pyvista as pv
import time
import sys
import os
import pandas as pd
from scipy.spatial.transform import Rotation
from bvh_converter import bvh_mod

class MocapVisualizerBase:
    def __init__(self, bvh_file, debug=False):
        self.debug = debug
        try:
            if not bvh_file.endswith('.bvh'):
                raise ValueError("Please provide a .bvh file")
            
            self.dir_csv = "extracted_mocap_csv"
            self.load_bvh(bvh_file)
            
            # Create PyVista plotter
            self.plotter = pv.Plotter(off_screen=True)
            self.plotter.set_background('white')
            
            # Define connections for standard skeleton
            self.connections = [
                # Spine
                ('Hips', 'Chest'),
                ('Chest', 'Chest2'),
                ('Chest2', 'Chest3'),
                ('Chest3', 'Chest4'),
                ('Chest4', 'Neck'),
                ('Neck', 'Head'),
                # Right Arm
                ('Chest4', 'RightCollar'),
                ('RightCollar', 'RightShoulder'),
                ('RightShoulder', 'RightElbow'),
                ('RightElbow', 'RightWrist'),
                # Left Arm
                ('Chest4', 'LeftCollar'),
                ('LeftCollar', 'LeftShoulder'),
                ('LeftShoulder', 'LeftElbow'),
                ('LeftElbow', 'LeftWrist'),
                # Right Leg
                ('Hips', 'RightHip'),
                ('RightHip', 'RightKnee'),
                ('RightKnee', 'RightAnkle'),
                ('RightAnkle', 'RightToe'),
                # Left Leg
                ('Hips', 'LeftHip'),
                ('LeftHip', 'LeftKnee'),
                ('LeftKnee', 'LeftAnkle'),
                ('LeftAnkle', 'LeftToe'),
            ]
            
            if self.debug:
                print("\nDebug Information:")
                print(f"Available markers: {self.labels}")
                print(f"T-pose positions: {self.t_pose_positions}")
                print(f"Connections: {self.connections}")
            
        except Exception as e:
            print(f"Error initializing visualizer: {str(e)}")
            sys.exit(1)

    def load_bvh(self, bvh_file):
        """Load data from BVH file"""
        # Convert BVH to CSV
        base_name = os.path.splitext(os.path.basename(bvh_file))[0] #os.path.splitext(bvh_file)[0]
        pos_csv = os.path.join(self.dir_csv, f"{base_name}_worldpos.csv")
        rot_csv = os.path.join(self.dir_csv, f"{base_name}_rotations.csv")
        
        if not os.path.exists(pos_csv) or not os.path.exists(rot_csv):
            if self.debug:
                print(f"Converting BVH to CSV files...")
            bvh_mod.convert_bvh_to_csv(bvh_file, do_rotations=True)
        
        # Load position data
        self.positions_df = pd.read_csv(pos_csv)
        
        # Get frame rate from the time column
        if 'Time' in self.positions_df.columns:
            # Calculate frame rate from the first two rows
            if len(self.positions_df) > 1:
                time_diff = self.positions_df['Time'].iloc[1] - self.positions_df['Time'].iloc[0]
                self.frame_rate = round(1.0 / time_diff)
            else:
                self.frame_rate = 240  # Default if we can't calculate
        else:
            self.frame_rate = 240  # Default if no time column
        
        # Get marker names (excluding 'end' markers and Time column)
        self.labels = [col.split('.')[0] for col in self.positions_df.columns 
                      if col != 'Time' and not col.endswith('End')]
        self.labels = list(dict.fromkeys(self.labels))  # Remove duplicates
        
        # Get frame information
        self.total_frames = len(self.positions_df)
        self.total_time = self.total_frames / self.frame_rate
        
        if self.debug:
            print(f"\nBVH file information:")
            print(f"Frame rate: {self.frame_rate} Hz")
            print(f"Total frames: {self.total_frames}")
            print(f"Total time: {self.total_time:.2f} seconds")
            print(f"Position columns: {self.positions_df.columns.tolist()}")
            print(f"Available markers: {self.labels}")
        
        # Store T-pose positions (first row after time column)
        self.t_pose_positions = {}
        for label in self.labels:
            x = self.positions_df.iloc[0][f"{label}.X"]
            y = self.positions_df.iloc[0][f"{label}.Y"]
            z = self.positions_df.iloc[0][f"{label}.Z"]
            self.t_pose_positions[label] = np.array([x, y, z])
        
        # Create label to index mapping for faster lookup
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        # Get reference position from first frame
        self.reference_pos = self.get_marker_position('Hips', 0)
        if self.reference_pos is None:
            raise ValueError("Reference marker 'Hips' not found in the first frame")
    
    def get_marker_position(self, label, frame):
        """Get the 3D position of a marker at a specific frame"""
        try:
            x = self.positions_df.iloc[frame][f"{label}.X"]
            y = self.positions_df.iloc[frame][f"{label}.Y"]
            z = self.positions_df.iloc[frame][f"{label}.Z"]
            return np.array([x, y, z])  # Keep original axis order: X (side), Y (vertical), Z (forward)
        except (KeyError, IndexError) as e:
            if self.debug:
                print(f"Warning: Marker {label} not found in frame {frame}: {str(e)}")
            return None
    
    def build_skeleton(self, frame):
        """Build the skeleton for a specific frame"""
        points = []
        lines = []
        
        # Get current pelvis position
        current_pelvis = self.get_marker_position('Hips', frame)
        if current_pelvis is None:
            if self.debug:
                print(f"Warning: Missing pelvis in frame {frame}")
            return False
            
        # Calculate translation offset
        translation_offset = self.reference_pos - current_pelvis
        
        # Collect valid marker positions
        marker_positions = {}
        for label in self.labels:
            pos = self.get_marker_position(label, frame)
            if pos is not None and not np.any(np.isnan(pos)):
                # Apply translation only (no rotation)
                pos = pos + translation_offset
                marker_positions[label] = pos
        
        if self.debug:
            print(f"Frame {frame} valid markers: {len(marker_positions)}/{len(self.labels)}")
        
        # Create points array and build connections
        points = []
        point_indices = {}
        current_idx = 0
        
        for start, end in self.connections:
            if start in marker_positions and end in marker_positions:
                # Add points if not already added
                if start not in point_indices:
                    points.append(marker_positions[start])
                    point_indices[start] = current_idx
                    current_idx += 1
                if end not in point_indices:
                    points.append(marker_positions[end])
                    point_indices[end] = current_idx
                    current_idx += 1
                
                # Add line connecting the points
                line = [2, point_indices[start], point_indices[end]]
                lines.append(line)
            elif self.debug:
                print(f"Warning: Missing connection {start} -> {end} in frame {frame}")
        
        if points:
            points = np.array(points)
            
            # Center and normalize the points
            center = np.mean(points, axis=0)
            points = points - center  # Center the points
            
            # Scale to fit in [-1, 1] box
            max_range = np.max(np.abs(points))
            if max_range > 0:
                points = points / max_range
            
            # Create the polydata with explicit cells array
            cells = []
            n_lines = len(lines)
            for line in lines:
                cells.extend(line)
            
            # Create cells array with proper format
            if n_lines > 0:
                cells = np.array(cells)
                self.skeleton = pv.PolyData(points, lines=cells)
                return True
            
        if self.debug:
            print(f"Warning: No valid skeleton built for frame {frame}")
        return False
    
    def generate_video(self, start_time=0, end_time=None, output_file="animation.mp4", output_fps=30, video_size=(1920, 1080), show_info=True):
        """Generate an MP4 video of the animation"""
        try:
            # Convert times to frames
            start_frame = int(start_time * self.frame_rate)
            if end_time is None:
                end_frame = self.total_frames
            else:
                end_frame = int(end_time * self.frame_rate)
            
            # Ensure valid frame range
            start_frame = max(0, min(start_frame, self.total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, self.total_frames))
            
            print(f"\nVideo generation settings:")
            print(f"Start time: {start_time:.2f}s (frame {start_frame})")
            print(f"End time: {end_time:.2f}s (frame {end_frame})")
            print(f"Output file: {output_file}")
            print(f"Output FPS: {output_fps}")
            print(f"Video size: {video_size[0]}x{video_size[1]}")
            
            # Create temporary directory for frames
            temp_dir = "temp_frames"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Initialize the first frame
            if self.build_skeleton(start_frame):
                # Add skeleton lines first (black)
                self.plotter.add_mesh(self.skeleton, color='black', line_width=5, render_lines_as_tubes=True)
                
                # Then add marker points (red)
                self.plotter.add_points(self.skeleton.points, color='red', point_size=10)
                
                # Add frame number and time if show_info is True
                if show_info:
                    self.plotter.add_text(f'Frame: {start_frame}/{self.total_frames}\nTime: {start_time:.2f}s\nFps: {output_fps:.1f}', position='upper_left')
                
                # Calculate frame step based on input and output FPS
                frame_step = max(1, int(self.frame_rate / output_fps))
                
                # Save frames
                frame_count = 0
                for frame in range(start_frame, end_frame, frame_step):
                    if self.build_skeleton(frame):
                        self.plotter.clear()
                        self.plotter.add_mesh(self.skeleton, color='black', line_width=5, render_lines_as_tubes=True)
                        self.plotter.add_points(self.skeleton.points, color='red', point_size=10)
                        
                        # Update frame number and time if show_info is True
                        if show_info:
                            current_time = frame / self.frame_rate
                            self.plotter.add_text(f'Frame: {frame}/{self.total_frames}\nTime: {current_time:.2f}s', position='upper_left')
                        
                        # Save frame to temporary directory with specified size
                        frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
                        self.plotter.screenshot(frame_path, window_size=video_size)
                        frame_count += 1
                        print(f"Saved frame {frame_count}/{((end_frame - start_frame) // frame_step)}", end='\r')
                
                print("\nConverting frames to video...")
                
                # Use ffmpeg to create video from frames with specified size
                ffmpeg_cmd = f'ffmpeg -y -framerate {output_fps} -i {temp_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p -s {video_size[0]}x{video_size[1]} {output_file}'
                os.system(ffmpeg_cmd)
                
                # Clean up temporary files
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
                
                print(f"Video generation complete! Saved to {output_file}")
            
        except Exception as e:
            print(f"Error during video generation: {str(e)}")
            self.plotter.close()
            sys.exit(1) 
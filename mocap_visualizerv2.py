import ezc3d
import numpy as np
import pyvista as pv
import time
import sys
import os

class MocapVisualizer:
    def __init__(self, c3d_file):
        try:
            # Load the C3D file
            self.c3d = ezc3d.c3d(c3d_file)
            
            # Get point data
            self.points_data = self.c3d['data']['points']
            self.labels = self.c3d['parameters']['POINT']['LABELS']['value']
            
            # Get frame rate and total frames
            self.frame_rate = self.c3d['header']['points']['frame_rate']
            self.total_frames = self.points_data.shape[2]
            self.total_time = self.total_frames / self.frame_rate
            
            print(f"\nMotion capture information:")
            print(f"Frame rate: {self.frame_rate} Hz")
            print(f"Total frames: {self.total_frames}")
            print(f"Total time: {self.total_time:.2f} seconds")
            
            # Create PyVista plotter
            self.plotter = pv.Plotter(off_screen=True)  # Create off-screen plotter
            self.plotter.set_background('white')
            
            # Define connections for standard skeleton using specified joint markers
            self.connections = [
                # Spine
                ('pSacrum', 'pL5SpinalProcess'),
                ('pL5SpinalProcess', 'pL3SpinalProcess'),
                ('pL3SpinalProcess', 'pT12SpinalProcess'),
                ('pT12SpinalProcess', 'pT8SpinalProcess'),
                ('pT8SpinalProcess', 'pT4SpinalProcess'),
                ('pT4SpinalProcess', 'pC7SpinalProcess'),
                # Head
                ('pC7SpinalProcess', 'pBackOfHead'),
                # ('pTopOfHead', 'pBackOfHead'),
                # Right Arm
                ('pC7SpinalProcess', 'pRightAcromion'),
                ('pRightAcromion', 'pRightOlecranon'),
                ('pRightOlecranon', 'pRightTopOfHand'),
                # ('pRightTopOfHand', 'pRightBallHand'),
                # Left Arm
                ('pC7SpinalProcess', 'pLeftAcromion'),
                ('pLeftAcromion', 'pLeftOlecranon'),
                ('pLeftOlecranon', 'pLeftTopOfHand'),
                # ('pLeftTopOfHand', 'pLeftBallHand'),
                # Right Leg
                ('pSacrum', 'pRightGreaterTrochanter'),
                ('pRightGreaterTrochanter', 'pRightPatella'),
                ('pRightPatella', 'pRightHeelFoot'),
                ('pRightHeelFoot', 'pRightToe'),
                # Left Leg
                ('pSacrum', 'pLeftGreaterTrochanter'),
                ('pLeftGreaterTrochanter', 'pLeftPatella'),
                ('pLeftPatella', 'pLeftHeelFoot'),
                ('pLeftHeelFoot', 'pLeftToe'),
            ]
            
            # Create label to index mapping for faster lookup
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
            
            # Get reference position from first frame
            self.reference_pos = self.get_marker_position('pSacrum', 0)
            if self.reference_pos is None:
                raise ValueError("Reference marker 'pSacrum' not found in the first frame")
            
            # Set up fixed camera position
            self.plotter.camera_position = [(0, 0, 3), (0, 0, 0), (0, 1, 0)]  # Front view
            self.plotter.camera.zoom(0.5)  # Zoom in a bit
            
        except Exception as e:
            print(f"Error loading C3D file: {str(e)}")
            sys.exit(1)
        
    def get_marker_position(self, label, frame):
        """Get the 3D position of a marker at a specific frame"""
        try:
            idx = self.label_to_idx[label]
            # Get the position and reorder axes if needed
            pos = self.points_data[:3, idx, frame]
            # Reorder axes: x, y, z -> x, z, y (assuming y is up in the C3D file)
            return np.array([pos[0], pos[2], pos[1]])
        except KeyError:
            print(f"Warning: Marker {label} not found")
            return None
    
    def build_skeleton(self, frame):
        """Build the skeleton for a specific frame"""
        points = []
        lines = []
        
        # Get current pelvis position
        current_pelvis = self.get_marker_position('pSacrum', frame)
        if current_pelvis is None:
            return False
            
        # Calculate translation offset
        translation_offset = self.reference_pos - current_pelvis
        
        # Collect valid marker positions
        marker_positions = {}
        for label in self.labels:
            pos = self.get_marker_position(label, frame)
            if pos is not None and not np.any(np.isnan(pos)):
                # Apply translation offset to lock skeleton in place
                marker_positions[label] = pos + translation_offset
        
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
            
        return False
    
    def generate_video(self, start_time=0, end_time=None, output_file="animation.mp4", output_fps=30):
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
            
            # Create temporary directory for frames
            temp_dir = "temp_frames"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Initialize the first frame
            if self.build_skeleton(start_frame):
                # Add skeleton lines first (black)
                self.plotter.add_mesh(self.skeleton, color='black', line_width=5, render_lines_as_tubes=True)
                
                # Then add marker points (red)
                self.plotter.add_points(self.skeleton.points, color='red', point_size=10)
                
                # Add frame number and time
                self.plotter.add_text(f'Frame: {start_frame}/{self.total_frames}\nTime: {start_time:.2f}s', position='upper_left')
                
                # Calculate frame step based on input and output FPS
                frame_step = max(1, int(self.frame_rate / output_fps))
                
                # Save frames
                frame_count = 0
                for frame in range(start_frame, end_frame, frame_step):
                    if self.build_skeleton(frame):
                        self.plotter.clear()
                        self.plotter.add_mesh(self.skeleton, color='black', line_width=5, render_lines_as_tubes=True)
                        self.plotter.add_points(self.skeleton.points, color='red', point_size=10)
                        
                        # Update frame number and time
                        current_time = frame / self.frame_rate
                        self.plotter.add_text(f'Frame: {frame}/{self.total_frames}\nTime: {current_time:.2f}s', position='upper_left')
                        
                        # Save frame to temporary directory
                        frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
                        self.plotter.screenshot(frame_path)
                        frame_count += 1
                        print(f"Saved frame {frame_count}/{((end_frame - start_frame) // frame_step)}", end='\r')
                
                print("\nConverting frames to video...")
                
                # Use ffmpeg to create video from frames
                ffmpeg_cmd = f'ffmpeg -y -framerate {output_fps} -i {temp_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p {output_file}'
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

if __name__ == "__main__":
    # Create visualizer
    visualizer = MocapVisualizer("BKO_E1_D5_01_Maraka_T.c3d")
    
    # Generate video with time selection
    start_time = 60.0  # Start at 70 seconds
    end_time = 125.0    # End at 80 seconds
    output_file = "mocap_animation2.mp4"
    output_fps = 24   # Output video frame rate
    
    visualizer.generate_video(start_time=start_time, end_time=end_time, output_file=output_file, output_fps=output_fps) 
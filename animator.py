import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import matplotlib.patches as patches
from ObstacleAvoidance import Obstacle


class DroneAnimator:
    def __init__(
        self,
        drones,
        charging_stations,
        blocks,
        routes,
        fcr,
        ugv_factor,
        ugv_init_loc=None,
        animation_speed=1.0,
        fps=30,
    ):
        """
        Initialize the drone animation.

        Parameters:
        -----------
        drones : list of tuples
            Each tuple contains ((init_x, init_y), (dest_x, dest_y), charge_level)
        charging_stations : numpy.ndarray
            Array of shape (m, 2) containing charging station coordinates
        blocks : list of lists
            Each inner list contains tuples of vertex coordinates for obstacles
        routes : list of lists
            Each inner list contains indices of charging stations to visit
        fcr : float
            Full charge range
        ugv_factor : float
            UGV cost factor
        ugv_init_loc : array, optional
            Initial UGV location
        animation_speed : float
            Speed multiplier for animation (1.0 = normal speed)
        fps : int
            Frames per second for animation
        """
        self.drones = drones
        self.charging_stations = charging_stations
        self.blocks = blocks
        self.routes = routes
        self.fcr = fcr
        self.ugv_factor = ugv_factor
        self.ugv_init_loc = ugv_init_loc
        self.animation_speed = animation_speed
        self.fps = fps

        # Generate colors for drones
        np.random.seed(242)
        self.colors = np.random.rand(len(drones), 3)

        # Calculate full paths for each drone including obstacle avoidance
        self.drone_paths = self._calculate_drone_paths()
        self.max_path_length = max(len(path) for path in self.drone_paths)
        
        # Calculate precise frame count needed for animation
        self._frames_needed = self._calculate_animation_frames()

        # Setup figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self._setup_static_elements()

        # Initialize drone markers and trail lines
        self.drone_markers = []
        self.drone_trails = []
        self.charge_indicators = []

        for i, drone in enumerate(self.drones):
            # Drone marker
            (marker,) = self.ax.plot(
                [], [], "o", color=self.colors[i], markersize=8, zorder=5
            )
            self.drone_markers.append(marker)

            # Trail line
            (trail,) = self.ax.plot(
                [], [], "-", color=self.colors[i], alpha=0.6, linewidth=2, zorder=2
            )
            self.drone_trails.append(trail)

            # Charge indicator text
            charge_text = self.ax.text(
                0, 0, "", fontsize=8, ha="center", va="bottom", zorder=6
            )
            self.charge_indicators.append(charge_text)

    def _setup_static_elements(self):
        """Setup static elements like obstacles, charging stations, and destinations."""
        # Plot obstacles
        if self.blocks is not None:
            for block in self.blocks:
                polygon = Polygon(
                    block,
                    facecolor="black",
                    alpha=0.8,
                    hatch="////",
                    edgecolor="black",
                    linewidth=1,
                )
                self.ax.add_patch(polygon)

        # Plot charging stations
        for i, station in enumerate(self.charging_stations):
            self.ax.plot(
                station[0],
                station[1],
                "^",
                color="blue",
                markersize=12,
                zorder=4,
            )
            self.ax.annotate(
                f"F{i+1}",
                (station[0], station[1]),
                xytext=(5, 5),
                textcoords="offset points",
                zorder=4,
                fontweight="bold",
            )

        # Plot destinations
        for drone_idx, drone in enumerate(self.drones):
            (init_x, init_y), (dest_x, dest_y), charge = drone
            self.ax.plot(dest_x, dest_y, "*", color="red", markersize=12, zorder=4)
            self.ax.text(
                dest_x,
                dest_y,
                f"D{drone_idx+1}",
                ha="center",
                va="bottom",
                zorder=4,
                fontweight="bold",
            )

        # Plot initial UGV location if provided
        if self.ugv_init_loc is not None:
            self.ax.scatter(
                self.ugv_init_loc[0, 0],
                self.ugv_init_loc[0, 1],
                color="orange",
                marker="s",
                s=100,
                zorder=4,
            )
            self.ax.text(
                self.ugv_init_loc[0, 0],
                self.ugv_init_loc[0, 1],
                "Initial UGV",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Setup axis properties
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X", fontsize=12)
        self.ax.set_ylabel("Y", fontsize=12)
        self.ax.set_title("Animated Drone Routes", fontsize=14, fontweight="bold")

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="blue",
                label="Charging Stations",
                markerfacecolor="blue",
                markersize=10,
                linestyle="None",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="gray",
                label="Drones",
                markerfacecolor="gray",
                markersize=8,
                linestyle="None",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="red",
                label="Destinations",
                markerfacecolor="red",
                markersize=12,
                linestyle="None",
            ),
            patches.Patch(
                facecolor="black", alpha=0.8, hatch="////", label="Obstacles"
            ),
        ]
        self.ax.legend(handles=legend_elements, loc="upper right")

    def _get_full_path_with_obstacles(self, start_point, end_point):
        """Get full path including obstacle avoidance waypoints."""
        if self.blocks is None:
            return [start_point, end_point]

        additional_points = []
        for block in self.blocks:
            obstacle = Obstacle(block)
            extra_points = obstacle.find_additional_length(
                [start_point, end_point], Path_=True
            )
            if extra_points:
                additional_points.extend(extra_points)

        if additional_points:
            return [start_point] + additional_points + [end_point]
        else:
            return [start_point, end_point]

    def _interpolate_path(self, waypoints, num_points=50):
        """Interpolate smooth path between waypoints."""
        if len(waypoints) < 2:
            return waypoints

        full_path = []
        for i in range(len(waypoints) - 1):
            start = np.array(waypoints[i])
            end = np.array(waypoints[i + 1])

            # Create interpolated points between waypoints
            t = np.linspace(0, 1, num_points)
            segment_points = [(1 - t_val) * start + t_val * end for t_val in t]
            full_path.extend(segment_points[:-1])  # Exclude last point to avoid duplicates

        full_path.append(np.array(waypoints[-1]))  # Add final point
        return full_path

    def _calculate_drone_paths(self):
        """Calculate complete paths for all drones including charging stations."""
        drone_paths = []

        for drone_idx, (drone, route) in enumerate(zip(self.drones, self.routes)):
            (init_x, init_y), (dest_x, dest_y), charge = drone
            current_pos = np.array([init_x, init_y])
            full_path = [current_pos]

            # Add path through charging stations
            for next_station_idx in route:
                if next_station_idx == len(self.charging_stations):
                    # Final destination
                    next_pos = np.array([dest_x, dest_y])
                else:
                    # Charging station
                    next_pos = self.charging_stations[next_station_idx]

                # Get waypoints including obstacle avoidance
                waypoints = self._get_full_path_with_obstacles(current_pos, next_pos)
                segment_path = self._interpolate_path(waypoints, num_points=30)

                # Add segment to full path (excluding first point to avoid duplicates)
                full_path.extend(segment_path[1:])
                current_pos = next_pos

            drone_paths.append(full_path)

        return drone_paths

    def _calculate_animation_frames(self):
        """Calculate the precise number of frames needed for the animation."""
        # Find when the last drone finishes its journey
        max_frames_for_completion = 0
        for path in self.drone_paths:
            frames_for_this_drone = int(len(path) * self.fps / self.animation_speed)
            max_frames_for_completion = max(max_frames_for_completion, frames_for_this_drone)
        
        # Add small buffer (2-3 frames) to show final state briefly
        return max_frames_for_completion + 3

    def _animate(self, frame):
        """Animation function called for each frame."""
        all_drones_finished = True
        
        for drone_idx, (drone_path, marker, trail, charge_text) in enumerate(
            zip(self.drone_paths, self.drone_markers, self.drone_trails, self.charge_indicators)
        ):
            # Calculate current position index
            path_progress = (frame * self.animation_speed) / self.fps
            path_index = int(path_progress * len(drone_path))

            if path_index < len(drone_path):
                # Drone is still moving
                all_drones_finished = False
                current_pos = drone_path[path_index]
                marker.set_data([current_pos[0]], [current_pos[1]])

                # Update trail
                trail_points = drone_path[:path_index + 1]
                if len(trail_points) > 1:
                    x_coords = [p[0] for p in trail_points]
                    y_coords = [p[1] for p in trail_points]
                    trail.set_data(x_coords, y_coords)

                # Update charge indicator (simplified - shows drone number)
                charge_text.set_position((current_pos[0], current_pos[1] + 2))
                charge_text.set_text(f"V{drone_idx + 1}")
            else:
                # Drone has reached destination
                if drone_path:
                    final_pos = drone_path[-1]
                    marker.set_data([final_pos[0]], [final_pos[1]])
                    # charge_text.set_position((final_pos[0], final_pos[1] + 2))
                    # charge_text.set_text(f"V{drone_idx + 1} ✓")

        # Store whether all drones are finished for potential early termination
        self._all_finished = all_drones_finished
        
        return self.drone_markers + self.drone_trails + self.charge_indicators

    def animate(self, save_path=None, interval=None):
        """
        Start the animation.

        Parameters:
        -----------
        save_path : str, optional
            If provided, save animation as GIF or MP4 file
        interval : int, optional
            Interval between frames in milliseconds (overrides fps setting)
        """
        if interval is None:
            interval = 1000 // self.fps

        # Calculate total frames needed based on when all drones finish
        # Use the more precise calculation, but still limit to prevent crashes
        total_frames = min(self._frames_needed, 500)

        anim = animation.FuncAnimation(
            self.fig,
            self._animate,
            frames=total_frames,
            interval=interval,
            blit=False,
            repeat=True,
        )

        if save_path:
            try:
                print(f"Saving animation to {save_path}...")
                
                # Determine file format and appropriate writer
                if save_path.lower().endswith('.gif'):
                    # For GIF files, try multiple writers in order of preference
                    writers_to_try = ['pillow', 'imagemagick', 'ffmpeg']
                    writer_used = None
                    
                    for writer_name in writers_to_try:
                        try:
                            if writer_name == 'pillow':
                                writer = animation.PillowWriter(fps=min(self.fps, 10))  # Limit fps for GIF
                            elif writer_name == 'imagemagick':
                                writer = animation.ImageMagickWriter(fps=min(self.fps, 10))
                            elif writer_name == 'ffmpeg':
                                writer = animation.FFMpegWriter(fps=min(self.fps, 10))
                            
                            anim.save(save_path, writer=writer)
                            writer_used = writer_name
                            break
                            
                        except Exception as e:
                            print(f"Writer {writer_name} failed: {e}")
                            continue
                    
                    if writer_used:
                        print(f"Animation saved using {writer_used} writer!")
                    else:
                        print("Failed to save GIF - no suitable writer found")
                        print("Try installing: pip install pillow")
                        
                elif save_path.lower().endswith(('.mp4', '.avi', '.mov')):
                    # For video files
                    try:
                        writer = animation.FFMpegWriter(fps=self.fps, bitrate=1800)
                        anim.save(save_path, writer=writer)
                        print("Video saved using FFMpeg!")
                    except Exception as e:
                        print(f"FFMpeg writer failed: {e}")
                        print("Try installing ffmpeg or use .gif format instead")
                        
                else:
                    print("Unsupported file format. Use .gif, .mp4, .avi, or .mov")
                    
            except Exception as e:
                print(f"Animation saving failed with error: {e}")
                print("Displaying animation without saving...")

        plt.show()
        return anim


def animate_drone_routes(
    drones,
    charging_stations,
    blocks,
    routes,
    fcr,
    ugv_factor,
    ugv_init_loc=None,
    animation_speed=1.0,
    fps=30,
    save_path=None,
):
    """
    Convenience function to create and run drone route animation.

    Parameters:
    -----------
    drones : list of tuples
        Each tuple contains ((init_x, init_y), (dest_x, dest_y), charge_level)
    charging_stations : numpy.ndarray
        Array of shape (m, 2) containing charging station coordinates
    blocks : list of lists
        Each inner list contains tuples of vertex coordinates for obstacles
    routes : list of lists
        Each inner list contains indices of charging stations to visit
    fcr : float
        Full charge range
    ugv_factor : float
        UGV cost factor
    ugv_init_loc : array, optional
        Initial UGV location
    animation_speed : float, optional
        Speed multiplier for animation (default: 1.0)
    fps : int, optional
        Frames per second (default: 30)
    save_path : str, optional
        If provided, save animation as GIF file

    Returns:
    --------
    animation : matplotlib.animation.FuncAnimation
        The animation object
    """
    animator = DroneAnimator(
        drones=drones,
        charging_stations=charging_stations,
        blocks=blocks,
        routes=routes,
        fcr=fcr,
        ugv_factor=ugv_factor,
        ugv_init_loc=ugv_init_loc,
        animation_speed=animation_speed,
        fps=fps,
    )

    return animator.animate(save_path=save_path)


# Example usage and demonstration
if __name__ == "__main__":
    """
    Example animation demonstrating drone route optimization visualization.
    This example uses the same data structure as in main.ipynb.
    """
    from utils import create_block
    
    # Example drone configuration
    example_drones = [
        ((10.0, 5.0), (45.0, 50.0), 0.7),  # Long distance, high charge
        ((3.0, 40.0), (50.0, 10.0), 0.5),  # Long distance, medium charge
        ((20.0, 15.0), (35.0, 35.0), 0.6),  # Moderate distance, medium charge
        ((5.0, 30.0), (25.0, 5.0), 0.4),   # Moderate distance, low charge
        ((40.0, 45.0), (10.0, 10.0), 0.8), # Long distance, high charge
    ]
    
    # Example charging stations (optimized positions)
    example_stations = np.array([
        [15.0, 20.0],
        [30.0, 25.0],
        [25.0, 40.0],
    ])
    
    # Example obstacles using utils.create_block()
    hexagon = create_block("hexagon", center=(30.0, 30.0), length=3.0, distortion="none")
    square = create_block("square", center=(15.0, 25.0), length=3.5, distortion="rotated")
    triangle = create_block("triangle", center=(30.0, 10.0), length=2.0, distortion="skewed")
    example_blocks = [hexagon, square, triangle]
    
    # Example routes (each drone's charging station sequence)
    example_routes = [
        [1, 2, 3],  # Drone 0: visits stations 1, 2, then destination
        [0, 1, 3],  # Drone 1: visits stations 0, 1, then destination  
        [2, 3],     # Drone 2: visits station 2, then destination
        [0, 3],     # Drone 3: visits station 0, then destination
        [1, 0, 3],  # Drone 4: visits stations 1, 0, then destination
    ]
    
    # Animation parameters
    fcr = 25.0  # Full Charge Range
    ugv_factor = 0.0  # UGV cost factor
    
    print("Starting drone route animation example...")
    print("- 5 drones with different charge levels")
    print("- 3 charging stations") 
    print("- 3 obstacles (hexagon, square, triangle)")
    print("- Animation speed: 1.5x")
    print("- Close the plot window when done viewing")
    
    # Create and run the animation
    try:
        anim = animate_drone_routes(
            drones=example_drones,
            charging_stations=example_stations,
            blocks=example_blocks,
            routes=example_routes,
            fcr=fcr,
            ugv_factor=ugv_factor,
            animation_speed=1.5,  # 1.5x speed for demonstration
            fps=24,               # 24fps animation
            save_path="example_animation.gif"  # Uncomment to save as GIF
        )
        
        print("Animation completed successfully!")
        
    except Exception as e:
        print(f"Animation failed with error: {e}")
        print("Make sure all required dependencies are installed:")
        print("- matplotlib")
        print("- numpy") 
        print("- shapely (for obstacle handling)")
        
    print("\nTo use this animation in your own code:")
    print("1. Import: from animator import animate_drone_routes")
    print("2. Use same parameters as plot_drone_routes() from viz.py")
    print("3. Add animation_speed and fps parameters for control")
    print("4. Optionally specify save_path to export as GIF")
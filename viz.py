import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches
from ObstacleAvoidance import Obstacle


def plot_drone_routes(
    drones,
    charging_stations,
    blocks,
    routes,
    fcr,
    ugv_factor,
    ugv_init_loc=None,
    save_=False,
):
    """
    Plot drone routes with obstacles, charging stations, and destinations.

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
    """

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Plot blocks (obstacles) with hash pattern
    if blocks is not None:
        for block in blocks:
            polygon = Polygon(
                block,
                facecolor="black",
                alpha=0.8,
                hatch="////",
                edgecolor="black",
                linewidth=1,
                label="Obs.",
            )
            ax.add_patch(polygon)

    # Colors for different drone routes - using a darker colormap
    np.random.seed(242)
    colors = np.random.rand(len(drones), 3)
    options = [
        "-",
        "--",
        "-.",
        ":",
    ]

    def draw_route_segment(start_point, end_point, color, style, blocks, range_):
        """Helper function to draw route segment including additional points from obstacles"""
        additional_points = []
        # Check each block for additional points
        if blocks is not None:
            for block in blocks:
                obstacle = Obstacle(block)
                extra_points = obstacle.find_additional_length(
                    [start_point, end_point], Path_=True
                )
                if extra_points:
                    additional_points.extend(extra_points)

        # If there are additional points, draw the route through them
        if additional_points:
            # Create full path including start, additional points, and end
            full_path = [start_point] + additional_points + [end_point]
            # Draw each segment
            for i in range(len(full_path) - 1):
                ax.plot(
                    [full_path[i][0], full_path[i + 1][0]],
                    [full_path[i][1], full_path[i + 1][1]],
                    "--",
                    color=color,
                    alpha=0.9,
                    zorder=1,
                )
        else:
            # Draw direct path if no additional points
            ax.plot(
                [start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                "--",
                color=color,
                alpha=0.9,
                zorder=1,
            )
        dist = np.linalg.norm(end_point - start_point)
        if dist > range_:
            text_color = "red"
            mid_x = (start_point[0] + end_point[0]) / 2
            mid_y = (start_point[1] + end_point[1]) / 2
            ax.text(
                mid_x,
                mid_y,
                f"{dist / range_:.2f}",
                color=text_color,
                ha="left",
                va="center",
            )
        else:
            text_color = "green"

    # Plot drone routes and annotations first (to be in background)
    for drone_idx, (drone, route, color) in enumerate(zip(drones, routes, colors)):
        (init_x, init_y), (dest_x, dest_y), charge = drone

        # Plot route lines first (lowest z-order)
        current_pos = np.array([init_x, init_y])
        range_ = charge * fcr
        style = options[np.random.randint(0, 4)]
        for next_station in route:
            if next_station == len(charging_stations):
                next_pos = np.array([dest_x, dest_y])
                # Draw final segment to destination
                draw_route_segment(current_pos, next_pos, color, style, blocks, range_)
                break
            else:
                next_pos = charging_stations[next_station]
                # Draw segment to next charging station
                draw_route_segment(current_pos, next_pos, color, style, blocks, range_)
            current_pos = next_pos
            range_ = fcr
        # Plot initial position and destination (middle z-order)
        ax.plot(init_x, init_y, "o", color="black", zorder=2, label="UAV")
        ax.plot(dest_x, dest_y, "*", color="red", zorder=2, label="Dest.")

        # Add annotations
        ax.text(init_x, init_y, f"V{drone_idx+1} ({charge})", ha="center", va="bottom")
        ax.text(dest_x, dest_y, f"D{drone_idx+1}", ha="center", va="bottom", zorder=2)

    # Plot charging stations last (highest z-order)
    for i, station in enumerate(charging_stations):
        ax.plot(
            station[0],
            station[1],
            "^",
            color="blue",
            markersize=10,
            zorder=3,
            label="UGV",
        )
        ax.annotate(
            f"F{i+1}",
            (station[0], station[1]),
            xytext=(5, 5),
            textcoords="offset points",
            zorder=3,
        )
    if not ugv_init_loc is None:
        plt.scatter(ugv_init_loc[0, 0], ugv_init_loc[0, 1], color="orange", marker="2")
        plt.text(
            ugv_init_loc[0, 0],
            ugv_init_loc[0, 1],
            "Initial UGV location",
            ha="center",
            va="bottom",
        )
    # Set equal aspect ratio and add grid
    ax.set_aspect("equal")
    # ax.grid(True, linestyle='--', alpha=0.3)

    # Add title and labels
    # plt.title('Drone Routes with Charging Stations and Obstacles')
    plt.xlabel("X")
    plt.ylabel("Y")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Add legend with new labels
    # legend_elements = [
    #     patches.Patch(facecolor='black', alpha=0.3, hatch='////', label='Obs.'),
    #     plt.Line2D([0], [0], marker='^', color='blue', label='UGVs',
    #               markerfacecolor='blue', markersize=10, linestyle='None'),
    #     plt.Line2D([0], [0], marker='o', color='black', label='UAVs',
    #               markerfacecolor='black', markersize=10, linestyle='None'),
    #     plt.Line2D([0], [0], marker='*', color='gray', label='Dest.',
    #               markerfacecolor='gray', markersize=12, linestyle='None')
    # ]
    ax.legend(by_label.values(), by_label.keys(), loc="best")
    if save_:
        plt.savefig(
            f"new sim res for M={len(charging_stations)} F.C.R. = {fcr} alpha = {ugv_factor}.pdf"
        )
    plt.show()

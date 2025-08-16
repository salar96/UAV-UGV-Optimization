import numpy as np
from collections import defaultdict
from shapely.geometry import LineString, Polygon
from heapq import heappop, heappush
from utils import *


# Utility for Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class Obstacle:
    def __init__(self, obstacle):
        self.polygon = Polygon(obstacle)
        self.vertices = list(self.polygon.exterior.coords)[
            :-1
        ]  # Exclude duplicate closing point

    def _build_graph(self, start, finish):
        nodes = [start, finish] + self.vertices
        edges = []

        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:
                    continue
                segment = LineString([node1, node2])
                if segment.touches(self.polygon):
                    edges.append((node1, node2, euclidean_distance(node1, node2)))

        graph = defaultdict(list)
        for node1, node2, weight in edges:
            graph[node1].append((weight, node2))
            graph[node2].append((weight, node1))

        return graph

    def _shortest_path(self, graph, source, target):
        pq = [(0, source)]  # (distance, current_node)
        visited = set()
        distances = {source: 0}
        predecessors = {source: None}  # To reconstruct the path

        while pq:
            dist, current = heappop(pq)
            if current in visited:
                continue
            visited.add(current)

            if current == target:
                # Reconstruct the path
                path = []
                while current is not None:
                    path.append(current)
                    current = predecessors[current]
                return path[::-1], dist

            for weight, neighbor in graph[current]:
                if neighbor not in visited:
                    new_dist = dist + weight
                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        predecessors[neighbor] = current
                        heappush(pq, (new_dist, neighbor))

        return [], float("inf")

    def find_additional_length(self, points, Path_=False):
        start = tuple(points[0])
        finish = tuple(points[1])
        line = LineString([start, finish])

        # Check if direct path intersects the polygon
        if not line.intersects(self.polygon):
            return 0.0

        # Compute intersection points
        # intersection_points = [tuple(pt) for pt in line.intersection(self.polygon).coords]

        # Build the graph and compute shortest path
        graph = self._build_graph(start, finish)
        path, length = self._shortest_path(graph, start, finish)

        # Remove start and finish points from the path
        path = path[1:-1]

        additional_length = length - euclidean_distance(start, finish)

        if not Path_:
            return additional_length
        else:
            return path


def add_block_dist(obstacle):
    return lambda u, v: obstacle.find_additional_length([u, v])

import pandas as pd
import numpy as np

# Function to compute the minimum distance from a point to a line segment
def point_to_segment_distance(p, a, b):
    """
    Compute the minimum distance from point `p` to the line segment defined by points `a` and `b`.
    """
    ab = b - a  # Vector from a to b
    ap = p - a  # Vector from a to p
    t = np.dot(ap, ab) / np.dot(ab, ab)  # Projection factor
    t = np.clip(t, 0, 1)  # Clamp t to [0,1] to stay within segment
    closest_point = a + t * ab  # Compute the closest point on the segment
    return np.linalg.norm(p - closest_point)  # Compute the Euclidean distance

def main():
    # File paths
    waypoints_file = "/home/plague/waypoints.csv"
    flight_log_file = "/home/plague/flight_log_20250324_172104.csv"

    # Load data
    waypoints_df = pd.read_csv(waypoints_file)
    flight_log_df = pd.read_csv(flight_log_file)

    # Extract coordinates (assuming columns are named 'x', 'y')
    waypoints = waypoints_df[['x', 'y']].values
    flight_points = flight_log_df[['x', 'y']].values

    # Compute the minimum distance of each flight point from any waypoint segment
    min_distances = []
    for p in flight_points:
        min_distance = min(point_to_segment_distance(p, waypoints[i], waypoints[i+1]) 
                        for i in range(len(waypoints) - 1))
        min_distances.append(min_distance)

    # Compute the average minimum distance (error)
    avg_min_distance = np.mean(min_distances)

    print(f"Average deviation (error) between flight log and waypoints: {avg_min_distance:.2f} units")

if __name__ == '__main__':
    main()
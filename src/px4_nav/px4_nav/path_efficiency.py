import pandas as pd
import numpy as np
from math import sqrt

def path_length(points) -> float:
    if len(points) <= 0:
        return 0
    res = 0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        res += sqrt(dx*dx + dy*dy)
    
    return res

def main(args=None) -> None:
    waypoints_file = "/home/plague/waypoints.csv"
    flight_log_file = "/home/plague/flight_log_20250324_172104.csv"

    waypoints_df = pd.read_csv(waypoints_file)
    flight_log_df = pd.read_csv(flight_log_file)

    waypoints = waypoints_df[['x', 'y']].values
    flight_points = flight_log_df[['x', 'y']].values

    optimal_path = path_length(waypoints)
    actual_path = path_length(flight_points)

    if actual_path == 0:
        print("Error: Actual path length is zero, cannot compute efficiency.")
        return

    path_efficiency = 100 * optimal_path/actual_path

    print(f"Path efficiency between optimal path and actual path: {path_efficiency:.2f}%")

if __name__ == '__main__':
    main()
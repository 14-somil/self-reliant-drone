import pandas as pd
import matplotlib.pyplot as plt

def plot_flight_paths(file1, file2, label1='Results', label2='Waypoints'):
    """
    Plot and compare two flight paths on the same graph.
    
    Parameters:
    file1 (str): Path to the first CSV file
    file2 (str): Path to the second CSV file
    label1 (str): Label for the first flight path
    label2 (str): Label for the second flight path
    """
    # Read CSV data from files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot flight path 1
    ax.plot(df1['x'], df1['y'], 'b-', linewidth=2, alpha=0.7, label=f'{label1} Path')
    ax.plot(df1['x'].iloc[0], df1['y'].iloc[0], 'bo', markersize=8, label=f'{label1} Start')
    ax.plot(df1['x'].iloc[-1], df1['y'].iloc[-1], 'bX', markersize=10, label=f'{label1} End')
    
    # Plot flight path 2
    ax.plot(df2['x'], df2['y'], 'r-', linewidth=2, alpha=0.7, label=f'{label2} Path')
    ax.plot(df2['x'].iloc[0], df2['y'].iloc[0], 'ro', markersize=8, label=f'{label2} Start')
    ax.plot(df2['x'].iloc[-1], df2['y'].iloc[-1], 'rX', markersize=10, label=f'{label2} End')
    
    # Add labels and title
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Comparison of Flight Paths (X-Y Coordinates)', fontsize=14)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=10)
    
    # Make sure the aspect ratio is equal so the paths aren't distorted
    ax.set_aspect('equal')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    file1 = '/home/plague/flight_log_20250324_172104.csv'
    file2 = '/home/plague/waypoints.csv'  # Replace with your second file
    
    plot_flight_paths(file1, file2, label1='Results', label2='Waypoints')
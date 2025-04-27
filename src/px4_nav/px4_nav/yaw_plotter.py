import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def adjust_yaw_discontinuity(yaw_values):
    """Correct yaw discontinuities where jumps occur from 3.14 to -3.13 or vice versa."""
    for i in range(1, len(yaw_values)):
        if yaw_values[i] - yaw_values[i - 1] < -6.0:
            yaw_values[i] += 2 * np.pi
        elif yaw_values[i] - yaw_values[i - 1] > 6.0:
            yaw_values[i] -= 2 * np.pi
    return yaw_values

def plot_yaw(filename) -> None:
    """Extract yaw and timestamp from CSV and plot them."""
    flight_df = pd.read_csv(filename)
    
    # Extract timestamp and yaw values
    timestamp = flight_df['timestamp']
    yaw = flight_df['yaw']
    
    # Adjust for yaw discontinuities
    yaw = adjust_yaw_discontinuity(yaw.values)
    
    # Plot yaw over time
    plt.figure(figsize=(10, 5))
    plt.plot(timestamp, yaw, label='Yaw', color='b')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Yaw (radians)')
    plt.title('Yaw Over Time')
    plt.legend()
    plt.grid()
    plt.show()

def main(args=None) -> None:
    filename = '/home/plague/flight_log_20250325_094856.csv'
    plot_yaw(filename)

if __name__ == '__main__':
    main()

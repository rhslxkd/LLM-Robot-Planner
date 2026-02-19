import numpy as np
import json

def convert_path_to_3d():
    # Input 2D path as list of dictionaries
    path_2d = [
      {"x": 0.0, "y": 0.0},
      {"x": -0.2, "y": 0.5},
      {"x": -0.5, "y": 1.0},
      {"x": -1.0, "y": 1.5},
      {"x": -2.0, "y": 1.5},
      {"x": -2.5, "y": 1.5},
      {"x": -3.0, "y": 1.0},
      {"x": -3.5, "y": 0.5},
      {"x": -4.0, "y": 0.0},
      {"x": -4.5, "y": 0.0}
    ]

    # Convert to numpy array of shape (N, 2)
    path_np_2d = np.array([[p["x"], p["y"]] for p in path_2d])

    # Create 3D array of shape (N, 3) initialized with zeros
    path_np_3d = np.zeros((len(path_2d), 3))

    # Fill X and Y from 2D path
    # Assuming the local coordinate application:
    # If this is for a robot on ground, usually:
    # X_3d = X_2d
    # Y_3d = Y_2d
    # Z_3d = 0.0 (Ground)
    
    path_np_3d[:, 0] = path_np_2d[:, 0] # X
    path_np_3d[:, 1] = path_np_2d[:, 1] # Y
    path_np_3d[:, 2] = 0.0              # Z

    return path_np_3d

if __name__ == "__main__":
    path_3d = convert_path_to_3d()
    
    print("Original 2D Path (Preview):")
    print(convert_path_to_3d.__code__.co_consts[1]) # Just printing a placeholder or the input for verification isn't easy with consts, let's just print the output.
    
    print("\nConverted 3D Path (Numpy Array):\n")
    print(path_3d)
    
    print("\nFormatted List for Dial-MPC (Python list of lists):")
    print(path_3d.tolist())

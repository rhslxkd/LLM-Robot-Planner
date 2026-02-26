import numpy as np
import json
import os

def convert_path_to_3d(json_path=None):
    # Default path for automated coordinate transfer
    if json_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        json_path = os.path.join(project_root, "outputs", "last_judged_path.json")

    # Try to load from JSON file
    path_2d = []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                path_2d = json.load(f)
            print(f"✅ Loaded {len(path_2d)} coordinates from: {json_path}")
        except Exception as e:
            print(f"❌ Failed to load JSON from {json_path}: {e}")
    else:
        print(f"⚠️ JSON file not found at {json_path}. Using fallback sample.")
        # Fallback sample path
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

    if not path_2d:
        print("Empty path provided.")
        return np.array([])

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
    
    if path_3d.size > 0:
        print("\nConverted 3D Path (Numpy Array):\n")
        print(path_3d)
        
        print("\nFormatted List for Dial-MPC (Python list of lists):")
        print(path_3d.tolist())

import numpy as np

class CoordinateConverter:
    def __init__(self, pos, xyaxes, fovy=45):
        """
        Initializes the CoordinateConverter with camera extrinsics and intrinsics.

        Args:
            pos (list or np.array): Camera position [x, y, z].
            xyaxes (list or np.array): Camera orientation as 6 values [x_axis_x, x_axis_y, x_axis_z, y_axis_x, y_axis_y, y_axis_z].
            fovy (float): Field of view in y direction in degrees. Default is 45.
        """
        self.pos = np.array(pos)
        self.fovy = fovy
        
        # 1. Externals (Extrinsics)
        # Parse xyaxes to get local X and Y axes
        xaxis = np.array(xyaxes[:3])
        yaxis = np.array(xyaxes[3:])
        
        # Calculate Z axis using cross product
        zaxis = np.cross(xaxis, yaxis)
        
        # Normalize axes just in case (though usually normalized in MJX/MuJoCo)
        xaxis = xaxis / np.linalg.norm(xaxis)
        yaxis = yaxis / np.linalg.norm(yaxis)
        zaxis = zaxis / np.linalg.norm(zaxis)
        
        # Rotation Matrix R = [xaxis, yaxis, zaxis] (columns)
        self.R = np.vstack([xaxis, yaxis, zaxis]).T
        self.t = self.pos

    def get_intrinsics(self, width, height):
        """
        Calculates the camera intrinsic matrix K.

        Args:
            width (int): Image width in pixels.
            height (int): Image height in pixels.

        Returns:
            np.array: 3x3 Intrinsic Matrix K.
        """
        # Calculate focal length f
        # f = H / (2 * tan(fovy / 2))
        f = height / (2 * np.tan(np.radians(self.fovy) / 2))
        
        # Principle point (assume center of image)
        cx = width / 2
        cy = height / 2
        
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        return K

    def pixel_to_world(self, u, v, width, height):
        """
        Converts 2D pixel coordinates (u, v) to 3D world coordinates on the Z=0 plane.

        Args:
            u (float): Pixel x-coordinate.
            v (float): Pixel y-coordinate.
            width (int): Image width.
            height (int): Image height.

        Returns:
            np.array: 3D world coordinates [X, Y, Z] (where Z is approx 0).
        """
        K = self.get_intrinsics(width, height)
        K_inv = np.linalg.inv(K)

        # Pixel coordinate in homogeneous form
        P_pixel = np.array([u, v, 1.0])

        # 1. Back-project to camera coordinate system (Direction vector d_cam)
        d_cam = K_inv @ P_pixel

        # 2. Transform direction to world coordinate system (Direction vector w)
        w = self.R @ d_cam

        # 3. Intersect with Z=0 plane
        # Ray equation: P = C + s * w
        # We need P_z = 0 => C_z + s * w_z = 0 => s = -C_z / w_z
        
        C = self.t
        
        if np.isclose(w[2], 0):
            print("Warning: Ray is parallel to the ground (Z=0 plane). No intersection.")
            return None

        s = -C[2] / w[2]
        
        P_world = C + s * w
        
        return P_world

    def world_to_pixel(self, world_pos, width, height):
        """
        Converts 3D world coordinates [X, Y, Z] to 2D pixel coordinates (u, v).

        Args:
            world_pos (list or np.array): 3D world coordinates.
            width (int): Image width.
            height (int): Image height.

        Returns:
            tuple: (u, v) pixel coordinates.
        """
        world_pos = np.array(world_pos)
        K = self.get_intrinsics(width, height)
        
        # Transform world to camera coordinates
        # P_cam = R^T * (P_world - t)
        P_cam = self.R.T @ (world_pos - self.t)
        
        # Project to pixel coordinates
        # P_pixel = K * P_cam
        
        # Note: We do not check for Z > 0 because MuJoCo cameras (OpenGL) look down -Z.
        # We assume the projection mathematics holds (dividing by Z normalizes the ray).
        
        projected = K @ P_cam
        
        if np.isclose(projected[2], 0):
             return None

        u = projected[0] / projected[2]
        v = projected[1] / projected[2]
        
        return (u, v)

if __name__ == "__main__":
    # Example usage based on provided parameters
    
    # Given parameters
    camera_pos = [0.846, -1.465, 0.916]
    camera_xyaxes = [0.866, 0.500, 0.000, -0.171, 0.296, 0.940]
    
    # Initialize converter
    converter = CoordinateConverter(pos=camera_pos, xyaxes=camera_xyaxes, fovy=45)
    
    # Assumptions for image size (example)
    img_width = 640
    img_height = 480
    
    # Test point: Center of the image
    u_test = img_width / 2
    v_test = img_height / 2
    
    world_pos = converter.pixel_to_world(u_test, v_test, img_width, img_height)
    
    print(f"Camera Pos: {camera_pos}")
    print(f"Test Pixel: ({u_test}, {v_test})")
    print("Calculated World Position (Z=0 plane):")
    print(world_pos)

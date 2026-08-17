"""
occupancy_grid.py -- builds a world-aligned occupancy grid from static
obstacle geometry, using the SAME pixel/world axis convention as
oracle_gen.py (+x forward = +u right, +y left = -v up, 150 px/m), so A* /
skeleton-planner waypoints line up with the VLM courtroom's own coordinate
system and can be fed to DIAL-MPC unchanged.

Important difference from oracle_gen.py: the oracle PNG is only 1263x1080 px
(~8.4m x 7.2m, robot-centered), because it just needs to be *visible* to a
VLM. Our planning grid needs to cover the FULL scenario extent (Scene D's
corridor is ~7m of forward progress, Scene E goes to x=6.5m), which the
default oracle frame does not fully contain. So this module takes an
explicit world-frame bounding box (`GridSpec`) instead of hardcoding the
oracle image size -- the pixel *origin* still matches oracle_gen.py's
robot-at-image-center convention when the default bounds are used, but the
grid can be made larger.

Calibration baseline (must stay consistent with oracle_gen.py / courtroom.py):
  - 150 px/m
  - +x(forward) = +u(right), +y(left) = -v(up)
  - default bounds reproduce oracle_gen.py's 1263x1080 frame exactly, with
    the robot origin at the image center (631.5, 540)
"""
from dataclasses import dataclass
import numpy as np

PPM_DEFAULT = 150.0


@dataclass
class GridSpec:
    """World-frame bounding box + resolution for the planning grid."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    ppm: float = PPM_DEFAULT

    @classmethod
    def matching_oracle_image(cls, ppm=PPM_DEFAULT, img_w=1263, img_h=1080):
        """Bounds that exactly reproduce oracle_gen.py's 1263x1080 frame."""
        return cls(x_min=-(img_w / 2) / ppm, x_max=(img_w / 2) / ppm,
                    y_min=-(img_h / 2) / ppm, y_max=(img_h / 2) / ppm, ppm=ppm)

    @property
    def img_w(self):
        return int(round((self.x_max - self.x_min) * self.ppm))

    @property
    def img_h(self):
        return int(round((self.y_max - self.y_min) * self.ppm))

    @property
    def robot_px(self):
        """Pixel location of world origin (0, 0)."""
        return (-self.x_min * self.ppm, self.y_max * self.ppm)

    def world_to_pixel(self, wx, wy):
        rpx, rpy = self.robot_px
        return rpx + wx * self.ppm, rpy - wy * self.ppm

    def pixel_to_world(self, u, v):
        rpx, rpy = self.robot_px
        return (u - rpx) / self.ppm, (rpy - v) / self.ppm


def obstacles_from_mjmodel(model):
    """
    Mirrors oracle_gen.py's _report_obstacles(): static bodies (no joints,
    parent = worldbody) are treated as obstacles. Returns a list of dicts:
        {"name": str, "pos": (wx, wy), "kind": "box"|"cylinder"|"sphere",
         "size": tuple}  # size follows MuJoCo geom_size semantics
    Run this only where `mujoco` + the real scene XML are available (i.e. on
    the lab server) -- not required for the grid/A* unit tests below.
    """
    import mujoco  # local import: keep this module importable without mujoco

    obstacles = []
    for bid in range(1, model.nbody):
        if model.body_jntnum[bid] == 0 and model.body_parentid[bid] == 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"
            wx, wy, _ = model.body_pos[bid]
            geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
            if not geom_ids:
                continue
            gid = geom_ids[0]
            gtype = model.geom_type[gid]
            size = tuple(float(s) for s in model.geom_size[gid])
            kind = {
                mujoco.mjtGeom.mjGEOM_BOX: "box",
                mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
                mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
            }.get(gtype, "box")
            obstacles.append({"name": name, "pos": (float(wx), float(wy)), "kind": kind, "size": size})
    return obstacles


def _rasterize_thick_line(grid, u0, v0, u1, v1, thickness_px=2):
    """Draw a wall segment (already in pixel coords) into `grid`."""
    h, w = grid.shape
    length = max(int(round(np.hypot(u1 - u0, v1 - v0))), 1)
    n = length * 2
    us = np.linspace(u0, u1, n)
    vs = np.linspace(v0, v1, n)
    half = thickness_px / 2.0
    offsets = np.arange(-half, half + 1) if half > 0 else [0]
    for du in offsets:
        for dv in offsets:
            ui = np.clip((us + du).round().astype(int), 0, w - 1)
            vi = np.clip((vs + dv).round().astype(int), 0, h - 1)
            grid[vi, ui] = True


def build_occupancy_grid(spec: GridSpec, obstacles, walls=None, wall_thickness_px=2):
    """
    Rasterizes `obstacles` (from obstacles_from_mjmodel, or a hand-built list
    with the same schema for synthetic testing) into a boolean grid at pixel
    resolution over the world bounds in `spec`. True = occupied.

    `walls` (optional): list of ((x0, y0), (x1, y1)) WORLD-frame line
    segments, e.g. Scene D's corridor walls or Scene E's walled room.
    """
    grid = np.zeros((spec.img_h, spec.img_w), dtype=bool)
    yy, xx = np.mgrid[0:spec.img_h, 0:spec.img_w]

    for obs in obstacles:
        wx, wy = obs["pos"]
        cu, cv = spec.world_to_pixel(wx, wy)
        kind = obs["kind"]
        size = obs["size"]
        if kind in ("sphere", "cylinder"):
            r_px = size[0] * spec.ppm
            mask = (xx - cu) ** 2 + (yy - cv) ** 2 <= r_px ** 2
        else:  # box: size = (half_x, half_y, half_z)
            hx_px = size[0] * spec.ppm
            hy_px = size[1] * spec.ppm
            mask = (np.abs(xx - cu) <= hx_px) & (np.abs(yy - cv) <= hy_px)
        grid |= mask

    if walls:
        for (x0, y0), (x1, y1) in walls:
            u0, v0 = spec.world_to_pixel(x0, y0)
            u1, v1 = spec.world_to_pixel(x1, y1)
            _rasterize_thick_line(grid, u0, v0, u1, v1, wall_thickness_px)

    return grid


def inflate_grid(grid, clearance_m, ppm=PPM_DEFAULT):
    """
    Grow the occupied region by `clearance_m` meters using a Euclidean
    distance transform (cheap and exact-circular, unlike binary_dilation
    with a large disk structuring element, which is slow/memory-heavy on a
    grid this size). Matches the courtroom's own effective clearance radius
    (0.8 m; see ROBOT_PHYSICAL_CONSTRAINTS / Table 1 of the paper -- any gap
    narrower than 2*0.8=1.6 m becomes fully blocked here, the same threshold
    the VLM agents are told to enforce).
    """
    from scipy.ndimage import distance_transform_edt
    r_px = clearance_m * ppm
    if r_px <= 0 or not grid.any():
        return grid.copy()
    dist_to_obstacle = distance_transform_edt(~grid)
    return grid | (dist_to_obstacle <= r_px)


if __name__ == "__main__":
    # Self-test with synthetic obstacles approximating the paper's Scene A
    # (single obstacle, Table 3: "generate 10 waypoints over ~5m of forward
    # progress") and Scene D (narrow corridor, 1.62m width, ~7m of forward
    # progress) scenario text. Real runs must use obstacles_from_mjmodel()
    # against the actual oracle_scene_{A,D,E}.xml on the lab server.

    print("[Scene A-like] single obstacle 2m ahead, 0.3m radius, goal at 5m")
    spec_a = GridSpec(x_min=-1.0, x_max=6.0, y_min=-3.0, y_max=3.0)
    obs_a = [{"name": "obs1", "pos": (2.0, 0.0), "kind": "cylinder", "size": (0.3, 0.5, 0.0)}]
    grid_a = build_occupancy_grid(spec_a, obs_a)
    inflated_a = inflate_grid(grid_a, clearance_m=0.8, ppm=spec_a.ppm)
    print(f"  grid shape: {grid_a.shape}  occupied px (raw): {grid_a.sum()}  (inflated): {inflated_a.sum()}")
    su, sv = spec_a.world_to_pixel(0.0, 0.0)
    gu, gv = spec_a.world_to_pixel(5.0, 0.0)
    print(f"  start px=({su:.0f},{sv:.0f}) blocked={inflated_a[int(sv), int(su)]}")
    print(f"  goal  px=({gu:.0f},{gv:.0f}) blocked={inflated_a[int(gv), int(gu)]}")

    print("\n[Scene D-like, OK width] corridor half-width 0.85m (full width 1.70m > 1.6m min)")
    spec_d = GridSpec(x_min=-1.0, x_max=8.0, y_min=-2.0, y_max=2.0)
    walls_d = [((-1.0, 0.85), (8.0, 0.85)), ((-1.0, -0.85), (8.0, -0.85))]
    grid_d = build_occupancy_grid(spec_d, [], walls=walls_d)
    inflated_d = inflate_grid(grid_d, clearance_m=0.8, ppm=spec_d.ppm)
    mu, mv = spec_d.world_to_pixel(3.5, 0.0)
    print(f"  corridor centerline blocked={inflated_d[int(mv), int(mu)]} (expect False -- passable)")

    print("\n[Scene D-like, too narrow] half-width 0.75m (full width 1.50m < 1.6m min)")
    walls_d2 = [((-1.0, 0.75), (8.0, 0.75)), ((-1.0, -0.75), (8.0, -0.75))]
    grid_d2 = build_occupancy_grid(spec_d, [], walls=walls_d2)
    inflated_d2 = inflate_grid(grid_d2, clearance_m=0.8, ppm=spec_d.ppm)
    print(f"  corridor centerline blocked={inflated_d2[int(mv), int(mu)]} (expect True -- impassable)")

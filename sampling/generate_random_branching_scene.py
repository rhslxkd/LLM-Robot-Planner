"""
generate_random_branching_scene.py
로봇이 좌/우 둘 다로 우회 가능하되, 의도적으로 살짝 비대칭인 장애물 씬 생성.
완벽 대칭이면 tie-break로 g_ratio 상관없이 늘 같은 쪽만 나올 위험이 있어서,
장애물을 y=0에서 소량 오프셋시켜 실제 비용 차이를 만든다.
사용법: python generate_random_branching_scene.py --seed 0 --goal-x 5.0
"""
import argparse
import os
import random

TEMPLATE = """<mujoco model="go2 oracle scene R - random branching obstacle (seed={seed})">
  <include file="mjx_go2_robot_only.xml"/>

  <statistic center="0 0 0.1" extent="0.8"/>

  <visual>
    <map zfar="100"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20" offwidth="1263" offheight="1080"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="wall_material" rgba="0.85 0.1 0.1 1" shininess="0.1" specular="0.5" roughness="0.3"/>
  </asset>

  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true" diffuse="1 1 1" ambient="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

    <!-- 의도적 비대칭 배치: y=0이 아니라 살짝 오프셋 -->
    <body name="obstacle" pos="{ox} {oy} 0.3">
      <geom name="obstacle_geom" type="{shape}" size="{size}" material="wall_material"/>
    </body>
  </worldbody>
</mujoco>
"""

def generate(seed, goal_x=5.0, min_clearance=0.9):
    rng = random.Random(seed)
    ox = rng.uniform(goal_x * 0.3, goal_x * 0.7)

    shape = rng.choice(["box", "cylinder"])
    if shape == "box":
        half_y = rng.uniform(0.2, min_clearance - 0.15)
        half_x = rng.uniform(0.2, 0.5)
        size = f"{half_x:.3f} {half_y:.3f} 0.3"
    else:
        radius = rng.uniform(0.2, min_clearance - 0.15)
        size = f"{radius:.3f} 0.3"

    # 비대칭 오프셋: y=0에서 0.1~0.3m 정도 치우치게 (좌우 우회 비용에 실제 차이 생성)
    oy = rng.choice([-1, 1]) * rng.uniform(0.1, 0.3)

    xml = TEMPLATE.format(seed=seed, ox=f"{ox:.3f}", oy=f"{oy:.3f}", shape=shape, size=size)
    return xml, {"ox": ox, "oy": oy, "shape": shape, "size": size, "goal_x": goal_x}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="dial_mpc/dial_mpc/models/unitree_go2/oracle_scene_R.xml")
    parser.add_argument("--goal-x", type=float, default=5.0)
    args = parser.parse_args()

    xml, meta = generate(args.seed, goal_x=args.goal_x)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(xml)
    print(f"saved: {args.out}")
    print(f"meta: {meta}")
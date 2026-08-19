"""
generate_corridor_width_test.py
Scene D 패턴을 폭만 바꿔서 여러 개 만들어 실제 최소 통과 폭을 실측하기 위한 스크립트.
"""
import os, json

TEMPLATE = """<mujoco model="go2 oracle scene D_w{width_tag} (corridor width test, open_width={open_width}m)">
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
    <body name="left_wall" pos="3.4 {half_gap} 0.3">
      <geom name="left_wall_geom" type="box" size="2.6 0.15 0.3" material="wall_material"/>
    </body>
    <body name="right_wall" pos="3.4 -{half_gap} 0.3">
      <geom name="right_wall_geom" type="box" size="2.6 0.15 0.3" material="wall_material"/>
    </body>
  </worldbody>
</mujoco>
"""

YAML_TEMPLATE = """# DIAL-MPC (corridor width test, auto-generated)
seed: 0
output_dir: data/{scene}
n_steps: 500
env_name: unitree_go2_walk
Nsample: 2048
Hsample: 16
Hnode: 4
Ndiffuse: 2
Ndiffuse_init: 10
temp_sample: 0.05
horizon_diffuse_factor: 0.9
traj_diffuse_factor: 0.5
update_method: mppi
dt: 0.02
timestep: 0.02
leg_control: torque
action_scale: 1.0
default_vx: 0.8
default_vy: 0.0
default_vyaw: 0.0
ramp_up_time: 1.0
gait: trot
scene_xml: {scene}.xml
vlm_path_json: data/{scene}/last_judged_path.json
"""

WIDTHS = [0.70, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35]

def width_tag(w):
    return f"{round(w*100):03d}"

if __name__ == "__main__":
    for w in WIDTHS:
        tag = width_tag(w)
        scene = f"oracle_scene_Dw{tag}"
        half_gap = round(w/2 + 0.15, 3)

        xml = TEMPLATE.format(width_tag=tag, open_width=w, half_gap=half_gap)
        with open(f"dial_mpc/dial_mpc/models/unitree_go2/{scene}.xml", "w") as f:
            f.write(xml)
        with open(f"dial_mpc/dial_mpc/examples/{scene}.yaml", "w") as f:
            f.write(YAML_TEMPLATE.format(scene=scene))

        os.makedirs(f"data/{scene}", exist_ok=True)
        path = [{"x": x, "y": 0.0} for x in [0.0,0.6,1.2,1.8,2.4,3.0,3.6,4.2,4.8,5.4,6.2,6.8]]
        with open(f"data/{scene}/last_judged_path.json", "w") as f:
            json.dump(path, f, indent=2)

        print(f"[{scene}] open_width={w}m 생성 완료")

from dataclasses import dataclass


@dataclass
class BaseEnvConfig: # 최상위 물리 규격 환경 설정 클래스
    task_name: str = "default" # 이름 없는 기본 작업 환경
    randomize_tasks: bool = False  # Whether to randomize the task.
    # P gain, or a list of P gains for each joint. 이뜻은 관절마다 다르게 줄수도 있다는 뜻
    kp: float = 30.0
    # D gain, or a list of D gains for each joint.
    kd: float = 1.0
    debug: bool = False
    # dt of the environment step, not the underlying simulator step.
    dt: float = 0.02
    # timestep of the underlying simulator step. user is responsible for making sure it matches their model.
    timestep: float = 0.02
    backend: str = "mjx"  # backend of the environment.
    # control method for the joints, either "torque" or "position"
    leg_control: str = "torque"
    action_scale: float = 1.0  # scale of the action space.

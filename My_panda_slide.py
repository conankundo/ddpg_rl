import numpy as np

import sys
sys.path.append("..")

from My_panda import My_Panda
from My_slide import My_Slide

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet

from PIL import Image

class My_PandaSlideEnv(RobotTaskEnv):
    """Slide task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
    """

    def __init__(self, render_mode: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render_mode=render_mode)
        self.robot = My_Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        self.task = My_Slide(sim, reward_type=reward_type)
        super().__init__(self.robot, self.task)

env = My_PandaSlideEnv(render_mode="rgb_array", reward_type='sparse', control_type= 'ee')


images = []
n = 100
force1 = [100, 100, 90, 80, 70, 60, 50, 0, 0]
force2 = [500, 200, 200, 200, 200, 200, 200, 0, 0] 
env.task.reset()
env.task.object_type = 0
for _ in range(100):
    if env.task.is_success(env.robot.get_obs()[0:3], env.task.get_obs()[0:3]):
        if env.task.object_type < 2:
            env.task.object_type += 1
            env.task.reset()
            env.robot.reset()
            env.sim.step()  # Ensure the simulation updates
            images.append(env.sim.render())
            
        elif env.task.object_type == 2:
            env.task.object_type = 0
            env.task.reset()
            env.robot.reset()
            env.sim.step()  # Ensure the simulation updates
            images.append(env.sim.render())
            

    # print(env.task.object_type)
    achieved_goal = env.robot.get_obs()[0:3]                            # ee position x y z
    desired_goal = env.task.get_obs()[0:3]                                  # box position x y z
    ee_displace = ((desired_goal - achieved_goal) ).tolist()            # distance

    if not env.robot.block_gripper:
        action = np.append(ee_displace, 0)  # Append 0 for gripper action
    else:
        action = ee_displace
    # 7 joint and 2 finger
    env.robot.set_action(action, force2),                               # do action
    env.sim.step()
    images.append(env.sim.render())

env.sim.close()


images_pil = [Image.fromarray(img) for img in images]
images_pil[0].save(
    "D:/UNI/cac_thuat_toan_thich_nghi/pybullet/Slide/gif_force/gg9.gif",
    save_all=True,
    append_images=images_pil[1:],
    duration=50,  # 50 ms between frames
    loop=1        # Loop forever
)
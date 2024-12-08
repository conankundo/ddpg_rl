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
n = 200
# env.task.reset()
# env.robot.reset()
for _ in range(n):
    if env.task.is_success(env.robot.get_obs()[0:3], env.task.get_goal()):
        env.task.reset()
        env.robot.reset()
        env.sim.step()  # Ensure the simulation updates
        images.append(env.sim.render())
        # break  # Exit loop after resetting
    
    achieved_goal = env.robot.get_obs()[0:3]
    desired_goal = env.task.get_goal()
    ee_displace = ((desired_goal - achieved_goal) ).tolist()
    
    if not env.robot.block_gripper:
        action = np.append(ee_displace, 0)  # Append 0 for gripper action
    else:
        action = ee_displace
    force = [100, 100, 90, 80, 70, 60, 50, 0, 0] # 7 joint and 2 finger
    env.robot.set_action(action, force),
    env.sim.step()
    images.append(env.sim.render())

env.sim.close()


# write_apng("anim20.png", images, delay=50)  # real-time rendering = 40 ms between frames
images_pil = [Image.fromarray(img) for img in images]
images_pil[0].save(
    "D:/UNI/cac_thuat_toan_thich_nghi/pybullet/Slide/gif/g12.gif",
    save_all=True,
    append_images=images_pil[1:],
    duration=50,  # 50 ms between frames
    loop=1        # Loop forever
)
from panda_gym.utils import distance
from typing import Any, Dict, Union, Optional, Tuple
from panda_gym.envs.core import RobotTaskEnv, Task, PyBulletRobot


from numpngw import write_apng
from PIL import Image
import sys
#########################################################
class My_PandaSlideEnv(RobotTaskEnv):
    """Slide task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
    """

    def __init__(self, render_mode: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render_mode=render_mode)
        robot = My_Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = My_Slide(sim, reward_type=reward_type)
        super().__init__(robot, task)
########################################################        
    

env = My_PandaSlideEnv(render_mode="rgb_array", reward_type='sparse', control_type= 'ee')

for __ in range(5):
    images = []
    env.task.reset()  # Reset task positions
    env.robot.reset()  # Reset robot positions
    env.sim.step()  # Step the simulation to apply the reset
    images.append(env.sim.render())
    n = 20
    for _ in range(n):
        achieved_goal = env.robot.get_obs()[0:3]
        desired_goal = env.task.get_goal()
        
        if env.task.is_success(achieved_goal, desired_goal):
            print("Resetting environment after success.")
            env.task.reset()
            env.robot.reset()
            env.sim.step()  # Ensure the simulation updates
            images.append(env.sim.render())
            break  # Exit loop after resetting

        ee_displace = ((desired_goal - achieved_goal) / 10).tolist()
        action = np.clip(ee_displace, env.robot.action_space.low[:3], env.robot.action_space.high[:3])
        env.robot.set_action(action)
        env.sim.step()
        images.append(env.sim.render())

env.sim.close()


# write_apng("anim20.png", images, delay=50)  # real-time rendering = 40 ms between frames
images_pil = [Image.fromarray(img) for img in images]
images_pil[0].save(
    "anim7.gif",
    save_all=True,
    append_images=images_pil[1:],
    duration=50,  # 50 ms between frames
    loop=0        # Loop forever
)
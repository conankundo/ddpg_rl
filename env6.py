import gymnasium as gym
import numpy as np
import random
from panda_gym.envs.core import RobotTaskEnv, Task, PyBulletRobot
from panda_gym.pybullet import PyBullet
# from panda_gym.envs.robots.panda import Panda
# from panda_gym.envs.tasks.slide import Slide
from panda_gym.utils import distance
from typing import Any, Dict, Union, Optional, Tuple
import random

from numpngw import write_apng
# from IPython.display import Image
from PIL import Image

class My_Slide(Task):
    def __init__(
        self,
        sim,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_xy_range=0.3,
        goal_x_offset=0.4,
        obj_xy_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.06
        self.goal_range_low = np.array([-goal_xy_range / 2 + goal_x_offset, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2 + goal_x_offset, goal_xy_range / 2, 0])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self):
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.4, width=0.7, height=0.4, x_offset=-0.1)
        self.sim.create_cylinder(
            body_name="object",
            mass=2,
            radius=self.object_size / 2,
            height=self.object_size / 2,
            position=np.array([0.0, 0.0, 0]),
            rgba_color=np.array([1, 1, 1, 1.0]), #white
            lateral_friction=0.04,
        )
        self.sim.create_cylinder(
            body_name="target",
            mass=0.0,
            ghost=True,
            radius=self.object_size / 2,
            height=self.object_size / 2,
            position=np.array([0.0, 0.0, 0]),
            rgba_color=np.array([0, 0, 1, 1.0]), #blue
        )

    def get_goal(self):
        return self.goal.copy()

    def get_obs(self):
        # position, rotation of the object
        object_position = np.array(self.sim.get_base_position("object"))
        object_rotation = np.array(self.sim.get_base_rotation("object"))
        object_velocity = np.array(self.sim.get_base_velocity("object"))
        object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object"))
        observation = np.concatenate(
            [
                object_position,
                object_rotation,
                object_velocity,
                object_angular_velocity,
            ]
        )
        return observation

    def get_achieved_goal(self):
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self):
        self.goal = np.array([round(random.uniform(-0.25,0.25), 2), round(random.uniform(-0.25,0.25), 2), self.object_size / 2])
        object_position = np.array([round(random.uniform(-0.25,0.25), 2), round(random.uniform(-0.25,0.25), 2), self.object_size / 2])
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    # def _sample_goal(self):
    #     """Randomize goal."""
    #     goal = np.array([random.uniform(-1, 1), random.uniform(-1, 1), self.object_size / 2])  # z offset for the cube center
    #     noise = random.uniform(self.goal_range_low, self.goal_range_high)
    #     goal += noise
    #     return goal.copy()

    # def _sample_object(self) -> np.ndarray:
    #     """Randomize start position of object."""
    #     object_position = np.array([random.uniform(-1, 1), random.uniform(-1, 1), self.object_size / 2])
    #     noise = random.uniform(self.obj_range_low, self.obj_range_high)
    #     object_position += noise
    #     return object_position

    def is_success(self, achieved_goal, desired_goal) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d

class My_Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
        control_type: str = "ee",
    ) -> None:
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )

        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray) -> None:
        # action = action.copy()  # ensure action don't change
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        # ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint anglesse
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            obs = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            obs = np.concatenate((ee_position, ee_velocity))
        return obs

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the ned-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

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
        
    

env = My_PandaSlideEnv(render_mode="rgb_array", reward_type='sparse', control_type= 'ee')

# for __ in range(5):
#     # observation, info = env.reset()
#     print(__)
#     # env.task.reset()
#     # env.robot.reset()
#     images = [env.sim.render()]
#     n = 20
#     for _ in range(n):
#         if env.task.is_success( env.robot.get_obs()[0:3], env.task.get_obs()[0:3] ):
#             env.task.reset()
#             env.robot.reset()
#             images.append(env.sim.render())
#             print('reset')
            
#         ee_displace = ((env.task.get_obs()[0:3] - env.robot.get_obs()[0:3]) / 10).tolist()
#         # print('3', ee_displace)
#         action = np.clip(ee_displace, env.robot.action_space.low[:3], env.robot.action_space.high[:3])
#         # print('3', action)
#         env.robot.set_action(action)
#         env.sim.step()
#         images.append(env.sim.render())

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
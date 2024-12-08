from typing import Any, Dict, Union

import numpy as np
import random
from gymnasium import utils

from panda_gym.envs.core import Task
from panda_gym.utils import distance


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
        self.object_type = 0
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
            body_name="object_r",
            mass=2,
            radius=self.object_size / 2,
            height=self.object_size / 2,
            position=np.array([0.1, 0, self.object_size / 2]),
            rgba_color=np.array([1, 0.1, 0.1, 1.0]), #r
            lateral_friction=0.04,
        )
        self.sim.create_cylinder(
            body_name="object_b",
            mass=2,
            radius=self.object_size / 2,
            height=self.object_size / 2,
            position=np.array([0.2, 0, self.object_size / 2]),
            rgba_color=np.array([0, 0, 1, 1.0]), #b
            lateral_friction=0.04,
        )
        self.sim.create_cylinder(
            body_name="object_g",
            mass=2,
            radius=self.object_size / 2,
            height=self.object_size / 2,
            position=np.array([-0.2, 0, self.object_size / 2]),
            rgba_color=np.array([0, 1, 0, 1.0]), #g
            lateral_friction=0.04,
        )
        self.sim.create_cylinder(
            body_name="target_b",
            mass=0.0,
            ghost=True,
            radius=self.object_size / 2,
            height=self.object_size / 2,
            position=np.array([0.25, 0.25, self.object_size / 2]),
            rgba_color=np.array([0, 0, 1, 0.01]), #blue
        )
        self.sim.create_cylinder(
            body_name="target_g",
            mass=0.0,
            ghost=True,
            radius=self.object_size / 2,
            height=self.object_size / 2,
            position=np.array([0.25, 0.5, 0, self.object_size / 2]),
            rgba_color=np.array([0, 0.9, 0, 0.1]), #blue
        )
        self.sim.create_cylinder(
            body_name="target_r",
            mass=0.0,
            ghost=True,
            radius=self.object_size / 2,
            height=self.object_size / 2,
            position=np.array([0.25, 0.75,  0, self.object_size / 2]),
            rgba_color=np.array([1, 0.1, 0.1, 0.1]), #blue
        )

    def get_goal(self):
        return self.goal.copy()

    def get_obs(self):
        # position, rotation of the object
        
        if self.object_type == 0:
            object_position = np.array(self.sim.get_base_position("object_r"))
            object_rotation = np.array(self.sim.get_base_rotation("object_r"))
            object_velocity = np.array(self.sim.get_base_velocity("object_r"))
            object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object_r"))
            observation = np.concatenate(
                [
                    object_position,
                    object_rotation,
                    object_velocity,
                    object_angular_velocity,
                ]
            )
        elif self.object_type == 1:
            object_position = np.array(self.sim.get_base_position("object_g"))
            object_rotation = np.array(self.sim.get_base_rotation("object_g"))
            object_velocity = np.array(self.sim.get_base_velocity("object_g"))
            object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object_g"))
            observation = np.concatenate(
                [
                    object_position,
                    object_rotation,
                    object_velocity,
                    object_angular_velocity,
                ]
            )
        elif self.object_type == 2:
            object_position = np.array(self.sim.get_base_position("object_b"))
            object_rotation = np.array(self.sim.get_base_rotation("object_b"))
            object_velocity = np.array(self.sim.get_base_velocity("object_b"))
            object_angular_velocity = np.array(self.sim.get_base_angular_velocity("object_b"))
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
        object_position = np.array(self.sim.get_base_position("object_r"))
        return object_position

    def reset(self):
        # object_position = self._sample_object()
        self.goal = np.array([0.25, 0.75, self.object_size / 2])
        # object_position = np.array([round(random.uniform(-0.25,0.25), 2), round(random.uniform(-0.25,0.25), 2), 0])
        self.sim.set_base_pose("target_r", np.array([0.25, 0.25, self.object_size / 2]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target_g", np.array([0.25, 0.0, self.object_size / 2]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("target_b", np.array([0.25, -0.25, self.object_size / 2]), np.array([0.0, 0.0, 0.0, 1.0]))
        # self.object_type = random.randint(0, 2)
        if self.object_type == 0:
            self.sim.set_base_pose("object_r",np.array([round(random.uniform(-0.25,0.25), 2), round(random.uniform(-0.25,0.25), 2), 0]), np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object_g", np.array([5, 5, 0.1]), np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object_b", np.array([5, 5, 0.1]), np.array([0.0, 0.0, 0.0, 1.0]))
        elif self.object_type == 1:
            self.sim.set_base_pose("object_r", np.array([5, 5, 0.1]), np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object_g", np.array([round(random.uniform(-0.25,0.25), 2), round(random.uniform(-0.25,0.25), 2), 0]), np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object_b", np.array([5, 5, 0.1]), np.array([0.0, 0.0, 0.0, 1.0]))
        elif self.object_type == 2:
            self.sim.set_base_pose("object_r", np.array([5, 5, 0.1]), np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object_g", np.array([5, 5, 0.1]), np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("object_b", np.array([round(random.uniform(-0.25,0.25), 2), round(random.uniform(-0.25,0.25), 2), 0]), np.array([0.0, 0.0, 0.0, 1.0]))
        # self.sim.set_base_pose("object_b", np.array([round(random.uniform(-0.25,0.25), 2), round(random.uniform(-0.25,0.25), 2), 0]), np.array([0.0, 0.0, 0.0, 1.0]))
        # self.sim.set_base_pose("object_g", np.array([round(random.uniform(-0.25,0.25), 2), round(random.uniform(-0.25,0.25), 2), 0]), np.array([0.0, 0.0, 0.0, 1.0]))
        # self.sim.set_base_pose("object_r",np.array([round(random.uniform(-0.25,0.25), 2), round(random.uniform(-0.25,0.25), 2), 0]), np.array([0.0, 0.0, 0.0, 1.0]))


    # def _sample_goal(self) -> np.ndarray:
    #     """Sample a goal."""
    #     self.sub_goalr = np.array([0.25, 0.75, self.object_size / 2])  # z offset for the cube center
    #     self.sub_goalg = np.array([0.25, 0.5, self.object_size / 2])  # z offset for the cube center
    #     self.sub_goalb = np.array([0.25, 0.25, self.object_size / 2])  # z offset for the cube center
    #     # noise1 = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
    #     # self.sub_goal1 += noise1

    #     # self.sub_goal2 = self.sub_goal1 + np.array([0.05, -0.05, 0.1])

    #     # goal = np.concatenate((self.sub_goalr, self.sub_goalg, self.sub_goalb))
    #     goal = self.sub_goalr

    #     return goal.copy()

    # def _sample_object(self) -> np.ndarray:
    #     """Randomize start position of object."""
    #     object_position = np.array([0.0, 0.0, self.object_size / 2])
    #     noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
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


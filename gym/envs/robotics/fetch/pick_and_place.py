from gym import utils
from gym.envs.robotics import fetch_env


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 1.05,
            'table0:slide1': 0.4,
            'table0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.45, 1., 0., 0., 0.],
            'object1:joint': [1.35, 0.63, 0.45, 1., 0., 0., 0.],
            # 'object2:joint': [1.45, 0.73, 0.45, 1., 0., 0., 0.],

            'bow0:joint': [1.15, 0.53, 0.45, 1., 0., 0., 0.],
            'bow1:joint': [1.35, 0.73, 0.45, 1., 0., 0., 0.],
            # 'bow2:joint': [1.45, 0.93, 0.45, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, 'fetch/pick_and_place.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.035, target_range=0.035, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)


        # fetch_env.FetchEnv.__init__(
        #     self, 'fetch/pick_and_place.xml', has_object=True, block_gripper=False, n_substeps=20,
        #     gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
        #     obj_range=0.0, target_range=0.0, distance_threshold=0.05,
        #     initial_qpos=initial_qpos, reward_type=reward_type)
        
        utils.EzPickle.__init__(self)

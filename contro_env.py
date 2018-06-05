import pickle
import gym
import numpy as np
import random
from PIL import Image


def decide_move(offset):
    global min
    global max
    plus = 2
    minus = 1
    fixed = 0
    action = [0, 0, 0]
    for i in range(3):
        if offset[i] > max:
            action[i] = plus
        elif offset[i] < min:
            action[i] = minus
        else:
            action[i] = fixed
    return action


def evalue_move(offset):
    global min
    global max
    if offset[0] < max and offset[1] < max and offset[2] < max and offset[0] > min and offset[1] > min and offset[2] > min:
        return True 
    else:
        return False


def pick_and_place(position_claw, position_object, position_target):
    # suppose the claw already at standard height

    global stage
    global close_counter
    global open_counter
    global raise_counter
    global ground_height

    
    plus = 2
    minus = 1
    fixed = 0
    hand_open = 1
    close = 0
    success = False

    action = [fixed, fixed, fixed, close]

    # stpe_1: reach_object_above (open hand)
    if stage == "reach_object_above":
        action_3 = position_object - position_claw + [0, 0, 0.1]

        if evalue_move(action_3):
            stage = "reach_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = hand_open

    # step_2: reach_object (open hand)
    elif stage == "reach_object":

        action_3 = position_object - position_claw

        if evalue_move(action_3):
            stage = "grasp_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = hand_open

    # step_3: grasp_object (close hand)
    elif stage == "grasp_object":
        if close_counter < 3:
            close_counter = close_counter + 1
        else:
            stage = "raise_object_up"
            close_counter = 0
        action[3] = close

    # step_4: raise_object_up (close hand)
    elif stage == "raise_object_up":

        action_3 = [0, 0, ground_height + 0.1 - position_claw[2] ]

        if evalue_move(action_3):
            stage = "reach_target_above"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = close

    # step_5: reach_target_above (close hand)
    elif stage == "reach_target_above":

        action_3 = position_target - position_object + [0, 0, 0.1]

        if evalue_move(action_3):
            stage = "lower_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = close

    # step_6: lower_object (close hand)
    elif stage == "lower_object":

        action_3 = position_target - position_object + [0, 0, 0.04]

        if evalue_move(action_3):
            stage = "release_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = close

    # step_7: release_object (open hand)
    elif stage == "release_object":

        if open_counter < 3:
            open_counter = open_counter + 1
        else:
            stage = "raise_claw_up"
            open_counter = 0
        action[3] = hand_open

    # step_8: raise_claw_up (open hand)
    elif stage == "raise_claw_up":

        action_3 = [0, 0, ground_height + 0.1 - position_claw[2] ]

        if evalue_move(action_3):
            stage = "inside_finish"
            success = True
        else:
            action[0:3] = decide_move(action_3)
        action[3] = hand_open

    return action, success


def subtask_decide(strategy_id, 
                    object_0_position, object_1_position, 
                    bow_0_position, bow_1_position, 
                    goal_0_position, goal_1_position):
    if strategy_id == 0:
        return object_0_position, bow_0_position
    elif strategy_id == 1:
        return object_1_position, bow_1_position
    elif strategy_id == 2:
        return bow_0_position, goal_0_position
    elif strategy_id == 3:
        return bow_1_position, goal_1_position


# Hyper parameters
max_push_step = 6
push_point_distance = 0.07
step_size = 0.01


# global parameters
close_counter = 0
open_counter = 0
push_counter = 0
raise_counter = 0
min = -step_size
max = step_size
ground_height = 0.0



env = gym.make('FetchPickAndPlace-v0')

stage_set = ["raise_gasket_above_again"

             "reach_object_above", "reach_object", "grasp_object", "raise_object_above",
             "reach_gasket_above", "reach_gasket_half_above", "release", 

             "reach_gasket_above_for_grasp", 

             "reach_gasket", "grasp_gasket_and_object", "raise_gasket_and_object_above",
             "reach_target_above", "reach_target_half_above", "release_gasket_and_object",

             "check", "stay"]

data = []
trajectory_num = 0

image_num_already_success = 0


while True:
    # begins a new trajectory

    stage = "reach_object_above"
    stage_outside = "step_1"

    observation = env.reset()
    done = False

    gripper_position = observation["my_new_observation"][0:3]
    object_0_position = observation["my_new_observation"][5:8]
    object_1_position = observation["my_new_observation"][8:11]
    bow_0_position = observation["my_new_observation"][11:14]
    bow_1_position = observation["my_new_observation"][14:17]
    goal_0_position = observation["my_new_observation"][17:20]
    goal_1_position = observation["my_new_observation"][20:23]

    plus = 2
    minus = 1
    fixed = 0
    hand_open = 1
    close = 0

    ground_height = object_0_position[2]

    one_trajectory = []
    final_success = False

    image_num = image_num_already_success

    strategy = np.random.permutation(4)

    # to avoid all black image
    # env.render(mode='rgb_array')

    while not done:

        # saving image
        # image = env.render(mode='rgb_array')
        # image = Image.fromarray(image)
        # w, h = image.size
        # image = image.resize((w//4, h//4),Image.ANTIALIAS)  
        # image.save('images/'+ str(image_num) +'.jpg', 'jpeg')
        # image_num += 1

        # NOT saving image
        # env.render()

        if stage_outside == "step_1":
            ob, tar = subtask_decide(strategy[0], 
                                    object_0_position, object_1_position, 
                                    bow_0_position, bow_1_position, 
                                    goal_0_position, goal_1_position)
            action_category, success = pick_and_place(gripper_position, ob, tar)
            if success:
                stage_outside = "step_2"
                stage = "reach_object_above"
        elif stage_outside == "step_2":
            ob, tar = subtask_decide(strategy[1], 
                                    object_0_position, object_1_position, 
                                    bow_0_position, bow_1_position, 
                                    goal_0_position, goal_1_position)
            action_category, success = pick_and_place(gripper_position, ob, tar)
            if success:
                stage_outside = "step_3"
                stage = "reach_object_above"
        elif stage_outside == "step_3":
            ob, tar = subtask_decide(strategy[2], 
                                    object_0_position, object_1_position, 
                                    bow_0_position, bow_1_position, 
                                    goal_0_position, goal_1_position)
            action_category, success = pick_and_place(gripper_position, ob, tar)
            if success:
                stage_outside = "step_4"
                stage = "reach_object_above"
        elif stage_outside == "step_4":
            ob, tar = subtask_decide(strategy[3], 
                                    object_0_position, object_1_position, 
                                    bow_0_position, bow_1_position, 
                                    goal_0_position, goal_1_position)
            action_category, success = pick_and_place(gripper_position, ob, tar)
            if success:
                stage_outside = "outside_finish"
                stage = "reach_object_above"
                final_success = True

        action = np.zeros((4))

        # change from category to value
        for i in range(3):
            if action_category[i] == plus:
                action[i] = (step_size/0.03)
            elif action_category[i] == minus:
                action[i] = -(step_size/0.03)
            elif action_category[i] == fixed:
                action[i] = 0.0

        if action_category[3] == hand_open:
            action[3] = 1.0
        elif action_category[3] == close:
            action[3] = -1.0

        previous_observation = observation
        observation, reward, done, info = env.step(action)

        gripper_position = observation["my_new_observation"][0:3]
        object_0_position = observation["my_new_observation"][5:8]
        object_1_position = observation["my_new_observation"][8:11]
        bow_0_position = observation["my_new_observation"][11:14]
        bow_1_position = observation["my_new_observation"][14:17]
        goal_0_position = observation["my_new_observation"][17:20]
        goal_1_position = observation["my_new_observation"][20:23]
        
        one_step = []
        one_step.extend(previous_observation["my_new_observation"])
        one_step.extend(action_category)
        one_step.extend(observation["my_new_observation"])
        one_step.extend([float(final_success)])
        one_trajectory.append(one_step)

        if stage_outside == "outside_finish":
            data.extend(one_trajectory)
            trajectory_num += 1
            image_num_already_success += len(one_trajectory)
            break


    print(trajectory_num)

    if trajectory_num == 24000:
        break

print("total trajectory num is : ",end="")
print(image_num_already_success)

data = np.array(data)
pickle.dump(data, open("PP-24-paths-"+str(trajectory_num)+".p", "wb"))


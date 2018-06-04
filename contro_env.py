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


def calculate_desired_goal(observation, hight):
    global push_point_distance
    target_position = observation["my_new_observation"][23:26]
    bow_position = observation["my_new_observation"][14:17]

    delta = target_position - bow_position
    ratio = abs(delta[1]/delta[0])
    if delta[0] < 0:
        x_sign = -1
    else:
        x_sign = 1

    if delta[1] < 0:
        y_sign = -1
    else:
        y_sign = 1 

    if ratio > 8:
        abs_delta = [0, 1]
    elif ratio > 8/3:
        abs_delta = [2/8, 1]
    elif ratio > 8/5:
        abs_delta = [4/8, 1]
    elif ratio > 8/7:
        abs_delta = [6/8, 1]
    elif ratio > 7/8:
        abs_delta = [1, 8/8]
    elif ratio > 5/8:
        abs_delta = [1, 6/8]
    elif ratio > 3/8:
        abs_delta = [1, 4/8]
    elif ratio > 1/8:
        abs_delta = [1, 2/8]
    else:
        abs_delta = [1, 0]

    delta_ac = [abs_delta[0]*x_sign*push_point_distance, abs_delta[1]*y_sign*push_point_distance, 0]
    desired_goal = bow_position - delta_ac
    desired_goal[2] = bow_position[2] + hight
    return desired_goal


def pick_and_place(position_claw, position_object, position_target):
    # suppose the claw already at standard height

    global stage
    global close_counter
    global open_counter
    global raise_counter
    
    plus = 2
    minus = 1
    fixed = 0
    hand_open = 1
    close = 0
    success = False

    action = [fixed, fixed, fixed, close]

    # stpe_1: reach_object_above (open hand)
    if stage == "reach_object_above":
        action_3 = position_object - position_claw
        action_3[2] = action_3[2] + 0.07

        if evalue_move(action_3):
            stage = "reach_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = hand_open

    # step_2: reach_object (open hand)
    elif stage == "reach_object":

        action_3 = position_object - position_claw
        action_3[2] = action_3[2] + 0.0

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
        if raise_counter < 8:
            raise_counter = raise_counter + 1
            action[2] = plus
        else:
            stage = "reach_target_above"
            raise_counter = 0
        action[3] = close

    # step_5: reach_target_above (close hand)
    elif stage == "reach_target_above":

        action_3 = position_target - position_object
        action_3[2] = action_3[2] + 0.1

        if evalue_move(action_3):
            stage = "lower_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = close

    # step_6: lower_object (close hand)
    elif stage == "lower_object":

        action_3 = position_target - position_object
        action_3[2] = action_3[2] + 0.04


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

        if raise_counter < 8:
            raise_counter = raise_counter + 1
            action[2] = plus
        else:
            stage = "finish"
            raise_counter = 0
            success = True
        action[3] = hand_open


    return action, success


def policy(observation):
    global stage
    global close_counter
    global open_counter
    global push_counter
    global max_push_step
    global raise_counter
    global strategy
    
    plus = 2
    minus = 1
    fixed = 0
    hand_open = 1
    close = 0
    success = False

    action = [fixed, fixed, fixed, close]

    # ======================== p_1 to e_1===========================

    if stage == "raise_gasket_above_again":
        if raise_counter < 8:
            raise_counter = raise_counter + 1
            action[2] = plus
        else:
            stage = "reach_object_above"
            raise_counter = 0
        action[3] = hand_open

    # ======================== e_1 to p_2/p_3=============================

    elif stage == "reach_object_above":
        # move towards the object
        # if distance > 0.03 of distance < -0.03, using 1/-1
        # else using distance exactly
        action_3 = observation["observation"][6:9]
        action_3[2] = action_3[2] + 0.07

        # remember:
        # CAN'T change min and max casually
        # because it should be the same to the step with classification standard step size

        if evalue_move(action_3):
            # print("reach the target !!")
            stage = "reach_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = hand_open

    elif stage == "reach_object":

        action_3 = observation["observation"][6:9]
        action_3[2] = action_3[2] + 0.0

        if evalue_move(action_3):
            # print("go down already !!")
            stage = "grasp_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = hand_open

    elif stage == "grasp_object":
        # close the claw !!
        if close_counter < 3:
            close_counter = close_counter + 1
        else:
            # print("close the claw !!")
            stage = "raise_object_above"
            close_counter = 0
        action[3] = close

    elif stage == "raise_object_above":
        if raise_counter < 8:
            raise_counter = raise_counter + 1
            action[2] = plus
        else:
            stage = "reach_gasket_above"
            raise_counter = 0
        action[3] = close

    elif stage == "reach_gasket_above":

        gasket_position = observation["my_new_observation"][14:17] + [0, 0, 0.1]


        object_position = observation["my_new_observation"][5:8]
        action_3 = gasket_position - object_position


        if evalue_move(action_3):
            # print("reach already !!")
            stage = "reach_gasket_half_above"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = close

    elif stage == "reach_gasket_half_above":

        gasket_position = observation["my_new_observation"][14:17] + [0, 0, 0.04]

        object_position = observation["my_new_observation"][5:8]
        action_3 = gasket_position - object_position


        if evalue_move(action_3):
            stage = "release"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = close

    elif stage == "release":

        if open_counter < 3:
            open_counter = open_counter + 1
        else:
            if strategy == "pick_object_first":
                stage = "reach_gasket"
            elif strategy == "pick_gasket_first":
                stage = "check"
            else: 
                print("wrong!!!!!!")
            open_counter = 0
        action[3] = hand_open

    # ======================= e_2 to p_2============================

    elif stage == "reach_gasket_above_for_grasp":

        gasket_position = observation["my_new_observation"][14:17] + [0, 0, 0.1]
        gripper_position = observation["my_new_observation"][0:3]

        action_3 = gasket_position - gripper_position

        if evalue_move(action_3):
            # print("go down already !!")
            stage = "reach_gasket"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = hand_open

    # ======================= p_2 to p_3/p_1============================

    elif stage == "reach_gasket":

        gasket_position = observation["my_new_observation"][14:17]
        gripper_position = observation["my_new_observation"][0:3]

        action_3 = gasket_position - gripper_position

        if evalue_move(action_3):
            # print("go down already !!")
            stage = "grasp_gasket_and_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = hand_open

    elif stage == "grasp_gasket_and_object":
        # close the claw !!
        if close_counter < 3:
            close_counter = close_counter + 1
        else:
            # print("close the claw !!")
            stage = "raise_gasket_and_object_above"
            close_counter = 0
        action[3] = close

    elif stage == "raise_gasket_and_object_above":
        if raise_counter < 8:
            raise_counter = raise_counter + 1
            action[2] = plus
        else:
            stage = "reach_target_above"
            raise_counter = 0
        action[3] = close

    elif stage == "reach_target_above":

        target_position = observation["my_new_observation"][23:26] + [0, 0, 0.1]
        gasket_position = observation["my_new_observation"][14:17]

        action_3 = target_position - gasket_position


        if evalue_move(action_3):
            # print("reach already !!")
            stage = "reach_target_half_above"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = close

    elif stage == "reach_target_half_above":

        target_position = observation["my_new_observation"][23:26] + [0, 0, 0.02]
        gasket_position = observation["my_new_observation"][14:17]

        action_3 = target_position - gasket_position


        if evalue_move(action_3):
            stage = "release_gasket_and_object"
        else:
            action[0:3] = decide_move(action_3)
        action[3] = close

    elif stage == "release_gasket_and_object":

        if open_counter < 3:
            open_counter = open_counter + 1
        else:
            if strategy == "pick_object_first":
                stage = "check"
            elif strategy == "pick_gasket_first":
                stage = "raise_gasket_above_again"
            else:
                print("wrong !!!!!!!!")
            open_counter = 0
        action[3] = hand_open

    # ======================= p_3 ============================

    elif stage == "check":
        # if failed, will loop here
        object0_height = observation["my_new_observation"][7]

        if object0_height > 0.441:
            stage = "stay"
            success = True

    elif stage == "stay":

        pass
        
    return action, success



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

env = gym.make('FetchPickAndPlace-v0')


stage_set = [
             "raise_gasket_above_again"


             "reach_object_above", "reach_object", "grasp_object", "raise_object_above",
             "reach_gasket_above", "reach_gasket_half_above", "release", 


             "reach_gasket_above_for_grasp", 


             "reach_gasket", "grasp_gasket_and_object", "raise_gasket_and_object_above",
             "reach_target_above", "reach_target_half_above", "release_gasket_and_object",


             "check", "stay"

             ]

strategy_set = ["pick_object_first", "pick_gasket_first"]


data = []
trajectory_num = 0

image_num_already_success = 0

strategy_pair = {"pick_object_first": "reach_object_above",
                 "pick_gasket_first": "reach_gasket_above_for_grasp"}

strategy = "pick_object_first"

while True:

    # if strategy == "pick_object_first":
    #     strategy = "pick_gasket_first"
    # elif strategy == "pick_gasket_first":
    #     strategy = "pick_object_first"
    

    # strategy = strategy_set[random.randint(0, 3)]

    # stage = strategy_pair[strategy]

    stage = "reach_object_above"

    observation = env.reset()
    done = False
    # print(observation)

    exbanded_reduced_new_obs = np.zeros((32,))

    # gripper position
    gripper_position = observation["my_new_observation"][0:3]
    exbanded_reduced_new_obs[0:5] = observation["my_new_observation"][0:5]

    # object_0 position
    object_0_position = observation["my_new_observation"][5:8]
    exbanded_reduced_new_obs[5:8] = observation["my_new_observation"][5:8]

    # object_1 position
    object_1_position = observation["my_new_observation"][8:11]
    exbanded_reduced_new_obs[8:11] = observation["my_new_observation"][8:11]

    # bow_0 position
    bow_0_position = observation["my_new_observation"][11:14]
    exbanded_reduced_new_obs[14:17] = observation["my_new_observation"][11:14]

    # bow_1 position
    bow_1_position = observation["my_new_observation"][14:17]
    exbanded_reduced_new_obs[17:20] = observation["my_new_observation"][14:17]

    # goal_0 position
    goal_0_position = observation["my_new_observation"][17:20]
    exbanded_reduced_new_obs[23:26] = observation["my_new_observation"][17:20]

    # goal_1 position
    goal_1_position = observation["my_new_observation"][20:23]
    exbanded_reduced_new_obs[26:29] = observation["my_new_observation"][20:23]

    observation["my_new_observation"] = exbanded_reduced_new_obs

    plus = 2
    minus = 1
    fixed = 0

    hand_open = 1
    close = 0

    one_trajectory = []

    image_num = image_num_already_success

    # to avoid all black image
    # env.render(mode='rgb_array')

    while not done:
        # saving image
        # image = env.render(mode='rgb_array')
        # image = Image.fromarray(image)
        # w, h = image.size
        # image = image.resize((w//4, h//4),Image.ANTIALIAS)  
        # image.save('images_'+strategy+'/'+ str(image_num) +'.jpg', 'jpeg')
        # image_num += 1

        # NOT saving image
        env.render()

        # action_category, success = policy(observation)
        action_category, success = pick_and_place(gripper_position, object_0_position, bow_0_position)

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

        exbanded_reduced_new_obs = np.zeros((32,))

        # gripper position
        gripper_position = observation["my_new_observation"][0:3]
        exbanded_reduced_new_obs[0:5] = observation["my_new_observation"][0:5]

        # object_0 position
        object_0_position = observation["my_new_observation"][5:8]
        exbanded_reduced_new_obs[5:8] = observation["my_new_observation"][5:8]

        # object_1 position
        object_1_position = observation["my_new_observation"][8:11]
        exbanded_reduced_new_obs[8:11] = observation["my_new_observation"][8:11]

        # bow_0 position
        bow_0_position = observation["my_new_observation"][11:14]
        exbanded_reduced_new_obs[14:17] = observation["my_new_observation"][11:14]

        # bow_1 position
        bow_1_position = observation["my_new_observation"][14:17]
        exbanded_reduced_new_obs[17:20] = observation["my_new_observation"][14:17]

        # goal_0 position
        goal_0_position = observation["my_new_observation"][17:20]
        exbanded_reduced_new_obs[23:26] = observation["my_new_observation"][17:20]

        # goal_1 position
        goal_1_position = observation["my_new_observation"][20:23]
        exbanded_reduced_new_obs[26:29] = observation["my_new_observation"][20:23]
        
        observation["my_new_observation"] = exbanded_reduced_new_obs

        one_step = []
        one_step.extend(previous_observation["my_new_observation"])
        one_step.extend(action_category)
        one_step.extend(observation["my_new_observation"])
        one_step.extend([float(success)])
        one_trajectory.append(one_step)

        if success:
            data.extend(one_trajectory)
            trajectory_num += 1
            image_num_already_success += len(one_trajectory)
            break


    print(trajectory_num)

    if trajectory_num == 200:
        break

print("total trajectory num is : ",end="")
print(image_num_already_success)

# data = np.array(data)
# pickle.dump(data, open("PPP"+strategy+"-200.p", "wb"))


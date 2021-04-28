"""
Environment for Robot Arm in V-REP.
"""
from robot import Robot
import numpy as np
import matplotlib.pyplot as plt
import time
from threading import Timer
import argparse
from PIL import Image
import cv2
import vrep
import utils
import random
from configs import Configs
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

class Agent(object):

    def __init__(self):
        train_configs = Configs()
        self.workspace_limits = train_configs.WORKSPACE_LIMITS
        self.obj_mesh_dir = train_configs.OBJ_MESH_DIR
        self.texture_dir = train_configs.TEXTURE_DIR
        self.max_num_obj = train_configs.MAX_OBJ_NUM

        # Initialize camera and robot
        self.robot = Robot(self.num_obj, self.workspace_limits)

    def printForce(self):
        self.robot.record_force()
        t = Timer(1, self.printForce, )
        t.start()

    def step(self, pred_pix_id, pred_rotation_id):
        '''
        :param pred_position: [x,y,z]
        :return:
        '''
        # action = (action_x,action_y,action_z,action_oz)
        # action = my_utils.predict_actionID_to_execution_action(action_id,current_depth_path)
        robot_position = utils.tran_pixID_to_robotXYZ()
        robot_position_up = [robot_position[0],robot_position[1],robot_position[2]+0.1]
        self.robot.move_to(robot_position_up,None)
        time.sleep(0.5)
        self.robot.rotate_gripper(pred_rotation_id*np.pi/2)
        time.sleep(0.5)
        self.robot.move_to(robot_position,None)
        time.sleep(0.3)
        self.robot.control_suctionPad(0)

        # return_r,finish = self.robot.step(action)
        # if finish == -1:
        #     return 'null',0,-1
        # else:
        #     # path_color,path_depth = self.robot.get_current_state()
        #     path_depth = self.robot.get_current_state()
        #     if return_r == 0:
        #         return_r = return_r - 0.05
        #     print('step:',step,', action_type:',ac_type,', action_id:',action_id,', execute action:', action,'reward:',return_r)
        #     return path_depth,return_r,finish #new_state, return_r,finish

    def step_eval(self, action_id,current_depth_path,step):
        action = my_utils.predict_actionID_to_execution_action(action_id,current_depth_path)

        return_r,finish = self.robot.step(action)
        if finish == -1:
            return 'null',0,-1
        else:
            # path_color,path_depth = self.robot.get_current_state()
            path_depth = self.robot.get_current_state_eval()
            if return_r == 0:
                return_r = return_r - 0.1
            print('step:',step,', action_type: 0, action_id:',action_id,', execute action:', action,'reward:',return_r)
            return path_depth,return_r,finish #new_state, return_r,finish

    def reset(self):
        # self.grab_counter = 0
        obj_nums = 0
        while obj_nums == 0:
            self.robot.restart_sim()
            obj_nums = self.robot.add_objects(self.texture_dir)
            #self.robot.random_position()
        path_depth = self.robot.get_current_state()
        return path_depth,obj_nums #new_state#self.robot.get_current_state()
    def reset_eval(self):
        # self.grab_counter = 0

        self.robot.restart_sim()
        obj_nums = self.robot.add_objects(self.texture_dir)
        #self.robot.random_position()
        path_depth = self.robot.get_current_state_eval()
        return obj_nums,path_depth #new_state#self.robot.get_current_state()
    def sample_action(self):
        # position_limits np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
        # angle_limits (0,2*np.pi)
        s_x = random.uniform(-0.724, -0.276)
        s_y = random.uniform(-0.224, 0.224)
        s_z = random.uniform(-0.0001, 0.4)
        s_o = random.uniform(0, 2*np.pi)
        #C = 2  # 随机数的精度round(数值，精度)

        p_x = round(s_x, 3)
        p_y = round(s_y, 3)
        p_z = round(s_z, 4)
        o_z = round(s_o, 4)

        return (p_x,p_y,p_z,o_z)

    def cal_change_for_R(self,p0,o0,p1,o1):
        num = len(p0)
        sum_p = 0
        sum_o = 0
        for i in range(num):
            sum_p = sum_p + abs(p0[i][0] - p1[i][0]) + abs(p0[i][1] - p1[i][1]) + abs(p0[i][2] - p1[i][2])
            sum_o = sum_o + abs(o0[i][0] - o1[i][0]) + abs(o0[i][1] - o1[i][1]) + abs(o0[i][2] - o1[i][2])

        sum_p /= (num*3)
        sum_o /= (num*3)
        if sum_p >= 0.01 or sum_o >= 0.01:
            return (sum_p - 0.01) * 0.5 + (sum_o - 0.01) * 0.5
        else:
            return 0

    def go_to_position(self,position):
        self.robot.go_to_position(position)

    def get_objects_positions(self):
        return self.robot.get_obj_positions_and_orientations()
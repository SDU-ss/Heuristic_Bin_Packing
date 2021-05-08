from robot import Robot
import numpy as np
import time
import os
import cv2
import random
import utils
import math

def save_vision_data(color_main,depth_main,color_single,depth_single):
    all_n = os.listdir('captured_image/')
    num = len(all_n)
    num = int(num/4)
    cv2.imwrite('captured_image/'+str(num+1)+'_colorMain.png',cv2.cvtColor(color_main, cv2.COLOR_RGB2BGR))
    cv2.imwrite('captured_image/' + str(num + 1) + '_colorSingle.png',cv2.cvtColor(color_single, cv2.COLOR_RGB2BGR))
    np.save('captured_image/' + str(num + 1) + '_depthMain.npy', depth_main)
    np.save('captured_image/' + str(num + 1) + '_depthSingle.npy', depth_single)
    #### get single obj size ####
    # ret, binary = cv2.threshold(cv2.cvtColor(color_single,cv2.COLOR_RGB2GRAY), 127, 255, cv2.THRESH_BINARY)
    binary = np.clip(depth_single-(depth_single[3][3]-0.01),0,255)
    binary = np.clip(binary*100000,0,255)
    binary = (binary -255)* -1
    cv2.imwrite('captured_image/' + str(num + 1) + '_maskSingle.png',binary)
    imim, contours, hierarchy = cv2.findContours(cv2.imread('captured_image/' + str(num + 1) + '_maskSingle.png',0), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_NONE,CHAIN_APPROX_SIMPLE
    w_x = 0
    h_y = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= 3 and h >= 3:
            w_x = w
            h_y = h
    z_margin = depth_single[5][5] - depth_single[64][64]
    return round(w_x*30/128), round(h_y*30/128), round(z_margin*100)

def check_if_obj_moved(robot,work_space_obj_handles,work_space_obj_pos):
    if len(work_space_obj_handles) == 0:
        return False
    is_moved = False
    now_pos = robot.find_obj_positions_by_handles(work_space_obj_handles)
    ind = 0
    for pos in work_space_obj_pos:
        pos_2 = now_pos[ind]
        dis = math.sqrt((pos[0]-pos_2[0])*(pos[0]-pos_2[0]) + (pos[1]-pos_2[1])*(pos[1]-pos_2[1]))
        # print('curr_dis:',dis)
        if dis > 3:
            print('curr_dis:', dis)
            print(pos,pos_2)
            is_moved = True
            break
        ind += 1
    return is_moved
def save_sample_data(pos,color_main,depth_main,color_single,depth_single,count):
    '''####
    if depth_type == 0:  # main
        depth_arr = np.load(npy_path)
        depth_arr = cv2.resize(depth_arr, (256, 256))
    else:
        depth_arr = np.load(npy_path)

    depth_arr = np.clip(np.round((depth_arr[0][0] - depth_arr) * 1000), 0, 255) / 255.0
    return np.stack((depth_arr, depth_arr, depth_arr))
    ####'''
    depth_main_arr = cv2.resize(depth_main, (256, 256))
    depth_main_arr = np.clip(np.round((depth_main_arr[0][0] - depth_main_arr) * 1000), 0, 255)
    depth_single = np.clip(np.round((depth_single[0][0] - depth_single) * 1000), 0, 255)

    wp, hp = utils.trans_label_arr_to_label_pix(pos)
    cv2.circle(color_main, (wp, hp), 3, (0, 0, 255), -1)
    cv2.imwrite('data/'+'%06d_main.png'%count,color_main)
    cv2.imwrite('data/' + '%06d_item.png' % count, color_single)
    cv2.imwrite('data/' + '%06d_depth_main.png' % count, depth_main_arr)
    cv2.imwrite('data/' + '%06d_depth_item.png' % count, depth_single)
    np.savetxt('data/' + '%06d_position.txt' % count, pos)
    # '%06d.png' % len(measured_pts)

workspace_limits = np.asarray([[-0.25, -0.85], [-0.3, 0.3], [0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
condidate_obj_space = np.asarray([[-0.35,0.35],[0.35+0.06666666,0.75-0.06666666]])#([[-0.35,0.35],[0.35,0.75]])

# ##############
# grid_x = np.linspace(condidate_obj_space[0][0], condidate_obj_space[0][1], 8)
# grid_y = np.linspace(condidate_obj_space[1][0], condidate_obj_space[1][1], 3)
# ##############
# count = 1
# for item1 in grid_x:
#     for item2 in grid_y:
#         print(str(count),', pos:',item1,item2)
#         count += 1


max_num_obj = 10
sample_count = 30000
for run_out in range(1000):
    if sample_count >= 40000:
        break

    ## remove tmp folder ##
    file_names = os.listdir('captured_image/')
    for fname in file_names:
        os.remove('captured_image/'+fname)

    work_space_obj_handles = []
    work_space_obj_pos = []
    stop_flag = False
    # Initialize pick-and-place system (camera and robot)
    robot = Robot(max_num_obj, workspace_limits)

    robot.move_to_new([-0.2, 0.2, 0.2])
    time.sleep(0.5)
    robot.add_assembled_objects_scene()
    time.sleep(1)
    print('--------------------------- start --------------------------------')
    ind = 0
    # sample_count = 0

    for item in robot.object_handles:
        robot.set_position_for_single_obj(item, [0,0.55,0.01])
        robot.set_angle_for_single_obj(item,(np.pi/2, 0 ,np.pi/2))
        color_main, depth_main = robot.get_camera_data()
        color_single, depth_single = robot.get_obj_camera_data()
        x,y,z = save_vision_data(color_main,depth_main,color_single,depth_single)
        # print(x,y,z)
        # robot.move_to_new([0, 0.55, 0.12])
        # robot.move_to_new([0, 0.55, z/100-0.002])
        # time.sleep(0.5)
        # robot.control_suctionPad(1)
        # time.sleep(0.5)
        # robot.move_to_new([0, 0.55, 0.12])
        x_m, y_m = robot.object_size[ind][0], robot.object_size[ind][1]
        if x_m == x and y_m == y:
            curr_rotate_angle = None
        else:
            curr_rotate_angle = (np.pi / 2, np.pi / 2, np.pi / 2)

        if not curr_rotate_angle == None:
            robot.set_angle_for_single_obj(item,curr_rotate_angle)
        time.sleep(0.2)
        
        ### 加误差 0.002 ##
        x_m = x_m +0.005
        y_m = y_m +0.005
        # save sample data #
        color_main, depth_main = robot.get_camera_data()
        color_single, depth_single = robot.get_obj_camera_data()
        ## ##
        if check_if_obj_moved(robot,work_space_obj_handles,work_space_obj_pos):
            stop_flag = True
            break
        save_sample_data([robot.object_positions[ind][0],robot.object_positions[ind][1],robot.object_positions[ind][2]-0.002], color_main, depth_main, color_single, depth_single, sample_count)#z/100-0.002
        print('obtain sample ',str(sample_count),' .....')


        #####
        # robot.set_position_for_single_obj(item,[robot.object_positions[ind][0],robot.object_positions[ind][1],robot.object_positions[ind][2]-0.0245])

        ### move to object ###
        robot.move_to_new([0, 0.55, 0.12])
        robot.move_to_new([0, 0.55, 5/100-0.002])
        time.sleep(0.5)
        robot.control_suctionPad(1)
        robot.move_to_new([0, 0.55, 0.12])
        print('suction_success: ', robot.check_suction())
        robot.move_to_new([-0.55, 0.22, 0.4])
        # time.sleep(0.5)


        ### move to box  ####
        robot.move_to_new([robot.object_positions[ind][0],robot.object_positions[ind][1],robot.object_positions[ind][2]+0.2])
        time.sleep(0.5)
        robot.move_to_new_low([robot.object_positions[ind][0],robot.object_positions[ind][1],robot.object_positions[ind][2]+0.005])#0.022 -0.0245   +0.01
        time.sleep(0.5)
        robot.control_suctionPad(0)
        time.sleep(0.5)
        robot.move_to_new_low([robot.object_positions[ind][0], robot.object_positions[ind][1],robot.object_positions[ind][2] + 0.05])
        # robot.rotate_gripper(0)
        # time.sleep(0.5)


        # robot.move_to_new_low([robot.object_positions[ind][0],robot.object_positions[ind][1],robot.object_positions[ind][2]+0.4])
        robot.move_to_new_low([-0.55,0,0.4])
        robot.move_to_new([-0.2, 0.2, 0.4])
        # time.sleep(0.5)

        work_space_obj_handles.append(item)
        work_space_obj_pos.append([robot.object_positions[ind][0],robot.object_positions[ind][1]])
        # time.sleep(0.5)

        ind += 1
        sample_count += 1

#     ##################### Disassemble #######################
#     if stop_flag:
#         continue
#     robot.move_to_new([-0.2,0.2,0.2])
#     time.sleep(0.5)

#     # all_position = robot.object_positions()
#     all_position_layer1 = []
#     all_position_layer2 = []
#     all_position_layer3 = []
#     # all_position_layer4 = []
#     all_handl_layer1 = []
#     all_handl_layer2 = []
#     all_handl_layer3 = []
#     # all_handl_layer4 = []
#     ind_list = random.sample(list(range(len(robot.object_positions))),len(robot.object_positions))
#     for p in ind_list:
#         if robot.object_positions[p][2] >= 0.01 and robot.object_positions[p][2] <= 0.06:
#             all_position_layer1.append(robot.object_positions[p])
#             all_handl_layer1.append(robot.object_handles[p])
#         elif robot.object_positions[p][2] > 0.06 and robot.object_positions[p][2] <= 0.11:
#             all_position_layer2.append(robot.object_positions[p])
#             all_handl_layer2.append(robot.object_handles[p])
#         elif robot.object_positions[p][2] > 0.11 and robot.object_positions[p][2] <= 0.16:
#             all_position_layer3.append(robot.object_positions[p])
#             all_handl_layer3.append(robot.object_handles[p])
#         # elif robot.object_positions[p][2] > 0.16 and robot.object_positions[p][2] <= 0.21:
#         #     all_position_layer4.append(robot.object_positions[p])
#         #     all_handl_layer4.append(robot.object_handles[p])

#         # all_position.append(robot.object_positions[p])
#         # all_handl.append(robot.object_handles[p])
#         # = random.sample(robot.object_positions(),len(robot.object_positions()))

#     cc = 0
#     all_handl = [all_handl_layer3,all_handl_layer2,all_handl_layer1]
#     for all_position in [all_position_layer3,all_position_layer2,all_position_layer1]:
#         if stop_flag:
#             break
#         n = 0
#         for pos in all_position:
#             robot.set_position_for_single_obj(all_handl[cc][n], [pos[0], pos[1], pos[2]+0.2])
#             robot.set_position_for_single_obj(all_handl[cc][n], [0, 0.55, 0.0255])
#             curr_angle = robot.get_single_obj_orientations(all_handl[cc][n])
#             if abs(curr_angle[1]) < 0.2:
#                 robot.set_angle_for_single_obj(all_handl[cc][n], (np.pi / 2, 0 , np.pi / 2))
#             else:
#                 robot.set_angle_for_single_obj(all_handl[cc][n], (np.pi / 2, np.pi / 2, np.pi / 2))
#             work_space_obj_handles.remove(all_handl[cc][n])
#             work_space_obj_pos.remove([pos[0], pos[1]])
#             time.sleep(0.5)
#             # save sample data #
#             color_main, depth_main = robot.get_camera_data()
#             color_single, depth_single = robot.get_obj_camera_data()
#             ## ##
#             if check_if_obj_moved(robot, work_space_obj_handles, work_space_obj_pos):
#                 stop_flag = True
#                 break
#             save_sample_data([pos[0], pos[1], 0.05 - 0.002], color_main, depth_main, color_single, depth_single, sample_count)  # z/100-0.002
#             print('obtain sample ', str(sample_count), ' .....')
#             #####

#             robot.set_position_for_single_obj(all_handl[cc][n], [0.5+np.random.randint(0,200)/1000, -np.random.randint(0,200)/1000, 0.3])
#             n += 1

#             sample_count += 1
#         cc += 1

    robot.restart_sim()
    ## remove tmp folder ##
    file_names = os.listdir('captured_image/')
    for fname in file_names:
        os.remove('captured_image/'+fname)

import time
import os
import numpy as np
import utils
# from simulation import sim
from simulation import vrep as sim
import random

class Robot(object):
    def __init__(self, max_num_obj, workspace_limits):

        self.workspace_limits = workspace_limits
        # Define colors for object meshes (Tableau palette)
        self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                       [89.0, 161.0, 79.0], # green
                                       [156, 117, 95], # brown
                                       [242, 142, 43], # orange
                                       [237.0, 201.0, 72.0], # yellow
                                       [186, 176, 172], # gray
                                       [255.0, 87.0, 89.0], # red
                                       [176, 122, 161], # purple
                                       [118, 183, 178], # cyan
                                       [255, 157, 167]])/255.0 #pink

        # Read files in object mesh directory
        self.obj_mesh_dir = 'objects/'
        self.max_num_obj = max_num_obj
        self.mesh_list = os.listdir(self.obj_mesh_dir)

        # Randomly choose objects to add to scene
        self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.max_num_obj)
        self.obj_mesh_color = self.color_space[np.asarray(range(self.max_num_obj)) % 10, :]

        # Connect to simulator
        sim.simxFinish(-1) # Just in case, close all opened connections
        self.sim_client = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')
            self.restart_sim()

        sim.simxStartSimulation(self.sim_client, sim.simx_opmode_blocking)
        time.sleep(1)
        # Setup virtual camera in simulation
        self.setup_sim_camera()

        # Add objects to simulation environment
        # self.add_objects()

    def setup_sim_camera(self):

        # Get handle to camera
        sim_ret, self.cam_handle = sim.simxGetObjectHandle(self.sim_client, 'Vision_sensor_ortho', sim.simx_opmode_blocking)
        sim_ret, self.cam_handle_obj = sim.simxGetObjectHandle(self.sim_client, 'Vision_sensor_object',sim.simx_opmode_blocking)

        # Get camera pose and intrinsics in simulation
        sim_ret, cam_position = sim.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, sim.simx_opmode_blocking)
        sim_ret, cam_orientation = sim.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, sim.simx_opmode_blocking)
        cam_trans = np.eye(4,4)
        cam_trans[0:3,3] = np.asarray(cam_position)
        cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        cam_rotm = np.eye(4,4)
        cam_rotm[0:3,0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
        self.cam_pose = np.dot(cam_trans, cam_rotm) # Compute rigid transformation representating camera pose
        self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
        self.cam_depth_scale = 1

        # Get background image
        self.bg_color_img, self.bg_depth_img = self.get_camera_data()
        self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale


    def add_objects(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        # add all fixed object in file , random location , random orientation
        self.object_handles = []
        sim_obj_handles = []
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            curr_mesh_file = '/home/zjh/songshuai/Bin_Packing/' + curr_mesh_file
            print('curr_mesh_file:',curr_mesh_file)

            curr_shape_name = 'shape_%02d' % object_idx
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            # if self.is_testing and self.test_preset_cases:
            #     object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1], self.test_obj_positions[object_idx][2]]
            #     object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = sim.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',sim.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), sim.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)    
        self.prev_obj_positions = []
        self.obj_positions = []

    def add_objects_new(self):

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        # add all fixed object in file , random location , random orientation

        self.object_handles = []
        sim_obj_handles = []
        for object_idx in range(len(self.obj_mesh_ind)):
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
            curr_mesh_file = '/home/zjh/songshuai/Bin_Packing/' + curr_mesh_file
            print('curr_mesh_file:',curr_mesh_file)

            curr_shape_name = 'shape_%02d' % object_idx
            drop_x =0.6# (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y =-0.2# (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            # if self.is_testing and self.test_preset_cases:
            #     object_position = [self.test_obj_positions[object_idx][0], self.test_obj_positions[object_idx][1], self.test_obj_positions[object_idx][2]]
            #     object_orientation = [self.test_obj_orientations[object_idx][0], self.test_obj_orientations[object_idx][1], self.test_obj_orientations[object_idx][2]]
            object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = sim.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',sim.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), sim.simx_opmode_blocking)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
        self.prev_obj_positions = []
        self.obj_positions = []

    

    def set_random_x_size(self):
        x_size = []
        x_size_1 = random.sample([5, 10, 15, 20], 1)[0]
        x_size_2 = random.sample([5, 10, 15, 20], 1)[0]
        x_size_3 = random.sample([5, 10, 15, 20], 1)[0]
        x_size_4 = random.sample([5, 10, 15, 20], 1)[0]
        x_size_5 = random.sample([5, 10, 15, 20], 1)[0]
        x_size_6 = random.sample([5, 10, 15, 20], 1)[0]
        x_size_7 = random.sample([5, 10, 15, 20], 1)[0]
        x_size_8 = random.sample([5, 10, 15, 20], 1)[0]
        x_size.append(x_size_1)
        x_size.append(x_size_2)
        if (x_size_1+x_size_2+x_size_3) <= 40:
            x_size.append(x_size_3)
        if (x_size_1+x_size_2+x_size_3+x_size_4) <= 40:
            x_size.append(x_size_4)
        if (x_size_1+x_size_2+x_size_3+x_size_4+x_size_5) <= 40:
            x_size.append(x_size_5)
        if (x_size_1+x_size_2+x_size_3+x_size_4+x_size_5+x_size_6) <= 40:
            x_size.append(x_size_6)
        if (x_size_1+x_size_2+x_size_3+x_size_4+x_size_5+x_size_6+x_size_7) <= 40:
            x_size.append(x_size_7)
        if (x_size_1+x_size_2+x_size_3+x_size_4+x_size_5+x_size_6+x_size_7+x_size_8) <= 40:
            x_size.append(x_size_8)

        empty_margin = 40 - np.sum(np.array(x_size))
        # empty_margin_num = int(empty_margin / 5)
        if not empty_margin == 0:
            x_size.append(int(empty_margin))
        # if empty_margin_num >= 4:
        #     x_size_4 = random.sample([5, 10, 15, 20], 1)[0]
        #     x_size.append(x_size_4)
        #     empty_margin = 60 - x_size_1 - x_size_2 - x_size_3 - x_size_4
        #     empty_margin_num = int(empty_margin / 5)
        #     for i in range(empty_margin_num):
        #         x_size.append(5)
        # elif empty_margin_num < 4:
        #     for i in range(empty_margin_num):
        #         x_size.append(5)

        return x_size
    def set_position_for_assembled_obj(self):
        ind = 0
        for obj_handle in self.object_handles:
            self.set_position_for_single_obj(obj_handle,self.object_positions[ind])
            ind += 1
            time.sleep(0.5)
    def pro_mesh_grid(self,z_ind,flag,x_size,y_size,mesh_grid):
        '''
        flag: 0-top, 1-mid, 2-bottom
        '''
        y_range = [0,4]
        num = len(x_size)
        x_start = 0
        y_start = y_range[flag]
        for ind in range(num):
            for x in range(int(x_size[ind]/5)):
                x += x_start
                for y in range(int(y_size[ind]/5)):
                    y += y_start
                    mesh_grid[x][y][z_ind] = 1
            x_start = x_start + int(x_size[ind]/5)

        return mesh_grid
    def find_pos_in_mesh_grid(self,x,y,z,single_type):#0-single one,1-x with 2, 2-y with 2
        # workspace_limits = np.asarray([[-0.85, -0.25], [-0.3, 0.3], [0.0001, 0.4]])
        # workspace_limits = np.asarray([[-0.862, -0.238], [-0.312, 0.312], [0.0001, 0.4]])
        # workspace_limits = np.asarray([[-0.868, -0.232], [-0.318, 0.318], [0.0001, 0.4]])
        workspace_limits = np.asarray([[-0.762, -0.338], [-0.212, 0.212], [0.0001, 0.4]])
        x_pos_start = workspace_limits[0][0]+0.0265
        y_pos_start = workspace_limits[1][0]+0.0265
        if single_type == 0:
            x_pos = x_pos_start + x*0.053#0.05
            y_pos = y_pos_start + y*0.053
            z_pos = (z+1)*0.05
        elif single_type == 1:
            x_pos = x_pos_start + x * 0.053
            x_pos_next = x_pos_start + (x+1) * 0.053
            x_pos = (x_pos+x_pos_next)/2
            y_pos = y_pos_start + y * 0.053
            z_pos = (z + 1) * 0.05
        elif single_type == 2:
            x_pos = x_pos_start + x * 0.053
            # x_pos_next = x_pos_start + (x + 1) * 0.053
            # x_pos = (x_pos + x_pos_next) / 2
            y_pos = y_pos_start + y * 0.053
            y_pos_next = y_pos_start + (y + 1) * 0.053
            y_pos = (y_pos + y_pos_next) / 2
            z_pos = (z + 1) * 0.05
        return [x_pos,y_pos,z_pos]



    def add_assembled_objects_scene(self):
        '''

        :return:
        '''
        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        object_idx = 0
        workspace_limits = np.asarray([[-0.762, -0.338], [-0.212, 0.212], [0.0001, 0.4]])
        mesh_grid = np.zeros((8,8,3)) # 3 layer is max

        self.object_handles = []
        self.object_positions = []
        self.object_size = []
        sim_obj_handles = []
        base_path = '/home/zjh/songshuai/Bin_Packing/objects/'#'/home/ys/2020_work/assembling_task/v2/objects/'#E:\phd_workspace\v2\objects

        for z_ind in range(3):
            ##### gen random size for top row #####
            x_size = self.set_random_x_size()
            y_size_1 = self.set_random_x_size()
            y_size_2 = self.set_random_x_size()
            y_size_3 = self.set_random_x_size()
            random_x_size_top = random.sample(x_size,len(x_size))
            random_y_size_top = random.sample(y_size_1+y_size_2+y_size_3,len(x_size))
            random_z_size_top = []
            ### 产生高度z相同的object , 都为5cm , 并随机散落###
            for i in range(len(x_size)):
                if np.random.randint(0,1000)%2 == 0:
                    random_z_size_top.append(5)
                else:
                    random_z_size_top.append(5)
            mesh_grid = self.pro_mesh_grid(z_ind, 0, random_x_size_top, random_y_size_top, mesh_grid)

            ##### gen random size for mid row #####
            x_size = self.set_random_x_size()
            y_size_1 = self.set_random_x_size()
            y_size_2 = self.set_random_x_size()
            y_size_3 = self.set_random_x_size()
            random_x_size_mid = random.sample(x_size,len(x_size))
            random_y_size_mid = random.sample(y_size_1+y_size_2+y_size_3,len(x_size))
            random_z_size_mid = []
            for i in range(len(x_size)):
                if np.random.randint(0, 1000) % 2 == 0:
                    random_z_size_mid.append(5)
                else:
                    random_z_size_mid.append(5)
            mesh_grid = self.pro_mesh_grid(z_ind, 1, random_x_size_mid, random_y_size_mid, mesh_grid)

            # ##### gen random size for bottom row #####
            # x_size = self.set_random_x_size()
            # y_size_1 = self.set_random_x_size()
            # y_size_2 = self.set_random_x_size()
            # y_size_3 = self.set_random_x_size()
            # random_x_size_bottom = random.sample(x_size, len(x_size))
            # random_y_size_bottom = random.sample(y_size_1+y_size_2+y_size_3, len(x_size))
            # random_z_size_bottom = []
            # for i in range(len(x_size)):
            #     if np.random.randint(0, 1000) % 2 == 0:
            #         random_z_size_bottom.append(5)
            #     else:
            #         random_z_size_bottom.append(5)
            # mesh_grid = self.pro_mesh_grid(z_ind, 2, random_x_size_bottom, random_y_size_bottom, mesh_grid)
            # # print(mesh_grid[:,:,0])

            # <editor-fold desc="begin to add objects for top row">
            x_add = 0
            ind = 0
            for n1 in range(len(random_x_size_top)):
                x_start = workspace_limits[0][0] + x_add/100
                y_start = workspace_limits[1][0]
                x_m = random_x_size_top[n1]
                y_m = random_y_size_top[n1]
                z_m = random_z_size_top[n1]
                self.object_size.append([x_m,y_m,z_m])
                x_add = x_add + (x_m/5*5.3)
                obj_name = 'x'+str(x_m)+'_y'+str(y_m)+'_z'+str(z_m)+'.obj'
                # set_pos = [x_start + x_m/200 + 0.002*ind, y_start+y_m/200, z_m/200]
                # set_pos = [x_start + x_m / 200 + 0.002 * ind, y_start + y_m / 200, z_m / 200]
                set_pos = [x_start + (x_m/5*5.3/200), y_start + (y_m/5*5.3/200), z_m / 100 *(z_ind+1)]

                curr_mesh_file = base_path + obj_name
                print('curr_mesh_file:',curr_mesh_file)
                curr_shape_name = 'shape_%02d' % object_idx
                print(curr_shape_name,', add obj:',obj_name)
                drop_x = 0.6#(self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
                drop_y = -0.2#(self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
                object_position = [drop_x, drop_y, 0.15]
                object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]

                object_color = [self.obj_mesh_color[object_idx%10][0], self.obj_mesh_color[object_idx%10][1], self.obj_mesh_color[object_idx%10][2]]
                ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = sim.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',sim.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), sim.simx_opmode_blocking)
                if ret_resp == 8:
                    print('Failed to add new objects to simulation. Please restart.')
                    exit()
                curr_shape_handle = ret_ints[0]
                self.object_handles.append(curr_shape_handle)
                self.object_positions.append(set_pos)
                # time.sleep(1)
                # self.set_position_for_single_obj(curr_shape_handle,set_pos)
                object_idx += 1
                ind += 1
            # </editor-fold>

            # <editor-fold desc="add object to the blank positions in top row">
            tmp_obj_size = []
            tmp_obj_position = []
            for xnd in range(8):
                for ynd in [0,1,2,3]:
                    if mesh_grid[xnd][ynd][z_ind] == 0:
                        if np.random.randint(0,1000)%2 == 0:
                            if xnd < 7 and mesh_grid[xnd+1][ynd][z_ind] == 0:
                                tmp_obj_size.append([10,5,5])
                                tmp_obj_position.append(self.find_pos_in_mesh_grid(xnd,ynd,z_ind,1))
                                mesh_grid[xnd][ynd][z_ind] = 1
                                mesh_grid[xnd + 1][ynd][z_ind] = 1
                            else:
                                tmp_obj_size.append([5, 5, 5])
                                tmp_obj_position.append(self.find_pos_in_mesh_grid(xnd, ynd, z_ind, 0))
                                mesh_grid[xnd][ynd][z_ind] = 1
                        else:
                            if ynd < 3 and mesh_grid[xnd][ynd+1][z_ind] == 0:
                                tmp_obj_size.append([5, 10, 5])
                                tmp_obj_position.append(self.find_pos_in_mesh_grid(xnd, ynd, z_ind, 2))
                                mesh_grid[xnd][ynd][z_ind] = 1
                                mesh_grid[xnd][ynd + 1][z_ind] = 1
                            else:
                                tmp_obj_size.append([5, 5, 5])
                                tmp_obj_position.append(self.find_pos_in_mesh_grid(xnd, ynd, z_ind, 0))
                                mesh_grid[xnd][ynd][z_ind] = 1
            # begin to add
            ind = 0
            for no in range(len(tmp_obj_size)):
                x_m = tmp_obj_size[no][0]
                y_m = tmp_obj_size[no][1]
                z_m = tmp_obj_size[no][2]
                self.object_size.append([x_m, y_m, z_m])
                obj_name = 'x' + str(x_m) + '_y' + str(y_m) + '_z' + str(z_m) + '.obj'
                set_pos = [tmp_obj_position[no][0]+0.002*ind,tmp_obj_position[no][1],tmp_obj_position[no][2]]#[x_start + x_m / 200 + 0.002 * ind, y_start + y_m / 200, z_m / 200]

                curr_mesh_file = base_path + obj_name
                curr_shape_name = 'shape_%02d' % object_idx
                print(curr_shape_name, ', add obj:', obj_name)
                drop_x = 0.6  # (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
                drop_y = -0.2  # (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
                object_position = [drop_x, drop_y, 0.15]
                object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                      2 * np.pi * np.random.random_sample()]

                object_color = [self.obj_mesh_color[object_idx % 10][0], self.obj_mesh_color[object_idx % 10][1],
                                self.obj_mesh_color[object_idx % 10][2]]
                ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = sim.simxCallScriptFunction(self.sim_client,
                                                                                                     'remoteApiCommandServer',
                                                                                                     sim.sim_scripttype_childscript,
                                                                                                     'importShape',
                                                                                                     [0, 0, 255, 0],
                                                                                                     object_position + object_orientation + object_color,
                                                                                                     [curr_mesh_file,
                                                                                                      curr_shape_name],
                                                                                                     bytearray(),
                                                                                                     sim.simx_opmode_blocking)
                if ret_resp == 8:
                    print('Failed to add new objects to simulation. Please restart.')
                    exit()
                curr_shape_handle = ret_ints[0]
                self.object_handles.append(curr_shape_handle)
                self.object_positions.append(set_pos)
                # time.sleep(1)
                # self.set_position_for_single_obj(curr_shape_handle,set_pos)
                object_idx += 1
                # ind += 1
            # </editor-fold>

            # <editor-fold desc="begin to add objects for mid row">
            x_add = 0
            ind = 0
            y_start = workspace_limits[1][0] + (5.3*4/100)
            for n2 in range(len(random_x_size_mid)):
                x_start = workspace_limits[0][0] + x_add/100
                x_m = random_x_size_mid[n2]
                y_m = random_y_size_mid[n2]
                z_m = random_z_size_mid[n2]
                self.object_size.append([x_m, y_m, z_m])
                x_add = x_add + (x_m/5*5.3)
                obj_name = 'x'+str(x_m)+'_y'+str(y_m)+'_z'+str(z_m)+'.obj'
                set_pos = [x_start + (x_m/5*5.3/200), y_start + (y_m/5*5.3/200), z_m / 100 *(z_ind+1)]

                curr_mesh_file = base_path + obj_name
                curr_shape_name = 'shape_%02d' % object_idx
                print(curr_shape_name, ', add obj:', obj_name)
                drop_x = 0.6#(self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
                drop_y = -0.2#(self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
                object_position = [drop_x, drop_y, 0.15]
                object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]

                object_color = [self.obj_mesh_color[object_idx%10][0], self.obj_mesh_color[object_idx%10][1], self.obj_mesh_color[object_idx%10][2]]
                ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = sim.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',sim.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), sim.simx_opmode_blocking)
                if ret_resp == 8:
                    print('Failed to add new objects to simulation. Please restart.')
                    exit()
                curr_shape_handle = ret_ints[0]
                self.object_handles.append(curr_shape_handle)
                self.object_positions.append(set_pos)
                # time.sleep(1)
                # self.set_position_for_single_obj(curr_shape_handle,set_pos)
                object_idx += 1
                ind += 1
            # </editor-fold>

            # <editor-fold desc="add object to the blank positions in mid row">
            tmp_obj_size = []
            tmp_obj_position = []
            for xnd in range(8):
                for ynd in [4, 5, 6, 7]:
                    if mesh_grid[xnd][ynd][z_ind] == 0:
                        if np.random.randint(0, 1000) % 2 == 0:
                            if xnd < 7 and mesh_grid[xnd + 1][ynd][z_ind] == 0:
                                tmp_obj_size.append([10, 5, 5])
                                tmp_obj_position.append(self.find_pos_in_mesh_grid(xnd, ynd, z_ind, 1))
                                mesh_grid[xnd][ynd][z_ind] = 1
                                mesh_grid[xnd + 1][ynd][z_ind] = 1
                            else:
                                tmp_obj_size.append([5, 5, 5])
                                tmp_obj_position.append(self.find_pos_in_mesh_grid(xnd, ynd, z_ind, 0))
                                mesh_grid[xnd][ynd][z_ind] = 1
                        else:
                            if ynd < 7 and mesh_grid[xnd][ynd + 1][z_ind] == 0:
                                tmp_obj_size.append([5, 10, 5])
                                tmp_obj_position.append(self.find_pos_in_mesh_grid(xnd, ynd, z_ind, 2))
                                mesh_grid[xnd][ynd][z_ind] = 1
                                mesh_grid[xnd][ynd + 1][z_ind] = 1
                            else:
                                tmp_obj_size.append([5, 5, 5])
                                tmp_obj_position.append(self.find_pos_in_mesh_grid(xnd, ynd, z_ind, 0))
                                mesh_grid[xnd][ynd][z_ind] = 1
            # begin to add
            ind = 0
            for no in range(len(tmp_obj_size)):
                x_m = tmp_obj_size[no][0]
                y_m = tmp_obj_size[no][1]
                z_m = tmp_obj_size[no][2]
                self.object_size.append([x_m, y_m, z_m])
                obj_name = 'x' + str(x_m) + '_y' + str(y_m) + '_z' + str(z_m) + '.obj'
                set_pos = [tmp_obj_position[no][0] + 0.002 * ind, tmp_obj_position[no][1], tmp_obj_position[no][
                    2]]  # [x_start + x_m / 200 + 0.002 * ind, y_start + y_m / 200, z_m / 200]

                curr_mesh_file = base_path + obj_name
                curr_shape_name = 'shape_%02d' % object_idx
                print(curr_shape_name, ', add obj:', obj_name)
                drop_x = 0.6  # (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
                drop_y = -0.2  # (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
                object_position = [drop_x, drop_y, 0.15]
                object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
                                      2 * np.pi * np.random.random_sample()]

                object_color = [self.obj_mesh_color[object_idx % 10][0], self.obj_mesh_color[object_idx % 10][1],
                                self.obj_mesh_color[object_idx % 10][2]]
                ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = sim.simxCallScriptFunction(
                    self.sim_client,
                    'remoteApiCommandServer',
                    sim.sim_scripttype_childscript,
                    'importShape',
                    [0, 0, 255, 0],
                    object_position + object_orientation + object_color,
                    [curr_mesh_file,
                     curr_shape_name],
                    bytearray(),
                    sim.simx_opmode_blocking)
                if ret_resp == 8:
                    print('Failed to add new objects to simulation. Please restart.')
                    exit()
                curr_shape_handle = ret_ints[0]
                self.object_handles.append(curr_shape_handle)
                self.object_positions.append(set_pos)
                # time.sleep(1)
                # self.set_position_for_single_obj(curr_shape_handle,set_pos)
                object_idx += 1
                # ind += 1
            # </editor-fold>

            # # <editor-fold desc="begin to add objects for bottom row">
            # x_add = 0
            # ind = 0
            # # y_start = workspace_limits[1][0] + 0.404
            # y_start = workspace_limits[1][0] + (5.3 * 8 / 100)
            # for n3 in range(len(random_x_size_bottom)):
            #     x_start = workspace_limits[0][0] + x_add/100
            #     x_m = random_x_size_bottom[n3]
            #     y_m = random_y_size_bottom[n3]
            #     z_m = random_z_size_bottom[n3]
            #     self.object_size.append([x_m, y_m, z_m])
            #     x_add = x_add + (x_m/5*5.3)
            #     obj_name = 'x' + str(x_m) + '_y' + str(y_m) + '_z' + str(z_m) + '.obj'
            #     # set_pos = [x_start + x_m / 200 + 0.002*ind, y_start + y_m / 200, z_m / 200]
            #     set_pos = [x_start + (x_m / 5 * 5.3 / 200), y_start + (y_m / 5 * 5.3 / 200), z_m / 100 *(z_ind+1)]
            #
            #     curr_mesh_file = base_path + obj_name
            #     curr_shape_name = 'shape_%02d' % object_idx
            #     print(curr_shape_name, ', add obj:', obj_name)
            #     drop_x = 0.6  # (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            #     drop_y = -0.2  # (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            #     object_position = [drop_x, drop_y, 0.15]
            #     object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
            #                           2 * np.pi * np.random.random_sample()]
            #
            #     object_color = [self.obj_mesh_color[object_idx % 10][0], self.obj_mesh_color[object_idx % 10][1],
            #                     self.obj_mesh_color[object_idx % 10][2]]
            #     ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = sim.simxCallScriptFunction(self.sim_client,'remoteApiCommandServer',sim.sim_scripttype_childscript,'importShape',[0, 0, 255, 0],object_position + object_orientation + object_color,[curr_mesh_file,curr_shape_name],bytearray(),sim.simx_opmode_blocking)
            #     if ret_resp == 8:
            #         print('Failed to add new objects to simulation. Please restart.')
            #         exit()
            #     curr_shape_handle = ret_ints[0]
            #     self.object_handles.append(curr_shape_handle)
            #     self.object_positions.append(set_pos)
            #     # time.sleep(1)
            #     # self.set_position_for_single_obj(curr_shape_handle, set_pos)
            #     object_idx += 1
            #     ind += 1
            # # </editor-fold>

            # # <editor-fold desc="add object to the blank positions in bottom row">
            # tmp_obj_size = []
            # tmp_obj_position = []
            # for xnd in range(12):
            #     for ynd in [8, 9, 10, 11]:
            #         if mesh_grid[xnd][ynd][z_ind] == 0:
            #             if xnd < 11 and mesh_grid[xnd + 1][ynd][z_ind] == 0:
            #                 tmp_obj_size.append([10, 5, 5])
            #                 tmp_obj_position.append(self.find_pos_in_mesh_grid(xnd, ynd, z_ind, False))
            #                 mesh_grid[xnd][ynd][z_ind] = 1
            #                 mesh_grid[xnd + 1][ynd][z_ind] = 1
            #             else:
            #                 tmp_obj_size.append([5, 5, 5])
            #                 tmp_obj_position.append(self.find_pos_in_mesh_grid(xnd, ynd, z_ind, True))
            #                 mesh_grid[xnd][ynd][z_ind] = 1
            # # begin to add
            # ind = 0
            # for no in range(len(tmp_obj_size)):
            #     x_m = tmp_obj_size[no][0]
            #     y_m = tmp_obj_size[no][1]
            #     z_m = tmp_obj_size[no][2]
            #     self.object_size.append([x_m, y_m, z_m])
            #     obj_name = 'x' + str(x_m) + '_y' + str(y_m) + '_z' + str(z_m) + '.obj'
            #     set_pos = [tmp_obj_position[no][0] + 0.002 * ind, tmp_obj_position[no][1], tmp_obj_position[no][
            #         2]]  # [x_start + x_m / 200 + 0.002 * ind, y_start + y_m / 200, z_m / 200]
            #
            #     curr_mesh_file = base_path + obj_name
            #     curr_shape_name = 'shape_%02d' % object_idx
            #     print(curr_shape_name, ', add obj:', obj_name)
            #     drop_x = 0.6  # (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            #     drop_y = -0.2  # (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            #     object_position = [drop_x, drop_y, 0.15]
            #     object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(),
            #                           2 * np.pi * np.random.random_sample()]
            #
            #     object_color = [self.obj_mesh_color[object_idx % 10][0], self.obj_mesh_color[object_idx % 10][1],
            #                     self.obj_mesh_color[object_idx % 10][2]]
            #     ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = sim.simxCallScriptFunction(
            #         self.sim_client,
            #         'remoteApiCommandServer',
            #         sim.sim_scripttype_childscript,
            #         'importShape',
            #         [0, 0, 255, 0],
            #         object_position + object_orientation + object_color,
            #         [curr_mesh_file,
            #          curr_shape_name],
            #         bytearray(),
            #         sim.simx_opmode_blocking)
            #     if ret_resp == 8:
            #         print('Failed to add new objects to simulation. Please restart.')
            #         exit()
            #     curr_shape_handle = ret_ints[0]
            #     self.object_handles.append(curr_shape_handle)
            #     self.object_positions.append(set_pos)
            #     # time.sleep(1)
            #     # self.set_position_for_single_obj(curr_shape_handle,set_pos)
            #     object_idx += 1
            #     # ind += 1
            # # </editor-fold>

        # for obj_name in random_obj_name_list:
        #     name = obj_name.replace('.obj','').replace('x','').replace('y','').replace('z','')
        #     size_str = name.split('_')
        #     x_size = int(size_str[0])
        #     y_size = int(size_str[1])
        #     z_size = int(size_str[2])
        #     curr_mesh_file = 'F:/2020_work/assembling_task/v2/objects/' + obj_name
        #     curr_shape_name = 'shape_%02d' % object_idx
        #     drop_x = 0#(self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
        #     drop_y = 0.55#(self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
        #     object_position = [drop_x, drop_y, 0.15]
        #     object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
        #
        #     object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
        #     ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = sim.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',sim.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), sim.simx_opmode_blocking)
        #     if ret_resp == 8:
        #         print('Failed to add new objects to simulation. Please restart.')
        #         exit()
        #     curr_shape_handle = ret_ints[0]
        #     self.object_handles.append(curr_shape_handle)
        #     time.sleep(0.5)
        #     self.set_position_for_single_obj(curr_shape_handle,)
        # self.prev_obj_positions = []
        # self.obj_positions = []

    def restart_sim(self):
        sim_ret, self.UR5_joint1_handle = sim.simxGetObjectHandle(self.sim_client, 'UR5_joint1',
                                                                   sim.simx_opmode_blocking)
        sim_ret, self.UR5_joint2_handle = sim.simxGetObjectHandle(self.sim_client, 'UR5_joint2',
                                                                   sim.simx_opmode_blocking)
        sim_ret, self.UR5_joint3_handle = sim.simxGetObjectHandle(self.sim_client, 'UR5_joint3',
                                                                   sim.simx_opmode_blocking)
        sim_ret, self.UR5_joint4_handle = sim.simxGetObjectHandle(self.sim_client, 'UR5_joint4',
                                                                   sim.simx_opmode_blocking)
        sim_ret, self.UR5_joint5_handle = sim.simxGetObjectHandle(self.sim_client, 'UR5_joint5',
                                                                   sim.simx_opmode_blocking)
        sim_ret, self.UR5_joint6_handle = sim.simxGetObjectHandle(self.sim_client, 'UR5_joint6',
                                                                   sim.simx_opmode_blocking)
        sim_ret, self.UR5_target_handle = sim.simxGetObjectHandle(self.sim_client,'UR5_target',sim.simx_opmode_blocking)
        sim.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), sim.simx_opmode_blocking)
        sim.simxStopSimulation(self.sim_client, sim.simx_opmode_blocking)
        sim.simxStartSimulation(self.sim_client, sim.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = sim.simxGetObjectHandle(self.sim_client, 'UR5_tip', sim.simx_opmode_blocking)
        # sim_ret, gripper_position = sim.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, sim.simx_opmode_blocking)
        # while gripper_position[2] > 0.4: # V-REP bug requiring multiple starts and stops to restart
        #     sim.simxStopSimulation(self.sim_client, sim.simx_opmode_blocking)
        #     sim.simxStartSimulation(self.sim_client, sim.simx_opmode_blocking)
        #     time.sleep(1)
        #     sim_ret, gripper_position = sim.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, sim.simx_opmode_blocking)


    def check_sim(self):

        # Check if simulation is stable by checking if gripper is within workspace
        sim_ret, gripper_position = sim.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, sim.simx_opmode_blocking)
        sim_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and gripper_position[0] < self.workspace_limits[0][1] + 0.1 and gripper_position[1] > self.workspace_limits[1][0] - 0.1 and gripper_position[1] < self.workspace_limits[1][1] + 0.1 and gripper_position[2] > self.workspace_limits[2][0] and gripper_position[2] < self.workspace_limits[2][1]
        if not sim_ok:
            print('Simulation unstable. Restarting environment.')
            self.restart_sim()
            self.add_objects()


    def get_obj_positions(self):

        obj_positions = []
        for object_handle in self.object_handles:
            sim_ret, object_position = sim.simxGetObjectPosition(self.sim_client, object_handle, -1, sim.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def find_obj_positions_by_handles(self, all_handles):
        obj_positions = []
        for object_handle in all_handles:
            sim_ret, object_position = sim.simxGetObjectPosition(self.sim_client, object_handle, -1, sim.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions

    def get_obj_positions_and_orientations(self):

        obj_positions = []
        obj_orientations = []
        for object_handle in self.object_handles:
            sim_ret, object_position = sim.simxGetObjectPosition(self.sim_client, object_handle, -1, sim.simx_opmode_blocking)
            sim_ret, object_orientation = sim.simxGetObjectOrientation(self.sim_client, object_handle, -1, sim.simx_opmode_blocking)
            obj_positions.append(object_position)
            obj_orientations.append(object_orientation)

        return obj_positions, obj_orientations


    def reposition_objects(self, workspace_limits):

        # Move gripper out of the way
        self.move_to([-0.1, 0, 0.3], None)
        # sim_ret, UR5_target_handle = sim.simxGetObjectHandle(self.sim_client,'UR5_target',sim.simx_opmode_blocking)
        # sim.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), sim.simx_opmode_blocking)
        # time.sleep(1)

        for object_handle in self.object_handles:

            # Drop object at random x,y location and random orientation in robot workspace
            drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
            drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, 0.15]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            sim.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, sim.simx_opmode_blocking)
            sim.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, sim.simx_opmode_blocking)
            time.sleep(2)


    def get_camera_data(self):
        # Get color image from simulation
        sim_ret, resolution, raw_image = sim.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, sim.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = sim.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, sim.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear

        return color_img, depth_img

    def get_obj_camera_data(self):
        # Get color image from simulation
        sim_ret, resolution, raw_image = sim.simxGetVisionSensorImage(self.sim_client, self.cam_handle_obj, 0, sim.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        sim_ret, resolution, depth_buffer = sim.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle_obj, sim.simx_opmode_blocking)
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 10
        depth_img = depth_img * (zFar - zNear) + zNear

        return color_img, depth_img
    # def close_gripper(self, asynch=False):
    #
    #     if self.is_sim:
    #         gripper_motor_velocity = -0.5
    #         gripper_motor_force = 100
    #         sim_ret, RG2_gripper_handle = sim.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', sim.simx_opmode_blocking)
    #         sim_ret, gripper_joint_position = sim.simxGetJointPosition(self.sim_client, RG2_gripper_handle, sim.simx_opmode_blocking)
    #         sim.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, sim.simx_opmode_blocking)
    #         sim.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, sim.simx_opmode_blocking)
    #         gripper_fully_closed = False
    #         while gripper_joint_position > -0.045: # Block until gripper is fully closed
    #             sim_ret, new_gripper_joint_position = sim.simxGetJointPosition(self.sim_client, RG2_gripper_handle, sim.simx_opmode_blocking)
    #             # print(gripper_joint_position)
    #             if new_gripper_joint_position >= gripper_joint_position:
    #                 return gripper_fully_closed
    #             gripper_joint_position = new_gripper_joint_position
    #         gripper_fully_closed = True
    #
    #     else:
    #         self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #         self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
    #         tcp_command = "set_digital_out(8,True)\n"
    #         self.tcp_socket.send(str.encode(tcp_command))
    #         self.tcp_socket.close()
    #         if asynch:
    #             gripper_fully_closed = True
    #         else:
    #             time.sleep(1.5)
    #             gripper_fully_closed =  self.check_grasp()
    #
    #     return gripper_fully_closed
    #
    # def open_gripper(self, asynch=False):
    #
    #     if self.is_sim:
    #         gripper_motor_velocity = 0.5
    #         gripper_motor_force = 20
    #         sim_ret, RG2_gripper_handle = sim.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', sim.simx_opmode_blocking)
    #         sim_ret, gripper_joint_position = sim.simxGetJointPosition(self.sim_client, RG2_gripper_handle, sim.simx_opmode_blocking)
    #         sim.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, sim.simx_opmode_blocking)
    #         sim.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, sim.simx_opmode_blocking)
    #         while gripper_joint_position < 0.03: # Block until gripper is fully open
    #             sim_ret, gripper_joint_position = sim.simxGetJointPosition(self.sim_client, RG2_gripper_handle, sim.simx_opmode_blocking)
    #
    #     else:
    #         self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #         self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
    #         tcp_command = "set_digital_out(8,False)\n"
    #         self.tcp_socket.send(str.encode(tcp_command))
    #         self.tcp_socket.close()
    #         if not asynch:
    #             time.sleep(1.5)
    def check_suction(self):
        z_force = self.get_force_sensor_data()[2]
        return z_force > 0.5

    def control_suctionPad(self,flag):
        '''
        :param flag, 1 for open and 0 for close
        :return:
        '''
        # # suctionPad = sim.getObjectHandle("suctionPad")
        # sim_ret, suctionPad = sim.simxGetObjectHandle(self.sim_client, 'suctionPad',sim.simx_opmode_blocking)
        # suctionPadScript = sim.getScriptAssociatedWithObject(suctionPad)
        # sim.setUserParameter(suctionPadScript, "active", flag)
        # sim_ret = sim.simxSetIntegerSignal(self.sim_client,'suction_control',flag,sim.simx_opmode_oneshot)
        sim.simxSetIntegerSignal(self.sim_client, 'suctionActive', flag, sim.simx_opmode_blocking)
        time.sleep(1)
        # print('suction_success: ', self.check_suction())

    def get_force_sensor_data(self):
        sensorName = 'UR5_connection'
        errorCode, forceSensorHandle = sim.simxGetObjectHandle(self.sim_client, sensorName,sim.simx_opmode_blocking)
        errorCode, state, forceVector, torqueVector = sim.simxReadForceSensor(self.sim_client, forceSensorHandle, sim.simx_opmode_streaming)
        # print('forceVec:',forceVector)
        # print('torqueVec:',torqueVector)
        return forceVector

    def move_to(self, tool_position, tool_orientation):

        # sim_ret, UR5_target_handle = sim.simxGetObjectHandle(self.sim_client,'UR5_target',sim.simx_opmode_blocking)
        sim_ret, UR5_target_position = sim.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,sim.simx_opmode_blocking)

        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.04*move_direction/move_magnitude #0.02
        num_move_steps = int(np.floor(move_magnitude/0.02))

        for step_iter in range(num_move_steps):
            sim.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),sim.simx_opmode_blocking)
            sim_ret, UR5_target_position = sim.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,sim.simx_opmode_blocking)
        sim.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),sim.simx_opmode_blocking)

    def move_to_new(self,tool_position):

        sim_ret, UR5_target_position = sim.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, sim.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05 * move_direction / move_magnitude  # 0.05  0.01

        if np.floor(move_direction[0] / move_step[0]) == np.floor(move_direction[0] / move_step[0]):
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))
        else:
            num_move_steps = 0
        ###
        # if move_step[0] != 0:
        #     num_move_steps = int(np.floor(move_direction[0] / move_step[0]))
        # elif move_step[1] != 0:
        #     num_move_steps = int(np.floor(move_direction[1] / move_step[1]))
        # elif move_step[2] != 0:
        #     num_move_steps = int(np.floor(move_direction[2] / move_step[2]))
        ###

        # Compute gripper orientation and rotation increments
        tool_rotation_angle = 0
        sim_ret, gripper_orientation = sim.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, sim.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            sim.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), sim.simx_opmode_blocking)
            # sim.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
            #     np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
            #                               sim.simx_opmode_blocking)

        sim.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                   (tool_position[0], tool_position[1], tool_position[2]), sim.simx_opmode_blocking)
        # sim.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
        #                               (np.pi / 2, tool_rotation_angle, np.pi / 2), sim.simx_opmode_blocking)

    def move_to_new_low(self,tool_position):

        sim_ret, UR5_target_position = sim.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, sim.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1],tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.03 * move_direction / move_magnitude  # 0.05  0.01

        if np.floor(move_direction[0] / move_step[0]) == np.floor(move_direction[0] / move_step[0]):
            num_move_steps = int(np.floor(move_direction[0] / move_step[0]))
        else:
            num_move_steps = 0
        ###
        # if move_step[0] != 0:
        #     num_move_steps = int(np.floor(move_direction[0] / move_step[0]))
        # elif move_step[1] != 0:
        #     num_move_steps = int(np.floor(move_direction[1] / move_step[1]))
        # elif move_step[2] != 0:
        #     num_move_steps = int(np.floor(move_direction[2] / move_step[2]))
        ###

        # Compute gripper orientation and rotation increments
        tool_rotation_angle = 0
        sim_ret, gripper_orientation = sim.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, sim.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            sim.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (
                UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps),
                UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps),
                UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), sim.simx_opmode_blocking)
            # sim.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (
            #     np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2),
            #                               sim.simx_opmode_blocking)

        sim.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1,
                                   (tool_position[0], tool_position[1], tool_position[2]), sim.simx_opmode_blocking)
        # sim.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1,
        #                               (np.pi / 2, tool_rotation_angle, np.pi / 2), sim.simx_opmode_blocking)

    def rotate_gripper(self,rotation_angle):#absolute rotate
        tool_rotation_angle = (rotation_angle % np.pi) - np.pi / 2
        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = sim.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, sim.simx_opmode_blocking)
        rotation_step = 0.1 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.1#0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

        for step_iter in range(num_rotation_steps):
            sim.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), sim.simx_opmode_blocking)

        sim.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), sim.simx_opmode_blocking)

    def set_random_position(self):

        random_pos = []
        # gen random position
        condidate_obj_space = np.asarray([[-0.35, 0.35], [0.35 + 0.06666666, 0.75 - 0.06666666]])  # ([[-0.35,0.35],[0.35,0.75]])
        ##############
        grid_x = np.linspace(condidate_obj_space[0][0], condidate_obj_space[0][1], 8)
        grid_y = np.linspace(condidate_obj_space[1][0], condidate_obj_space[1][1], 3)
        ##############
        for item1 in grid_x:
            for item2 in grid_y:
                random_pos.append((item1, item2, 0.025))

        # ####### 随机生成十个旋转角度
        random_angles = []
        # angles = list(range(90))
        # random_an = random.sample(angles,10)
        for i in range(24):
        #     #random_angles.append((0,0,np.pi/2*(an_num/90.0)))
            random_angles.append((np.pi / 2, np.pi / 2 * (0 / 90.0),np.pi / 2))
        #     #random_angles.append((0, np.pi/2 , 0))
        #
        #     #random_angles.append((0, - np.pi, 0)) # hammar
        #
        #     #print ("ten random angles:",np.pi/2*(an_num/90.0))

        index = 0

        for object_handle in self.object_handles:
            '''随机位置'''
            sim.simxSetObjectPosition(self.sim_client, object_handle, -1, random_pos[index], sim.simx_opmode_blocking)

            '''随机角度'''
            sim.simxSetObjectOrientation(self.sim_client, object_handle, -1, random_angles[index],sim.simx_opmode_blocking)

            index+=1

    def get_object_size(self,obj_handle):
        '''
        sim.objfloatparam_objbbox_min_x (15): float parameter (can only be read) : object bounding box, min. x position (relative to the object reference frame)
        sim.objfloatparam_objbbox_min_y (16): float parameter (can only be read) : object bounding box, min. y position (relative to the object reference frame)
        sim.objfloatparam_objbbox_min_z (17): float parameter (can only be read) : object bounding box, min. z position (relative to the object reference frame)
        sim.objfloatparam_objbbox_max_x (18): float parameter (can only be read) : object bounding box, max. x position (relative to the object reference frame)
        sim.objfloatparam_objbbox_max_y (19): float parameter (can only be read) : object bounding box, max. y position (relative to the object reference frame)
        sim.objfloatparam_objbbox_max_z (20): float parameter (can only be read) : object bounding box, max. z position (relative to the object reference frame)
        :param obj_handle:
        :return:
        '''
        ret, min_x = sim.simxGetObjectFloatParameter (self.sim_client, obj_handle,15,sim.simx_opmode_blocking)
        ret, min_y = sim.simxGetObjectFloatParameter (self.sim_client, obj_handle,16,sim.simx_opmode_blocking)
        ret, min_z = sim.simxGetObjectFloatParameter (self.sim_client, obj_handle,17,sim.simx_opmode_blocking)
        ret, max_x = sim.simxGetObjectFloatParameter (self.sim_client, obj_handle,18,sim.simx_opmode_blocking)
        ret, max_y = sim.simxGetObjectFloatParameter (self.sim_client, obj_handle,19,sim.simx_opmode_blocking)
        ret, max_z = sim.simxGetObjectFloatParameter (self.sim_client, obj_handle,20,sim.simx_opmode_blocking)
        return  round((max_x-min_x)*100), round((max_y-min_y)*100), round((max_z-min_z)*100)

    def set_position_for_single_obj(self,obj_handle,position):

        '''随机位置'''
        sim.simxSetObjectPosition(self.sim_client, obj_handle, -1, position, sim.simx_opmode_blocking)

        '''随机角度'''
        # sim.simxSetObjectOrientation(self.sim_client, obj_handle, -1, (np.pi/2, 0 ,np.pi/2), sim.simx_opmode_blocking)
        # sim.simxSetObjectOrientation(self.sim_client, obj_handle, -1, (0, 0, 0), sim.simx_opmode_blocking)

    def set_angle_for_single_obj(self,obj_handle,angle):

        '''随机角度'''
        # sim.simxSetObjectOrientation(self.sim_client, obj_handle, -1, (np.pi/2, 0 ,np.pi/2), sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(self.sim_client, obj_handle, -1, angle, sim.simx_opmode_blocking)
    def resize_object(self,obj_handle,len_x,len_y,len_z):
        '''
        sim.objfloatparam_objbbox_min_x (15): float parameter (can only be read) : object bounding box, min. x position (relative to the object reference frame)
        sim.objfloatparam_objbbox_min_y (16): float parameter (can only be read) : object bounding box, min. y position (relative to the object reference frame)
        sim.objfloatparam_objbbox_min_z (17): float parameter (can only be read) : object bounding box, min. z position (relative to the object reference frame)
        sim.objfloatparam_objbbox_max_x (18): float parameter (can only be read) : object bounding box, max. x position (relative to the object reference frame)
        sim.objfloatparam_objbbox_max_y (19): float parameter (can only be read) : object bounding box, max. y position (relative to the object reference frame)
        sim.objfloatparam_objbbox_max_z (20): float parameter (can only be read) : object bounding box, max. z position (relative to the object reference frame)
        '''
        sim.simxSetObjectIntParameter(self.sim_client,obj_handle,sim.objfloatparam_objbbox_min_x)

    def get_single_obj_orientations(self,object_handle):
        sim_ret, object_orientation = sim.simxGetObjectOrientation(self.sim_client, object_handle, -1, sim.simx_opmode_blocking)
        return object_orientation
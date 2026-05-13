#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
import threading
import copy
import cv2
import os
from os import listdir, environ
from os.path import isfile, join
from tf2_msgs.msg import TFMessage
from scipy.spatial.transform import Rotation as R
import numpy as np
from cv_bridge import CvBridge
from math import sin, cos, pi, comb, sqrt, atan, tan, atan2
import pandas as pd# pip3 install pandas pyarrow
import pyarrow as pa
import pyarrow.parquet as pq

bridge = CvBridge()

tool_pose_xy = [0.0, 0.0] # tool(end effector) pose
tbar_pose_xyw = [0.0, 0.0, 0.0]
vid_H = 240
vid_W = 240
wrist_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
top_camera_image = np.zeros((vid_H, vid_W, 3), np.uint8)
#gripper_state = 1 #1:open 0:close
action = np.array([0.0, 0.0], float)

# --- Compute Bezier curve points ---
def bezier_curve(points, num_points=100):
    """
    Compute Bezier curve points.
    :param points: list or np.array of control points, shape (n+1, 2)
    :param num_points: number of points on the curve
    :return: np.array of shape (num_points, 2)
    """
    n = len(points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))

    for i in range(n + 1):
        binom = comb(n, i)
        curve += (
            binom
            * ((1 - t) ** (n - i))[:, None]
            * (t ** i)[:, None]
            * points[i]
        )

    return curve

class Get_Poses_Subscriber(Node):

    def __init__(self):
        super().__init__('get_modelstate')
        self.subscription = self.create_subscription(
            TFMessage,
            '/isaac_tf',
            self.listener_callback,
            10)
        self.subscription

        self.euler_angles = np.array([0.0, 0.0, 0.0], float)

    def listener_callback(self, data):
        global tool_pose_xy, tbar_pose_xyw

        # 0:tool
        tool_pose = data.transforms[0].transform.translation
        tool_pose_xy[0] = tool_pose.y
        tool_pose_xy[1] = tool_pose.x 

        # 1:tbar
        tbar_translation  = data.transforms[1].transform.translation       
        tbar_rotation = data.transforms[1].transform.rotation 
        tbar_pose_xyw[0] = tbar_translation.y
        tbar_pose_xyw[1] = tbar_translation.x
        self.euler_angles[:] = R.from_quat([tbar_rotation.x, tbar_rotation.y, tbar_rotation.z, tbar_rotation.w]).as_euler('xyz', degrees=False)
        tbar_pose_xyw[2] = self.euler_angles[2]

class Wrist_Camera_Subscriber(Node):

    def __init__(self):
        super().__init__('wrist_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb_wrist',
            self.camera_callback,
            10)
        self.subscription 

    def camera_callback(self, data):
        global wrist_camera_image
        img = bridge.imgmsg_to_cv2(data, "bgr8")
        cropped_img = img[:,80:560,:]
        # interpolation https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
        wrist_camera_image = cv2.resize(cropped_img, (vid_W, vid_H), cv2.INTER_LINEAR)

class Top_Camera_Subscriber(Node):

    def __init__(self):
        super().__init__('top_camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb_top',
            self.camera_callback,
            10)
        self.subscription 

    def camera_callback(self, data):
        global top_camera_image
        img = bridge.imgmsg_to_cv2(data, "bgr8")
        cropped_img = img[:,80:560,:]
        # interpolation https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
        top_camera_image = cv2.resize(cropped_img, (vid_W, vid_H), cv2.INTER_LINEAR)

class Data_Recorder(Node):

    def __init__(self):
        super().__init__('Data_Recorder')
        self.Hz = 10 # bridge data frequency
        self.prev_ee_pose = np.array([0, 0, 0], float)
        self.timer = self.create_timer(1/self.Hz, self.timer_callback)
        self.start_recording = False
        self.data_recorded = False

        #### log files for multiple runs are NOT overwritten
        base_dir = os.environ["HOME"] + "/ur5_simulation/src/data_collection/scripts/my_pusht/"
        self.log_dir = base_dir + "data/chunk-000/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        base_vid_dir = base_dir + 'videos/chunk-000/observation.images.'
        self.wrist_vid_dir = base_vid_dir + 'wrist/'
        if not os.path.exists(self.wrist_vid_dir):
            os.makedirs(self.wrist_vid_dir)

        self.top_vid_dir = base_vid_dir + 'top/'
        if not os.path.exists(self.top_vid_dir):
            os.makedirs(self.top_vid_dir)

        self.state_vid_dir = base_vid_dir + 'state/'
        if not os.path.exists(self.state_vid_dir):
            os.makedirs(self.state_vid_dir)

        # image of a T shape on the table
        self.initial_image = cv2.imread(os.environ['HOME'] + "/ur5_simulation/images/stand_top_plane.png")
        self.initial_image = cv2.rotate(self.initial_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
 
        # for reward calculation
        self.Tbar_region = np.zeros((self.initial_image.shape[0], self.initial_image.shape[1]), np.uint8)
        self.tool_trajectory = np.zeros((self.initial_image.shape[0], self.initial_image.shape[1]), np.uint8)

        # filled image of T shape on the table
        self.T_image = cv2.imread(os.environ['HOME'] + "/ur5_simulation/images/stand_top_plane_filled.png")
        self.T_image = cv2.rotate(self.T_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # border image
        self.border_image = cv2.imread(os.environ['HOME'] + "/ur5_simulation/images/border.png")
        self.border_image = cv2.rotate(self.border_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(self.border_image, cv2.COLOR_BGR2GRAY)
        _, self.border_gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        img_gray = cv2.cvtColor(self.T_image, cv2.COLOR_BGR2GRAY)
        thr, img_th = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        self.blue_region = cv2.bitwise_not(img_th)
        self.blue_region_sum = cv2.countNonZero(self.blue_region)
        # for debug
        self.sum_img_pub = self.create_publisher(Image, '/sum_image', 10)
        
        self.pub_pose = self.create_publisher(Pose, '/target_pose', 10)
        self.target_pose = Pose()
        self.target_pose.position.z = 0.7389
        self.target_pose.orientation.w = 0.5
        self.target_pose.orientation.x = -0.5
        self.target_pose.orientation.y = -0.5
        self.target_pose.orientation.z = 0.5        

        self.pub_img = self.create_publisher(Image, '/pushT_image', 10)
        self.tool_radius = 10 # millimeters
        self.scale = 1.639344 # mm/pix
        self.C_W = 182  # pix
        self.C_H = 152  # pix
        self.OBL1 = 150 # mm
        self.OBL2 = 120 # mm
        self.OBW = 30   # mm
        self.OBL1_pix = int(150/self.scale) # pix
        self.OBL2_pix = int(120/self.scale) # pix
        self.OBW_pix = int(30/self.scale)   # pix        
        # radius of the tool
        self.radius = int(10/self.scale)
        self.touch_L = int((120 - 15)/self.scale) # side ponit:15mm from the bottom
        self.touch_W = int((30 + 14)/self.scale)  # side ponits 6+6mm from each sicde
        self.bottom_L = int((120 + 8)/self.scale) # side ponit:6mm from the bottom
        self.up_horizontal_L = int((150 + 14)/self.scale) # side ponit:7mm from the Tbar horizontal part
        self.up_vertical_L = int((30 + 14)/self.scale)
        self.r = sqrt((self.touch_L/1000)**2 + (self.touch_W/1000)**2) # in mm
        self.theta_const = atan((30/2 + 12/2)/(120 - 15))

        self.df = pd.DataFrame(columns=['observation.state', 'action', 'episode_index', 'frame_index', 'timestamp', 'next.reward', 'next.done', 'next.success', 'index', 'task_index'])
        data_path = environ['HOME'] + '/ur5_simulation/src/data_collection/scripts/my_pusht/data/chunk-000'
        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        if len(onlyfiles) > 0:
            onlyfiles.sort()
            df = pd.read_parquet(os.environ["HOME"] + f'/ur5_simulation/src/data_collection/scripts/my_pusht/data/chunk-000/{onlyfiles[-1]}')
            self.index = df['index'][len(df)-1] + 1
            self.episode_index = df['episode_index'][len(df)-1] + 1
        else:
            self.index = 0
            self.episode_index = 0

        print(f"Begin recording from episode indes:{self.episode_index}, index:{self.index}")
        
        self.frame_index = 0
        self.time_stamp = 0.0
        self.success = False
        self.done = False
        self.column_index = 0
        self.prev_sum = 0.0

        self.wrist_camera_array = []
        self.top_camera_array = []
        self.state_image_array = []
        self.time_steps = 0

        self.first_stage = True
        self.second_stage = False
        self.third_state = False
        self.fourth_stage = False
        self.fifth_stage = False
        self.sixth_stage = False
        self.seventh_stage = False
        self.eighth_stage = False
        self.ninth_stage = False
        self.has_trajectory = False
        self.current_step = 0
        self.curve = []
        self.points_num = 20
        self.tbar_goal_pose = [-0.00572, 0.5311, 1.9816]
        self.goal_sum = 0.92
        self.move_dist_per_step = 0.002 # 2mm per step

        # for test
        self.i = 0
        self.target_point = np.array([0, 0], int)
        self.angle_dif = 0

        self.tp_x_real = 0
        self.tp_y_real = 0

        self.total_steps = 0
        self.record_data = True

    def timer_callback(self):
        global tool_pose_xy, tbar_pose_xyw, action, wrist_camera_image, top_camera_image
        
        if np.any(tool_pose_xy) and np.any(tbar_pose_xyw) and np.any(wrist_camera_image) and np.any(top_camera_image):

            image = copy.copy(self.initial_image)

            self.Tbar_region[:] = 0
            self.tool_trajectory[:] = 0

            # x,y coordinates of the EE tip
            x_tool = int((tool_pose_xy[0]*1000 + 300)/self.scale)
            y_tool = int((tool_pose_xy[1]*1000 - 320)/self.scale)     
            
            # horizontal part of the T
            x1 = tbar_pose_xyw[0]
            y1 = tbar_pose_xyw[1]
            th1 = -tbar_pose_xyw[2] - pi/2
            dx1 = -self.OBW_pix/2*cos(th1 - pi/2)
            dy1 = -self.OBW_pix/2*sin(th1 - pi/2)
            self.tbar1_ob = [[int(cos(th1)*self.OBL1_pix/2    - sin(th1)*self.OBW_pix/2   + dx1 + self.C_W + 1000*x1/self.scale),
                              int(sin(th1)*self.OBL1_pix/2    + cos(th1)*self.OBW_pix/2   + dy1 + (1000*y1-320)/self.scale)],
                            [int(cos(th1)*self.OBL1_pix/2    - sin(th1)*(-self.OBW_pix/2)+ dx1 + self.C_W + 1000*x1/self.scale),
                             int(sin(th1)*self.OBL1_pix/2    + cos(th1)*(-self.OBW_pix/2)+ dy1 + (1000*y1-320)/self.scale)],
                            [int(cos(th1)*(-self.OBL1_pix/2) - sin(th1)*(-self.OBW_pix/2)+ dx1 + self.C_W + 1000*x1/self.scale),
                             int(sin(th1)*(-self.OBL1_pix/2) + cos(th1)*(-self.OBW_pix/2)+ dy1 + (1000*y1-320)/self.scale)],
                            [int(cos(th1)*(-self.OBL1_pix/2) - sin(th1)*self.OBW_pix/2   + dx1 + self.C_W + 1000*x1/self.scale),
                             int(sin(th1)*(-self.OBL1_pix/2) + cos(th1)*self.OBW_pix/2   + dy1 + (1000*y1-320)/self.scale)]]  
            pts1_ob = np.array(self.tbar1_ob, np.int32)
            cv2.fillPoly(image, [pts1_ob], (0, 0, 180))
            cv2.fillPoly(self.Tbar_region, [pts1_ob], 255)

            self.upper_ponit = [int(-sin(th1)*(self.up_vertical_L/2) + dx1 + self.C_W + 1000*x1/self.scale),
                                int(cos(th1)*(self.up_vertical_L/2) + dy1 + (1000*y1 - 320)/self.scale)]
            
            #vertical part of the T
            th2 = -tbar_pose_xyw[2] - pi
            dx2 = self.OBL2_pix/2*cos(th2)
            dy2 = self.OBL2_pix/2*sin(th2)
            self.tbar2_ob = [[int(cos(th2)*self.OBL2_pix/2     - sin(th2)*self.OBW_pix/2   + dx2 + self.C_W + 1000*x1/self.scale),
                              int(sin(th2)*self.OBL2_pix/2     + cos(th2)*self.OBW_pix/2   + dy2 + (1000*y1 - 320)/self.scale)],
                            [int(cos(th2)*self.OBL2_pix/2    - sin(th2)*(-self.OBW_pix/2)+ dx2 + self.C_W + 1000*x1/self.scale),
                             int(sin(th2)*self.OBL2_pix/2    + cos(th2)*(-self.OBW_pix/2)+ dy2 + (1000*y1 - 320)/self.scale)],
                            [int(cos(th2)*(-self.OBL2_pix/2) - sin(th2)*(-self.OBW_pix/2)+ dx2 + self.C_W + 1000*x1/self.scale),
                             int(sin(th2)*(-self.OBL2_pix/2) + cos(th2)*(-self.OBW_pix/2)+ dy2 + (1000*y1 - 320)/self.scale)],
                            [int(cos(th2)*(-self.OBL2_pix/2) - sin(th2)*self.OBW_pix/2   + dx2 + self.C_W + 1000*x1/self.scale),
                             int(sin(th2)*(-self.OBL2_pix/2) + cos(th2)*self.OBW_pix/2   + dy2 + (1000*y1 - 320)/self.scale)]]  
            pts2_ob = np.array(self.tbar2_ob, np.int32)
            cv2.fillPoly(image, [pts2_ob], (0, 0, 180))
            cv2.fillPoly(self.Tbar_region, [pts2_ob], 255)

            common_part = cv2.bitwise_and(self.blue_region, self.Tbar_region)
            common_part_sum = cv2.countNonZero(common_part)
            sum = common_part_sum/self.blue_region_sum
            self.prev_sum = sum
            print(f"sum:{sum}, tbar_pose_xyw: x:{x1}, y:{y1}, theta:{tbar_pose_xyw[2]}")
            Tbar_center = [int(self.C_W + 1000*x1/self.scale), int((1000*y1 - 320)/self.scale)]

            # To adjust T-bar angle
            # Calculating 2 ponits from the left and right side of the vertical part of the T-bar
            self.side_points = [[int(cos(th2)*self.touch_L/2 - sin(th2)*self.touch_W/2 + dx2 + self.C_W + 1000*x1/self.scale),
                                 int(sin(th2)*self.touch_L/2 + cos(th2)*self.touch_W/2 + dy2 + (1000*y1 - 320)/self.scale)],
                                [int(cos(th2)*self.touch_L/2 - sin(th2)*(-self.touch_W/2)+ dx2 + self.C_W + 1000*x1/self.scale),
                                 int(sin(th2)*self.touch_L/2 + cos(th2)*(-self.touch_W/2)+ dy2 + (1000*y1 - 320)/self.scale)]]

            # Calculating 2 tip points on horizontal part of the T-bar
            self.tip_ponits = [[int(cos(th1)*self.up_horizontal_L/2 + dx1 + self.C_W + 1000*x1/self.scale),
                                int(sin(th1)*self.up_horizontal_L/2 + dy1 + (1000*y1 - 320)/self.scale)],
                               [int(cos(th1)*(-self.up_horizontal_L/2) + dx1 + self.C_W + 1000*x1/self.scale),
                                int(sin(th1)*(-self.up_horizontal_L/2) + dy1 + (1000*y1 - 320)/self.scale)]]      

            self.bottom_ponit = (int(cos(th2)*self.bottom_L/2 + dx2 + self.C_W + 1000*x1/self.scale),
                                 int(sin(th2)*self.bottom_L/2 + dy2 + (1000*y1 - 320)/self.scale)) 
            
            debug_image = copy.copy(image)
            
            cv2.circle(debug_image, center=(self.tip_ponits[0][0], self.tip_ponits[0][1]), radius=2, color=(200, 0, 0), thickness=cv2.FILLED)
            cv2.circle(debug_image, center=(self.tip_ponits[1][0], self.tip_ponits[1][1]), radius=2, color=(200, 0, 0), thickness=cv2.FILLED) 
            cv2.circle(debug_image, center=Tbar_center, radius=2, color=(0, 200, 0), thickness=cv2.FILLED) 

            # Moving to initial position
            if self.first_stage:
                if self.has_trajectory == False:
                    if tbar_pose_xyw[2] >= 0:
                        self.angle_dif = 1.9816 - tbar_pose_xyw[2]
                        if self.angle_dif > 0:
                            self.target_point[0] = self.side_points[0][0]
                            self.target_point[1] = self.side_points[0][1]      
                        else:
                            self.target_point[0] = self.side_points[1][0]
                            self.target_point[1] = self.side_points[1][1]   
                    else:
                        self.angle_dif = 1.9816 - pi - tbar_pose_xyw[2]
                        if self.angle_dif < 0:
                            self.target_point[0] = self.side_points[0][0]
                            self.target_point[1] = self.side_points[0][1]      
                        else:
                            self.target_point[0] = self.side_points[1][0]
                            self.target_point[1] = self.side_points[1][1]   

                    self.tp_x_real = (self.scale*self.target_point[0] - 300)/1000
                    self.tp_y_real = (self.scale*self.target_point[1] + 320)/1000

                    # step 1
                    # Move to position from which we can rotate the bar
                    cv2.circle(self.tool_trajectory, center=(x_tool, y_tool), radius=self.radius, color=255, thickness=cv2.FILLED)
                    cv2.circle(self.tool_trajectory, center=(self.target_point[0], self.target_point[1]), radius=self.radius, color=255, thickness=cv2.FILLED)
                    cv2.line(self.tool_trajectory, (x_tool, y_tool), (self.target_point[0], self.target_point[1]), 255, 2*self.radius)
                    line_tbar_common = cv2.bitwise_and(self.tool_trajectory, self.Tbar_region)
                    line_tbar_common_sum = cv2.countNonZero(line_tbar_common)
            
                    # there is an obstacle in a way of the tool bar
                    if line_tbar_common_sum > 30:
                        contours, _ = cv2.findContours(line_tbar_common, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # Initialize variables to store the center coordinates
                        centers = []
                        # Loop through contours to calculate centroids
                        for contour in contours:
                            # Calculate moments
                            M = cv2.moments(contour)
                            if M['m00'] != 0:
                                # Calculate x, y coordinates of the centroid
                                cX = int(M['m10'] / M['m00'])
                                cY = int(M['m01'] / M['m00'])
                                centers.append((cX, cY))

                        self.tool_trajectory[:] = 0
                        cv2.circle(self.tool_trajectory, center=(x_tool, y_tool), radius=self.radius, color=255, thickness=cv2.FILLED)
                        cv2.circle(self.tool_trajectory, center=(self.target_point[0], self.target_point[1]),
                        radius=self.radius, color=255, thickness=cv2.FILLED)

                        contour_x = (self.scale*centers[0][0] - 300)/1000
                        control_y = (tool_pose_xy[1] + self.tp_y_real)/2 - 0.07

                        if tbar_pose_xyw[2] < -pi*45/180:
                            control_1 = [(tool_pose_xy[0] + contour_x)/2, control_y]
                            control_2 = [self.tp_x_real - 4*abs(self.tp_x_real - contour_x), control_y]
                        elif -pi*45/180 <= tbar_pose_xyw[2] < pi*45/180:
                            control_1 = [(tool_pose_xy[0] + contour_x)/2, control_y - 0.1]
                            control_2 = [self.tp_x_real - 4*abs(self.tp_x_real - contour_x), control_y]
                        elif pi*45/180 <= tbar_pose_xyw[2] < pi*60/180:
                            control_1 = [(tool_pose_xy[0] + contour_x)/2, control_y - 0.15]
                            control_2 = [self.tp_x_real - 5*abs(self.tp_x_real - contour_x), control_y]
                        elif pi*60/180 <= tbar_pose_xyw[2] < pi*75/180:
                            control_1 = [(tool_pose_xy[0] + contour_x)/2, control_y - 0.15]
                            control_2 = [self.tp_x_real - 5*abs(self.tp_x_real - contour_x), control_y - 0.1]
                        elif pi*75/180 <= tbar_pose_xyw[2] < pi*90/180:
                            control_1 = [(tool_pose_xy[0] + contour_x)/2, control_y - 0.15]
                            a = sqrt((tbar_pose_xyw[2] - 1.309)/0.2688) # 0 < a < 1
                            control_2 = [self.tp_x_real - (5 + 8*a)*abs(self.tp_x_real - contour_x), control_y - 0.1]
                        else:
                            control_1 = [contour_x - 0.08, control_y - 0.1]                   
                            control_2 = [self.tp_x_real - 25*abs(self.tp_x_real - contour_x), control_y + 0.1]
                
                        points = np.array([
                            [tool_pose_xy[0], tool_pose_xy[1]], # start
                            control_1,
                            control_2,
                            [self.tp_x_real, self.tp_y_real] # end
                        ])

                        self.curve = np.fliplr(bezier_curve(points, num_points=50))
                        self.time_steps = len(self.curve)
                        self.has_trajectory = True
                    else:
                        print(F"tool_pose:{tool_pose_xy}")
                        for i in range(self.points_num):
                            # x, y in isaacsim (this model) is y, x in moveit 
                            self.curve.append([tool_pose_xy[1] + (i+1)*(self.tp_y_real- tool_pose_xy[1])/self.points_num,
                                            tool_pose_xy[0] + (i+1)*(self.tp_x_real - tool_pose_xy[0])/self.points_num])
                        
                        self.time_steps = len(self.curve)
                        self.has_trajectory = True
                        
                # Drawing Bezier curve
                for i in range(len(self.curve) - 1):
                    cv2.line(debug_image, (int((self.curve[i][1]*1000 + 300)/self.scale), int((self.curve[i][0]*1000 - 320)/self.scale)),
                                    (int((self.curve[i+1][1]*1000 + 300)/self.scale), int((self.curve[i+1][0]*1000 - 320)/self.scale)),
                                    (255, 255, 0), 2*self.radius)                
                    
                if self.time_steps > 1:
                    self.time_steps -= 1
                    self.current_step = len(self.curve) - self.time_steps
                elif self.time_steps == 1:
                    self.first_stage = False
                    self.second_stage = True
                    self.has_trajectory = False

                self.target_pose.position.x = self.curve[self.current_step][0]
                self.target_pose.position.y = self.curve[self.current_step][1]
                self.pub_pose.publish(self.target_pose)

            # Rotating the T-bar
            elif self.second_stage:
                if self.has_trajectory == False:
                    self.current_step = 0

                    if (self.tbar_goal_pose[2] - pi) <= tbar_pose_xyw[2]:
                        self.angle_dif = self.tbar_goal_pose[2] - tbar_pose_xyw[2] 
                    else:
                        self.angle_dif = self.tbar_goal_pose[2] - tbar_pose_xyw[2] - 2*pi

                    one_step = pi/180
                    steps_num = 2 + int(abs(self.angle_dif)/one_step) # we at least have start and end points

                    self.curve = []
                    for i in range(steps_num):
                        if tbar_pose_xyw[2] >= (self.tbar_goal_pose[2] - pi) and tbar_pose_xyw[2] < self.tbar_goal_pose[2]:
                            angle = tbar_pose_xyw[2] - self.theta_const - pi/2 + i*self.angle_dif/(steps_num - 1)
                        else:
                            angle = tbar_pose_xyw[2] + self.theta_const - pi/2 + i*self.angle_dif/(steps_num - 1)
                            
                        x = x1 + (self.r*self.scale)*sin(angle)
                        y = y1 + (self.r*self.scale)*cos(angle)
                        self.curve.append([y, x]) # for arm trajectory calculation

                    self.time_steps = len(self.curve)
                    self.has_trajectory = True                

                for i in range(len(self.curve) - 1):        
                    cv2.line(debug_image, (int((self.curve[i][1]*1000 + 300)/self.scale), int((self.curve[i][0]*1000 - 320)/self.scale)),
                                    (int((self.curve[i+1][1]*1000 + 300)/self.scale), int((self.curve[i+1][0]*1000 - 320)/self.scale)),
                                    (255, 255, 0), 2*self.radius)         

                dist = sqrt((tbar_pose_xyw[0] - self.tp_x_real)**2 + (tbar_pose_xyw[1] - self.tp_y_real)**2)

                if dist < 0.135:
                    if self.time_steps > 1:
                        self.time_steps -= 1
                        self.current_step = len(self.curve) - self.time_steps
                    elif self.time_steps == 1:
                        if self.tbar_goal_pose[2] - abs(tbar_pose_xyw[2]) < 0.04:
                            self.second_stage = False  
                            self.third_state = False
                            if sum < self.goal_sum:
                                self.fourth_stage = True
                            else:
                                self.fourth_stage = False
                                self.ninth_stage = False
                            self.has_trajectory = False
                        else:
                            self.second_stage = True
                            self.third_stage = False
                            self.fourth_stage = False
                            self.fifth_stage = False
                            self.sixth_stage = False
                            self.seventh_stage = False
                            self.eighth_stage = False
                            self.has_trajectory = False                         
                else:
                    self.second_stage = False
                    self.third_state = True
                    self.fourth_stage = False
                    self.fifth_stage = False
                    self.sixth_stage = False
                    self.seventh_stage = False
                    self.eighth_stage = False
                    self.has_trajectory = False   

                self.target_pose.position.x = self.curve[self.current_step][0]
                self.target_pose.position.y = self.curve[self.current_step][1]
                self.pub_pose.publish(self.target_pose)

            # Adjusting the T-bar position if it deviates from preferable position
            elif self.third_state:
                if self.has_trajectory == False:
                    if tbar_pose_xyw[2] >= 0:
                        self.angle_dif = self.tbar_goal_pose[2] - tbar_pose_xyw[2]
                        if self.angle_dif > 0:
                            self.target_point[0] = self.side_points[0][0]
                            self.target_point[1] = self.side_points[0][1]      
                        else:
                            self.target_point[0] = self.side_points[1][0]
                            self.target_point[1] = self.side_points[1][1]   
                    else:
                        self.angle_dif = self.tbar_goal_pose[2] - pi - tbar_pose_xyw[2]
                        if self.angle_dif < 0:
                            self.target_point[0] = self.side_points[0][0]
                            self.target_point[1] = self.side_points[0][1]      
                        else:
                            self.target_point[0] = self.side_points[1][0]
                            self.target_point[1] = self.side_points[1][1]   

                    self.tp_x_real = (self.scale*self.target_point[0] - 300)/1000
                    self.tp_y_real = (self.scale*self.target_point[1] + 320)/1000

                    self.curve = []
                    self.points_num = 5
                    for i in range(self.points_num):
                        # x, y in isaacsim (this model) is y, x in moveit 
                        self.curve.append([tool_pose_xy[1] + (i+1)*(self.tp_y_real - tool_pose_xy[1])/self.points_num,
                                           tool_pose_xy[0] + (i+1)*(self.tp_x_real - tool_pose_xy[0])/self.points_num])
                        
                        self.time_steps = len(self.curve)
                        self.has_trajectory = True

                if self.time_steps > 1:
                    self.time_steps -= 1
                    self.current_step = len(self.curve) - self.time_steps
                elif self.time_steps == 1:
                    self.third_stage = False
                    self.second_stage = True
                    self.has_trajectory = False

                self.target_pose.position.x = self.curve[self.current_step][0]
                self.target_pose.position.y = self.curve[self.current_step][1]
                self.pub_pose.publish(self.target_pose)

            # Moving to position to push the T-bar
            elif self.fourth_stage:
                if self.has_trajectory == False:

                    dist1 = sqrt((tool_pose_xy[0] - (self.scale*self.side_points[0][0] - 300)/1000)**2 +
                                (tool_pose_xy[1] - (self.scale*self.side_points[0][1] + 320)/1000)**2)
                    
                    dist2 = sqrt((tool_pose_xy[0] - (self.scale*self.side_points[1][0] - 300)/1000)**2 +
                                (tool_pose_xy[1] - (self.scale*self.side_points[1][1] + 320)/1000)**2)
                    
                    if dist1 < dist2:
                        side_point_x = (self.scale*self.side_points[0][0] - 300)/1000
                        side_point_y = (self.scale*self.side_points[0][1] + 320)/1000
                    else:
                        side_point_x = (self.scale*self.side_points[1][0] - 300)/1000
                        side_point_y = (self.scale*self.side_points[1][1] + 320)/1000

                    above_or_under_line = (tbar_pose_xyw[1] - self.tbar_goal_pose[1]) - tan(self.tbar_goal_pose[2] + pi/2)*(tbar_pose_xyw[0] - self.tbar_goal_pose[0])

                    if self.border_gray[Tbar_center[1], Tbar_center[0]] == 0:

                        bottom_point_x = (self.scale*self.bottom_ponit[0] - 300)/1000
                        bottom_point_y = (self.scale*self.bottom_ponit[1] + 320)/1000     

                        middle_point_x = (side_point_x + bottom_point_x)/2
                        middle_point_y = (side_point_y + bottom_point_y)/2
        
                        mag = sqrt((middle_point_x - tbar_pose_xyw[0])**2 + (middle_point_y - tbar_pose_xyw[1])**2)
                        unit_vec = [(middle_point_x - tbar_pose_xyw[0])/mag, (middle_point_y - tbar_pose_xyw[1])/mag]

                        control_point = [middle_point_x + 0.06*unit_vec[0], middle_point_y + 0.06*unit_vec[1]]      

                        points = np.array([
                            [tool_pose_xy[0], tool_pose_xy[1]], # start
                            control_point,
                            [bottom_point_x, bottom_point_y] # end
                        ])

                    else:
                        upper_point_real = [(self.scale*self.upper_ponit[0] - 300)/1000, (self.scale*self.upper_ponit[1] + 320)/1000]

                        B = 120 #mm
                        C = 70 # mm
                        dist = 0.2 # m
                        if dist1 < dist2:
                            control_point2 = [int(-sin(th1 - 0.6)*(self.up_vertical_L/2 + B) + dx1 + self.C_W + 1000*x1/self.scale),
                                            int(cos(th1 - 0.6)*(self.up_vertical_L/2 + B) + dy1 + (1000*y1 - 320)/self.scale)]

                            control_point1_real = [tool_pose_xy[0] - dist*sin(tbar_pose_xyw[2]),
                                                tool_pose_xy[1] - dist*cos(tbar_pose_xyw[2])]
                        else:
                            control_point2 = [int(-sin(th1 + 0.6)*(self.up_vertical_L/2 + B) + dx1 + self.C_W + 1000*x1/self.scale),
                                            int(cos(th1 + 0.6)*(self.up_vertical_L/2 + B) + dy1 + (1000*y1 - 320)/self.scale)]

                            control_point1_real = [tool_pose_xy[0] + dist*sin(tbar_pose_xyw[2]),
                                                tool_pose_xy[1] + dist*cos(tbar_pose_xyw[2])]                        

                        control_point2_real = [(self.scale*control_point2[0] - 300)/1000,
                                            (self.scale*control_point2[1] + 320)/1000]
                        
                        control_point3 = [int(-sin(th1)*(self.up_vertical_L/2 + C) + dx1 + self.C_W + 1000*x1/self.scale),
                                        int(cos(th1)*(self.up_vertical_L/2 + C) + dy1 + (1000*y1 - 320)/self.scale)]

                        control_point3_real = [(self.scale*control_point3[0] - 300)/1000,
                                            (self.scale*control_point3[1] + 320)/1000]
                        
                        points = np.array([
                            tool_pose_xy[:2],    # start
                            control_point1_real,
                            control_point2_real,
                            control_point3_real, 
                            upper_point_real     # end
                        ]) 

                    self.curve = np.fliplr(bezier_curve(points, num_points=40))
                    self.time_steps = len(self.curve)
                    self.has_trajectory = True

                # Drawing Bezier curve
                for i in range(len(self.curve) - 1):
                    cv2.line(debug_image, (int((self.curve[i][1]*1000 + 300)/self.scale), int((self.curve[i][0]*1000 - 320)/self.scale)),
                                    (int((self.curve[i+1][1]*1000 + 300)/self.scale), int((self.curve[i+1][0]*1000 - 320)/self.scale)),
                                    (255, 255, 0), 2*self.radius)                
                    
                if self.time_steps > 1:
                    self.time_steps -= 1
                    self.current_step = len(self.curve) - self.time_steps
                elif self.time_steps == 1:
                    self.first_stage = False
                    self.second_stage = False
                    self.third_stage = False
                    self.fourth_stage = False
                    self.fifth_stage = True
                    self.sixth_stage = False
                    self.seventh_stage = False
                    self.has_trajectory = False

                self.target_pose.position.x = self.curve[self.current_step][0]
                self.target_pose.position.y = self.curve[self.current_step][1]
                self.pub_pose.publish(self.target_pose)  

            # Pushing the T-bar
            elif self.fifth_stage:
                if self.has_trajectory == False:
                    self.curve = []
                    angle = tbar_pose_xyw[2] - pi/2

                    dist = sqrt((tbar_pose_xyw[0] - self.tbar_goal_pose[0])**2 + (tbar_pose_xyw[1] - self.tbar_goal_pose[1])**2)
                    dist_diff = 0
                    alpha = atan2((tbar_pose_xyw[1] - self.tbar_goal_pose[1]), (tbar_pose_xyw[0] - self.tbar_goal_pose[0]))
                    if tbar_pose_xyw[0] < self.tbar_goal_pose[0] and tbar_pose_xyw[1] < self.tbar_goal_pose[1]:
                        theta = self.tbar_goal_pose[2] + pi/2 + alpha
                        dist_diff = dist*sin(theta)
                        self.points_num = max(int(abs(dist_diff)/self.move_dist_per_step), 2)
                        for i in range(self.points_num):
                            self.curve.append([tool_pose_xy[1] + (i+1)*dist_diff*cos(angle)/self.points_num,
                                            tool_pose_xy[0] + (i+1)*dist_diff*sin(angle)/self.points_num])
                    elif tbar_pose_xyw[0] >= self.tbar_goal_pose[0] and tbar_pose_xyw[1] < self.tbar_goal_pose[1]:
                        theta = pi/2 - alpha - self.tbar_goal_pose[2]
                        dist_diff = dist*sin(theta)
                        self.points_num = max(int(abs(dist_diff)/self.move_dist_per_step), 2)
                        for i in range(self.points_num):
                            self.curve.append([tool_pose_xy[1] + (i+1)*dist_diff*cos(angle)/self.points_num,
                                            tool_pose_xy[0] + (i+1)*dist_diff*sin(angle)/self.points_num])
                    elif tbar_pose_xyw[0] >= self.tbar_goal_pose[0] and tbar_pose_xyw[1] >= self.tbar_goal_pose[1]:
                        theta = alpha + self.tbar_goal_pose[2] - pi/2
                        dist_diff = dist*sin(theta)
                        self.points_num = max(int(abs(dist_diff)/self.move_dist_per_step), 2)
                        for i in range(self.points_num):
                            self.curve.append([tool_pose_xy[1] - (i+1)*dist_diff*cos(angle)/self.points_num,
                                            tool_pose_xy[0] - (i+1)*dist_diff*sin(angle)/self.points_num])
                    else:
                        theta = alpha + self.tbar_goal_pose[2] - 3*pi/2
                        dist_diff = dist*sin(theta)
                        self.points_num = max(int(abs(dist_diff)/self.move_dist_per_step), 2)
                        for i in range(self.points_num):
                            self.curve.append([tool_pose_xy[1] + (i+1)*dist_diff*cos(angle)/self.points_num,
                                            tool_pose_xy[0] + (i+1)*dist_diff*sin(angle)/self.points_num])

                    self.time_steps = len(self.curve)
                    self.has_trajectory = True

                # Drawing Bezier curve
                for i in range(len(self.curve) - 1):
                    cv2.line(debug_image, (int((self.curve[i][1]*1000 + 300)/self.scale), int((self.curve[i][0]*1000 - 320)/self.scale)),
                                    (int((self.curve[i+1][1]*1000 + 300)/self.scale), int((self.curve[i+1][0]*1000 - 320)/self.scale)),
                                    (255, 255, 0), 2*self.radius) 

                if self.time_steps > 1:
                    self.time_steps -= 1
                    self.current_step = len(self.curve) - self.time_steps
                elif self.time_steps == 1:
                    self.first_stage = False
                    self.second_stage = False
                    self.third_stage = False
                    self.fourth_stage = False
                    self.fifth_stage = False
                    self.sixth_stage = True
                    self.has_trajectory = False

                self.target_pose.position.x = self.curve[self.current_step][0]
                self.target_pose.position.y = self.curve[self.current_step][1]
                self.pub_pose.publish(self.target_pose)

            elif self.sixth_stage:
                if self.has_trajectory == False:
                    
                    C = 100
                    dist = 0.15
                    L = 52
                    if self.tbar_goal_pose[0] - tbar_pose_xyw[0] > 0:
                        control_point2 = [int(cos(th1)*C + dx1 + self.C_W + 1000*x1/self.scale),
                                        int(sin(th1)*C + dy1 + (1000*y1 - 320)/self.scale)]
                        control_point1_real = [tool_pose_xy[0] - dist*sin(tbar_pose_xyw[2]),
                                            tool_pose_xy[1] - dist*cos(tbar_pose_xyw[2])]
                        
                        tip_point = [int(cos(th1)*L + dx1 + self.C_W + 1000*x1/self.scale),
                                    int(sin(th1)*L + dy1 + (1000*y1 - 320)/self.scale)]
                    else:
                        control_point2 = [int(cos(th1)*(-C) + dx1 + self.C_W + 1000*x1/self.scale),
                                        int(sin(th1)*(-C) + dy1 + (1000*y1 - 320)/self.scale)]
                        control_point1_real = [tool_pose_xy[0] + dist*sin(tbar_pose_xyw[2]),
                                            tool_pose_xy[1] + dist*cos(tbar_pose_xyw[2])]
                        
                        tip_point = [int(cos(th1)*(-L) + dx1 + self.C_W + 1000*x1/self.scale),
                                    int(sin(th1)*(-L) + dy1 + (1000*y1 - 320)/self.scale)] 

                    control_point2_real = [(self.scale*control_point2[0] - 300)/1000, (self.scale*control_point2[1] + 320)/1000]
                    control_point2 = [int((control_point1_real[0]*1000 + 300)/self.scale), int((control_point1_real[1]*1000 - 320)/self.scale)]
                    tip_point_real = [(self.scale*tip_point[0] - 300)/1000, (self.scale*tip_point[1] + 320)/1000]

                    points = np.array([
                            [tool_pose_xy[0], tool_pose_xy[1]], # start
                            control_point1_real,   # control 1
                            control_point2_real, # control 2
                            tip_point_real # end
                    ])           

                    self.curve = np.fliplr(bezier_curve(points, num_points=40))
                    self.time_steps = len(self.curve)
                    self.has_trajectory = True     

                for i in range(len(self.curve) - 1):
                    cv2.line(debug_image, (int((self.curve[i][1]*1000 + 300)/self.scale), int((self.curve[i][0]*1000 - 320)/self.scale)),
                                    (int((self.curve[i+1][1]*1000 + 300)/self.scale), int((self.curve[i+1][0]*1000 - 320)/self.scale)),
                                    (255, 255, 0), 2*self.radius)
                    
                if self.time_steps > 1:
                    self.time_steps -= 1
                    self.current_step = len(self.curve) - self.time_steps
                elif self.time_steps == 1:
                    self.first_stage = False
                    self.second_stage = False
                    self.third_stage = False
                    self.fourth_stage = False
                    self.fifth_stage = False
                    self.sixth_stage = False
                    self.seventh_stage = True
                    self.has_trajectory = False

                self.target_pose.position.x = self.curve[self.current_step][0]
                self.target_pose.position.y = self.curve[self.current_step][1]
                self.pub_pose.publish(self.target_pose)

            elif self.seventh_stage:
                if self.has_trajectory == False:

                    x_dif = self.tbar_goal_pose[0] - tbar_pose_xyw[0]
                    y_dif = x_dif*tan(tbar_pose_xyw[2] - pi/2)
                    y_dif = -y_dif

                    self.curve = []
                    self.points_num = 20
                    for i in range(self.points_num):
                        # x, y in isaacsim (this model) is y, x in moveit 
                        self.curve.append([tool_pose_xy[1] + (i+1)*y_dif/self.points_num,
                                        tool_pose_xy[0] + (i+1)*x_dif/self.points_num])
                        
                        self.time_steps = len(self.curve)
                        self.has_trajectory = True

                for i in range(len(self.curve) - 1):
                    cv2.line(debug_image, (int((self.curve[i][1]*1000 + 300)/self.scale),
                                     int((self.curve[i][0]*1000 - 320)/self.scale)),
                                    (int((self.curve[i+1][1]*1000 + 300)/self.scale),
                                     int((self.curve[i+1][0]*1000 - 320)/self.scale)),
                                    (255, 255, 0), 2*self.radius)
                    
                if self.time_steps > 1:
                    self.time_steps -= 1
                    self.current_step = len(self.curve) - self.time_steps
                elif self.time_steps == 1:
                    self.first_stage = False
                    self.second_stage = False
                    self.third_stage = False
                    self.fifth_stage = False
                    self.sixth_stage = False
                    self.seventh_stage = False
                    self.has_trajectory = False

                    if sum < self.goal_sum:
                        if abs(self.tbar_goal_pose[2] - tbar_pose_xyw[2]) > 0.01745: # 1 deg
                            self.fourth_stage = False
                            self.eighth_stage = True
                        else:
                            self.fourth_stage = True
                            self.eighth_stage = False
                    else:
                        self.fourth_stage = False
                        self.eighth_stage = False

                self.target_pose.position.x = self.curve[self.current_step][0]
                self.target_pose.position.y = self.curve[self.current_step][1]
                self.pub_pose.publish(self.target_pose)

            elif self.eighth_stage:
                if self.has_trajectory == False:

                    dist1 = sqrt((tool_pose_xy[0] - (self.scale*self.side_points[0][0] - 300)/1000)**2 +
                                (tool_pose_xy[1] - (self.scale*self.side_points[0][1] + 320)/1000)**2)
                    
                    dist2 = sqrt((tool_pose_xy[0] - (self.scale*self.side_points[1][0] - 300)/1000)**2 +
                                (tool_pose_xy[1] - (self.scale*self.side_points[1][1] + 320)/1000)**2)
                    
                    angle_dif = self.tbar_goal_pose[2] - tbar_pose_xyw[2]

                    B = 120
                    C = 100
                    dist = 0.1
                    L = 52
                    L1 = 0.1
                    L2 = 0.25
                    L3 = 0.15
                    L4 = 0.10

                    if dist1 < dist2 and angle_dif > 0:
                        control_point1_real = [tool_pose_xy[0] - dist*sin(tbar_pose_xyw[2]),
                                               tool_pose_xy[1] - dist*cos(tbar_pose_xyw[2])]
                        
                        side_point_x = (self.scale*self.side_points[0][0] - 300)/1000
                        side_point_y = (self.scale*self.side_points[0][1] + 320)/1000

                        control_point2_real = [side_point_x - dist*sin(tbar_pose_xyw[2]),
                                               side_point_y - dist*cos(tbar_pose_xyw[2])]
                        
                        self.target_point = [self.side_points[0][0],
                                             self.side_points[0][1]]

                        tp_real = [(self.scale*self.target_point[0] - 300)/1000,
                                   (self.scale*self.target_point[1] + 320)/1000]

                        points = np.array([
                             tool_pose_xy[:2], # start
                             control_point1_real,
                             control_point2_real,
                             tp_real  # end
                        ])
                    elif dist1 < dist2 and angle_dif <= 0: 
                        control_point1_real = [tool_pose_xy[0] - L1*sin(tbar_pose_xyw[2]),
                                               tool_pose_xy[1] - L1*cos(tbar_pose_xyw[2])]
                        
                        control_point2_real = [tool_pose_xy[0] - L2*sin(tbar_pose_xyw[2] + 1.8),
                                               tool_pose_xy[1] - L2*cos(tbar_pose_xyw[2] + 1.8)]
                        
                        self.target_point = [self.side_points[1][0],
                                             self.side_points[1][1]]
                        
                        tp_real = [(self.scale*self.target_point[0] - 300)/1000,
                                   (self.scale*self.target_point[1] + 320)/1000]
                        
                        control_point4_real = [tp_real[0] + L4*sin(tbar_pose_xyw[2]),
                                               tp_real[1] + L4*cos(tbar_pose_xyw[2])] 
                        
                        control_point3_real = [tp_real[0] + L3*sin(tbar_pose_xyw[2] - 1.5),
                                               tp_real[1] + L3*cos(tbar_pose_xyw[2] - 1.5)] 

                        points = np.array([
                            tool_pose_xy[:2],
                            control_point1_real,
                            control_point2_real,
                            control_point3_real,
                            control_point4_real,
                            tp_real
                        ])                      
                    elif dist1 >= dist2 and angle_dif > 0:
                        control_point1_real = [tool_pose_xy[0] + L1*sin(tbar_pose_xyw[2]),
                                               tool_pose_xy[1] + L1*cos(tbar_pose_xyw[2])]
                        
                        control_point2_real = [tool_pose_xy[0] + L2*sin(tbar_pose_xyw[2] - 1.8),
                                               tool_pose_xy[1] + L2*cos(tbar_pose_xyw[2] - 1.8)]
                        
                        self.target_point = [self.side_points[0][0],
                                             self.side_points[0][1]]
                        
                        tp_real = [(self.scale*self.target_point[0] - 300)/1000,
                                   (self.scale*self.target_point[1] + 320)/1000]
                        
                        control_point4_real = [tp_real[0] - L4*sin(tbar_pose_xyw[2]),
                                               tp_real[1] - L4*cos(tbar_pose_xyw[2])] 
                        
                        control_point3_real = [tp_real[0] - L3*sin(tbar_pose_xyw[2] + 1.5),
                                               tp_real[1] - L3*cos(tbar_pose_xyw[2] + 1.5)] 

                        points = np.array([
                            tool_pose_xy[:2],
                            control_point1_real,
                            control_point2_real,
                            control_point3_real,
                            control_point4_real,
                            tp_real
                        ])                        
                    else: # dist1 >= dist2 and angle_dif <= 0:
                        control_point1_real = [tool_pose_xy[0] + dist*sin(tbar_pose_xyw[2]),
                                               tool_pose_xy[1] + dist*cos(tbar_pose_xyw[2])]
                        
                        side_point_x = (self.scale*self.side_points[1][0] - 300)/1000
                        side_point_y = (self.scale*self.side_points[1][1] + 320)/1000

                        control_point2_real = [side_point_x + dist*sin(tbar_pose_xyw[2]),
                                               side_point_y + dist*cos(tbar_pose_xyw[2])]
                        
                        self.target_point[0] = self.side_points[1][0]
                        self.target_point[1] = self.side_points[1][1]

                        tp_real = [(self.scale*self.target_point[0] - 300)/1000,
                                   (self.scale*self.target_point[1] + 320)/1000]

                        points = np.array([
                             tool_pose_xy[:2], 
                             control_point1_real,
                             control_point2_real,
                             tp_real
                        ])                        

                    self.curve = np.fliplr(bezier_curve(points, num_points=50))
                    self.time_steps = len(self.curve)
                    self.has_trajectory = True

                for i in range(len(self.curve) - 1):
                    cv2.line(debug_image, (int((self.curve[i][1]*1000 + 300)/self.scale),int((self.curve[i][0]*1000 - 320)/self.scale)),
                                    (int((self.curve[i+1][1]*1000 + 300)/self.scale), int((self.curve[i+1][0]*1000 - 320)/self.scale)),
                                    (255, 255, 0), 2*self.radius) 
                    
                if self.time_steps > 1:
                    self.time_steps -= 1
                    self.current_step = len(self.curve) - self.time_steps
                elif self.time_steps == 1:
                    self.first_stage = False
                    self.third_stage = False
                    self.fourth_stage = False
                    self.sixth_stage = False
                    self.seventh_stage = False
                    self.eighth_stage = False
                    self.has_trajectory = False

                    if sum < self.goal_sum:
                        if abs(self.tbar_goal_pose[2] - tbar_pose_xyw[2]) > 0.01745: # 1 deg
                            self.second_stage = True
                            self.fifth_stage = False
                        elif abs(tbar_pose_xyw[1] - self.tbar_goal_pose[1]) > 0.001:
                            self.second_stage = False
                            self.fifth_stage = True
                        else:
                            self.second_stage = False
                            self.fifth_stage = False       
                    else:
                        self.second_stage = False
                        self.fifth_stage = False                

                self.target_pose.position.x = self.curve[self.current_step][0]
                self.target_pose.position.y = self.curve[self.current_step][1]
                self.pub_pose.publish(self.target_pose)

            elif self.ninth_stage:
                if self.has_trajectory == False:

                    dist1 = sqrt((tool_pose_xy[0] - (self.scale*self.side_points[0][0] - 300)/1000)**2 +
                                (tool_pose_xy[1] - (self.scale*self.side_points[0][1] + 320)/1000)**2)
                    
                    dist2 = sqrt((tool_pose_xy[0] - (self.scale*self.side_points[1][0] - 300)/1000)**2 +
                                (tool_pose_xy[1] - (self.scale*self.side_points[1][1] + 320)/1000)**2)
                    
                    B = 120
                    C = 100
                    L = 52
                    L1 = 0.06
                    L2 = 0.1
                    L3 = 0.15
                    if dist1 < dist2:
                        control_point1_real = [tool_pose_xy[0] - L1*sin(tbar_pose_xyw[2]),
                                               tool_pose_xy[1] - L1*cos(tbar_pose_xyw[2])]
                        
                        control_point2_real = [tool_pose_xy[0] - L2*sin(tbar_pose_xyw[2] + 1.0),
                                               tool_pose_xy[1] - L2*cos(tbar_pose_xyw[2] + 1.0)]
                                    
                        control_point3_real = [tool_pose_xy[0] - L3*sin(tbar_pose_xyw[2] + 2.3),
                                               tool_pose_xy[1] - L3*cos(tbar_pose_xyw[2] + 2.3)]  
                        
                        control_point4 = [int(cos(th1)*(-C) + dx1 + self.C_W + 1000*x1/self.scale),
                                          int(sin(th1)*(-C) + dy1 + (1000*y1 - 320)/self.scale)]
                        
                        tip_point = [int(cos(th1)*(-L) + dx1 + self.C_W + 1000*x1/self.scale),
                                     int(sin(th1)*(-L) + dy1 + (1000*y1 - 320)/self.scale)] 
                    else:
                        control_point1_real = [tool_pose_xy[0] + L1*sin(tbar_pose_xyw[2]),
                                               tool_pose_xy[1] + L1*cos(tbar_pose_xyw[2])]
        
                        control_point2_real = [tool_pose_xy[0] + L2*sin(tbar_pose_xyw[2] - 1.0),
                                               tool_pose_xy[1] + L2*cos(tbar_pose_xyw[2] - 1.0)]
                                    
                        control_point3_real = [tool_pose_xy[0] + L3*sin(tbar_pose_xyw[2] - 2.3),
                                               tool_pose_xy[1] + L3*cos(tbar_pose_xyw[2] - 2.3)]
                        
                        control_point4 = [int(cos(th1)*C + dx1 + self.C_W + 1000*x1/self.scale),
                                          int(sin(th1)*C + dy1 + (1000*y1 - 320)/self.scale)]
                        
                        tip_point = [int(cos(th1)*L + dx1 + self.C_W + 1000*x1/self.scale),
                                     int(sin(th1)*L + dy1 + (1000*y1 - 320)/self.scale)]

                    control_point4_real = [(self.scale*control_point4 [0] - 300)/1000, (control_point4[1] + 320)/1000]  
                    tip_point_real = [(self.scale*tip_point[0] - 300)/1000, (self.scale*tip_point[1] + 320)/1000]  
                    points = np.array([
                        tool_pose_xy[:2], # start
                        control_point1_real, # control 1
                        control_point2_real,
                        control_point3_real,
                        control_point4_real,
                        tip_point_real # end
                    ])            

                    self.curve = np.fliplr(bezier_curve(points, num_points=50))
                    self.time_steps = len(self.curve)
                    self.has_trajectory = True

                for i in range(len(self.curve) - 1):
                    cv2.line(debug_image, (int((self.curve[i][1]*1000 + 300)/self.scale), int((self.curve[i][0]*1000 - 320)/self.scale)),
                                    (int((self.curve[i+1][1]*1000 + 300)/self.scale), int((self.curve[i+1][0]*1000 - 320)/self.scale)),
                                    (255, 255, 0), 2*self.radius)
                    
                if self.time_steps > 1:
                    self.time_steps -= 1
                    self.current_step = len(self.curve) - self.time_steps
                elif self.time_steps == 1:
                    self.first_stage = False
                    self.second_stage = False
                    self.third_stage = False
                    self.fourth_stage = False
                    self.fifth_stage = False
                    self.sixth_stage = False
                    self.seventh_stage = False
                    self.eighth_stage = True
                    self.ninth_stage = False
                    self.has_trajectory = False

                self.target_pose.position.x = self.curve[self.current_step][0]
                self.target_pose.position.y = self.curve[self.current_step][1]
                self.pub_pose.publish(self.target_pose)

            print(f"step {self.total_steps}, 2nd:{self.second_stage}, 3rd:{self.third_state}, 4th:{self.fourth_stage}, 5th:{self.fifth_stage}, 6th:{self.sixth_stage}, 7th:{self.seventh_stage}, 8th:{self.eighth_stage}, 9th:{self.ninth_stage}, has_traj:{self.has_trajectory}")

            cv2.circle(image, center=(x_tool, y_tool), radius=self.radius, color=(50, 50, 50), thickness=cv2.FILLED)
            cv2.circle(debug_image, center=(x_tool, y_tool), radius=self.radius, color=(50, 50, 50), thickness=cv2.FILLED)
            cv2.circle(debug_image, center=(self.side_points[0][0], self.side_points[0][1]), radius=2, color=(0, 200, 0), thickness=cv2.FILLED)
            cv2.circle(debug_image, center=(self.side_points[1][0], self.side_points[1][1]), radius=2, color=(0, 200, 0), thickness=cv2.FILLED)
            cv2.circle(debug_image, center=self.bottom_ponit, radius=2, color=(0, 200, 0), thickness=cv2.FILLED)
            cv2.circle(debug_image, center=self.upper_ponit, radius=2, color=(0, 200, 0), thickness=cv2.FILLED)

            # for debug1
            img_msg = bridge.cv2_to_imgmsg(debug_image)  
            self.pub_img.publish(img_msg) 

            # for debug2
            sum_image = bridge.cv2_to_imgmsg(self.tool_trajectory)  
            self.sum_img_pub.publish(sum_image)

            self.total_steps += 1

            if self.record_data:
                print('\033[32m'+f'RECORDING episode:{self.episode_index}, index:{self.index} sum:{sum}'+'\033[0m')

                if sum >= self.goal_sum:
                    self.success = True
                    self.done = True
                    self.record_data = False
                    print('\033[31m'+'SUCCESS!'+f': {sum}'+'\033[0m')
                else:
                    self.success = False

                self.df.loc[self.column_index] = [copy.copy(tool_pose_xy), copy.copy([self.target_pose.position.x, self.target_pose.position.y]), \
                                                  self.episode_index, self.frame_index, self.time_stamp, sum, self.done, self.success, self.index, 0]
                self.column_index += 1
                self.frame_index += 1
                self.time_stamp += 1/self.Hz
                self.index += 1

                self.start_recording = True

                self.wrist_camera_array.append(wrist_camera_image)
                self.top_camera_array.append(top_camera_image)
                image_temp = image[1:300,1:361,:]
                observation_image = cv2.resize(image_temp, (240, 200), cv2.INTER_LINEAR)
                self.state_image_array.append(observation_image)

            else:
                if(self.start_recording and self.data_recorded == False):
                    print('\033[31m'+'WRITING A PARQUET FILE'+'\033[0m')

                    if self.episode_index <= 9:
                        data_file_name = 'episode_00000' + str(self.episode_index) + '.parquet'
                        video_file_name = 'episode_00000' + str(self.episode_index) + '.mp4'
                    elif 9 < self.episode_index <= 99:
                        data_file_name = 'episode_0000' + str(self.episode_index) + '.parquet'
                        video_file_name = 'episode_0000' + str(self.episode_index) + '.mp4'
                    elif 99 < self.episode_index <= 999:
                        data_file_name = 'episode_000' + str(self.episode_index) + '.parquet'
                        video_file_name = 'episode_000' + str(self.episode_index) + '.mp4'
                    elif 999 < self.episode_index <= 9999:
                        data_file_name = 'episode_00' + str(self.episode_index) + '.parquet'
                        video_file_name = 'episode_00' + str(self.episode_index) + '.mp4'
                    else:
                        data_file_name = 'episode_0' + str(self.episode_index) + '.parquet'
                        video_file_name = 'episode_0' + str(self.episode_index) + '.mp4'

                    table = pa.Table.from_pandas(self.df)
                    pq.write_table(table, self.log_dir + data_file_name)
                    print("The parquet file is generated!")

                    
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    out1 = cv2.VideoWriter(self.wrist_vid_dir + video_file_name, fourcc, self.Hz, (240, 240))
                    for frame1 in self.wrist_camera_array:
                        out1.write(frame1)
                    out1.release()
                    print("The wrist video is generated!")
                    out2 = cv2.VideoWriter(self.top_vid_dir + video_file_name, fourcc, self.Hz, (240, 240))
                    for frame2 in self.top_camera_array:
                        out2.write(frame2)
                    out2.release()
                    print("The top video is generated!")
                    out3 = cv2.VideoWriter(self.state_vid_dir + video_file_name, fourcc, self.Hz, (240, 200))
                    for frame3 in self.state_image_array:
                        out3.write(frame3)
                    out3.release()
                    print("The state video is generated!")

                    self.data_recorded = True
                    rclpy.shutdown()

        else:
            print("Waiting Isaac Sim to startup...")

        if self.total_steps >700:
            rclpy.shutdown

if __name__ == '__main__':
    rclpy.init(args=None)

    get_poses_subscriber = Get_Poses_Subscriber()
    wrist_camera_subscriber = Wrist_Camera_Subscriber()
    top_camera_subscriber = Top_Camera_Subscriber()
    data_recorder = Data_Recorder()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(get_poses_subscriber)
    executor.add_node(wrist_camera_subscriber)
    executor.add_node(top_camera_subscriber)
    executor.add_node(data_recorder)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        rate = get_poses_subscriber.create_rate(2)
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        print("Ctrl+C pressed.")
    except rclpy.exceptions.ROSInterruptException:
        print("ROS shutdown triggered — stopping loop safely.")
    finally:
        executor.shutdown()
        rclpy.shutdown()
        executor_thread.join()
        print("ROS nodes stopped.")
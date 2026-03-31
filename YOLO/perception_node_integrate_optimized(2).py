#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PointStamped
from cv_bridge import CvBridge
from lab2_perception.msg import ObjectCoordinates
import tf2_ros
import tf2_geometry_msgs
from lab2_perception.cfg import PerceptionHSVConfig
from dynamic_reconfigure.server import Server

from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CameraInfo

# 导入 YOLO
from ultralytics import YOLO
from enum import Enum

class RobotState(Enum):
    SEARCHING = 0      # 搜索目标
    ALIGNING = 1       # 对准目标(旋转调整)
    APPROACHING = 2    # 接近目标(前进)
    REACHED = 3        # 到达目标


class PerceptionNode:
    def __init__(self):
        rospy.init_node('perception_node')
        
        self.bridge = CvBridge()
        self.latest_depth = None
        self.depth_colormap = None
               
        # 获取检测模式：'color' (默认) 或 'yolo'
        self.detect_mode = rospy.get_param('~mode', 'color')
        rospy.loginfo(f"当前感知模式为: {self.detect_mode.upper()}")

        # 如果是 YOLO 模式，则加载模型
        if self.detect_mode == 'yolo':
            model_path = os.path.expanduser("~/catkin_ws/models/yolo26s.pt")
            rospy.loginfo(f"正在加载 YOLO 模型: {model_path} ...")
            self.yolo_model = YOLO(model_path)
            rospy.loginfo("YOLO 模型加载完成！")

        # 使用动态重新配置
        self.hsv = dict()
        self.dr_srv = Server(PerceptionHSVConfig, self.reconfig_cb)
        self.display_scale = rospy.get_param("~display_scale", 0.5)
        
        # TF Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribers
        self.rgb_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        
        # Publisher
        self.coord_pub = rospy.Publisher('detected_object', ObjectCoordinates, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # 状态机相关
        self.current_state = RobotState.SEARCHING
        self.target_lost_count = 0
        self.max_lost_frames = 10  # 连续丢失多少帧后重新搜索
        self.target_seen_count = 0
        self.target_seen_required = rospy.get_param("~target_seen_required", 2)
        self.search_lock_until = None
        self.search_lock_duration = rospy.get_param("~search_lock_duration", 0.5)
        
        # 控制参数
        self.align_threshold = 0.15  # 对准阈值(归一化坐标)
        self.distance_threshold = 0.1  # 距离到达阈值(米)
        self.desired_distance = 0.5  # 期望距离(米)
        
        # PID控制参数
        self.angular_kp = 0.5
        self.linear_kp = 0.5
        self.max_angular_speed = 0.3
        self.max_linear_speed = 0.2

        # 误差滤波与稳定控制参数
        self.error_alpha = 0.2
        self.horizontal_error_filt = 0.0
        self.angular_deadband = 0.03
        self.min_angular_speed = 0.03
        self.prev_horizontal_error = None
        # 分段控制阈值
        self.angular_fast_threshold = 0.12
        self.angular_slow_threshold = 0.05

        # 对准稳定计数
        self.align_ok_count = 0
        self.align_bad_count = 0
        self.align_ok_required = 3
        self.align_bad_required = 3

        # 距离误差滤波与限幅
        self.dist_alpha = 0.2
        self.distance_error_filt = 0.0
        self.max_distance_error = 1.0

        # 初始化内参变量
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # 添加 CameraInfo 订阅者
        self.info_sub = rospy.Subscriber("/camera/rgb/camera_info", CameraInfo, self.camera_info_callback)

        # 发布 odom 下的点（可选：用于调试查看）
        self.point_odom_pub = rospy.Publisher('/detected_point_odom', PointStamped, queue_size=10)

        # marker id 计数
        self.marker_id = 0
        
        rospy.loginfo("Perception Node Started")

    def camera_info_callback(self, msg):
        self.fx = msg.K[0]
        self.cx = msg.K[2]
        self.fy = msg.K[4]
        self.cy = msg.K[5]
        self.info_sub.unregister()
        rospy.loginfo(f"Camera Info received: fx={self.fx}, cx={self.cx}")

    def reconfig_cb(self, config, level):
        self.hsv["lower"] = np.array([config.lower_h, config.lower_s, config.lower_v])
        self.hsv["upper"] = np.array([config.upper_h, config.upper_s, config.upper_v])
        self.display_scale = config.display_scale
        return config

    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_image(cv_image)
        except Exception as e:
            rospy.logerr(f"RGB error: {e}")

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            if self.latest_depth is not None:
                depth_normalized = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        except Exception as e:
            rospy.logerr(f"Depth error: {e}")

    def process_image(self, rgb_image):
        cmd = Twist()  # 默认停止命令
        target_found = False
        cX, cY = None, None

        hsv_result = None
        edges = None

        # ==================== 第 1 部分：获取二维坐标 cX, cY ====================
        
        # 模式 1：基于颜色的检测
        if self.detect_mode == 'color':
            hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_image, self.hsv["lower"], self.hsv["upper"])        
            hsv_result = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
            
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_image, 100, 200)
            contours_canny, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_image, contours_canny, -1, (0, 255, 0), 3) # 把 canny 画在主图上
            
            contours_mask, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours_mask:
                c = max(contours_mask, key=cv2.contourArea)
                if cv2.contourArea(c) > 100:
                    target_found = True
                    cv2.drawContours(rgb_image, [c], -1, (0, 0, 255), 3)
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

        # 模式 2：基于 YOLO 的检测
        elif self.detect_mode == 'yolo':
            results = self.yolo_model(rgb_image, conf=0.5, verbose=False)
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                # 默认取检测到的第一个目标
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 计算中心点
                cX = int((x1 + x2) / 2)
                cY = int((y1 + y2) / 2)
                target_found = True
                
                # 画框和名字
                cls_id = int(box.cls[0])
                cls_name = self.yolo_model.names[cls_id]
                cv2.rectangle(rgb_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(rgb_image, f"{cls_name}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                rospy.loginfo_throttle(1.0, f"YOLO 锁定了目标: 【{cls_name}】")

        # ==================== 第 2 部分：3D坐标计算与状态机 ====================
        
        if target_found and cX is not None and cY is not None:
            self.target_lost_count = 0
            
            # 画中心小白点
            cv2.circle(rgb_image, (cX, cY), 7, (255, 255, 255), -1)
            
            # --- Calculate 3D Coordinates ---
            if self.latest_depth is not None:
                h, w = self.latest_depth.shape
                if 0 <= cX < w and 0 <= cY < h:
                    X, Y, Z = self.calculate_3d_coordinates(cX, cY, self.latest_depth)

                    # 转到 odom 并发布 RViz Marker
                    self.publish_point_in_odom(X, Y, Z)
                    
                    # --- Publish Coordinates ---
                    msg = ObjectCoordinates()
                    msg.x = X
                    msg.y = Y
                    msg.z = Z
                    self.coord_pub.publish(msg)
                    
                    # --- State Machine Control ---
                    image_center_x = w / 2.0
                    horizontal_error_raw = (cX - image_center_x) / image_center_x
                    self.horizontal_error_filt = (
                        self.error_alpha * horizontal_error_raw
                        + (1.0 - self.error_alpha) * self.horizontal_error_filt
                    )
                    horizontal_error = self.horizontal_error_filt
                    current_distance = Z
                    depth_valid = (not np.isnan(current_distance)) and (current_distance > 0.0)
                    if depth_valid:
                        distance_error = current_distance - self.desired_distance
                    else:
                        distance_error = 0.0
                    
                    if self.current_state == RobotState.SEARCHING:
                        self.target_seen_count += 1
                        if self.target_seen_count == 1:
                            self.search_lock_until = rospy.Time.now() + rospy.Duration(self.search_lock_duration)
                        if self.target_seen_count >= self.target_seen_required:
                            self.current_state = RobotState.ALIGNING
                            rospy.loginfo("State: SEARCHING -> ALIGNING")
                        else:
                            if self.search_lock_until is not None and rospy.Time.now() < self.search_lock_until:
                                cmd.angular.z = 0.0
                                cmd.linear.x = 0.0
                    
                    elif self.current_state == RobotState.ALIGNING:
                        if abs(horizontal_error) > self.align_threshold:
                            self.align_ok_count = 0
                            self.align_bad_count += 1
                            cmd.angular.z = self.compute_angular_cmd(horizontal_error)
                            cmd.linear.x = 0.0
                        else:
                            self.align_bad_count = 0
                            self.align_ok_count += 1
                            if self.align_ok_count >= self.align_ok_required:
                                self.current_state = RobotState.APPROACHING
                                self.align_ok_count = 0
                                rospy.loginfo("State: ALIGNING -> APPROACHING")
                    
                    elif self.current_state == RobotState.APPROACHING:
                        if abs(horizontal_error) > self.align_threshold * 2:
                            self.align_bad_count += 1
                            if self.align_bad_count >= self.align_bad_required:
                                self.align_ok_count = 0
                                self.current_state = RobotState.ALIGNING
                                rospy.loginfo("State: APPROACHING -> ALIGNING (lost alignment)")
                        elif depth_valid and abs(distance_error) < self.distance_threshold: # 🌟 保护1：必须测到距离才算到达
                            self.current_state = RobotState.REACHED
                            rospy.loginfo("State: APPROACHING -> REACHED")
                        else:
                            cmd.angular.z = self.compute_angular_cmd(horizontal_error, scale=0.5)
                            
                            # 🌟 保护2：核心修复！测不到距离(Z=0)时，绝对不倒车！
                            if not depth_valid:
                                cmd.linear.x = 0.0  
                            else:
                                distance_error = max(-self.max_distance_error, min(self.max_distance_error, distance_error))
                                self.distance_error_filt = (
                                    self.dist_alpha * distance_error
                                    + (1.0 - self.dist_alpha) * self.distance_error_filt
                                )
                                if self.distance_error_filt > 0:
                                    cmd.linear.x = self.linear_kp * self.distance_error_filt
                                    cmd.linear.x = max(0, min(self.max_linear_speed, cmd.linear.x))
                                else:
                                    cmd.linear.x = self.linear_kp * self.distance_error_filt
                                    cmd.linear.x = max(-self.max_linear_speed * 0.5, min(0, cmd.linear.x))
                    
                    elif self.current_state == RobotState.REACHED:
                        if abs(horizontal_error) > self.align_threshold:
                            cmd.angular.z = self.compute_angular_cmd(horizontal_error, scale=0.3)
                        # 🌟 保护3：测不到距离时，不要退出到达状态
                        if depth_valid and abs(distance_error) > self.distance_threshold: 
                            self.current_state = RobotState.APPROACHING
                            rospy.loginfo("State: REACHED -> APPROACHING (distance changed)")
                        cmd.linear.x = 0.0
                    
                    # --- TF Transformation ---
                    try:
                        rospy.loginfo_throttle(0.5, f"Camera-frame coordinates: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")
                        point_stamped = PointStamped()
                        point_stamped.header.frame_id = "camera_rgb_optical_frame"
                        point_stamped.header.stamp = rospy.Time(0)
                        point_stamped.point.x = X
                        point_stamped.point.y = Y
                        point_stamped.point.z = Z
                        
                        if self.tf_buffer.can_transform("base_footprint", point_stamped.header.frame_id, rospy.Time(0), rospy.Duration(0.1)):
                            point_base = self.tf_buffer.transform(point_stamped, "base_footprint", rospy.Duration(0.1))
                            rospy.loginfo_throttle(0.5, f"Robot-frame: X={point_base.point.x:.3f}, Y={point_base.point.y:.3f}, Z={point_base.point.z:.3f}")
                    except Exception as e:
                        rospy.logwarn_throttle(1.0, f"TF Error: {e}")
                    
                    # Display info
                    state_text = f"State: {self.current_state.name}"
                    coord_text = f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}"
                    error_text = f"H_Err:{horizontal_error:.3f} D_Err:{distance_error:.2f}"
                    
                    cv2.putText(rgb_image, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(rgb_image, coord_text, (cX - 50, cY - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(rgb_image, error_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                    
                    cv2.line(rgb_image, (int(image_center_x) - 20, int(h/2)), (int(image_center_x) + 20, int(h/2)), (0, 255, 255), 2)
                    cv2.line(rgb_image, (int(image_center_x), int(h/2) - 20), (int(image_center_x), int(h/2) + 20), (0, 255, 255), 2)

                    # 记录上一帧误差用于过零保护
                    self.prev_horizontal_error = horizontal_error
        
        # 目标丢失处理
        if not target_found:
            self.target_lost_count += 1
            self.target_seen_count = 0
            if self.target_lost_count > self.max_lost_frames:
                if self.current_state != RobotState.SEARCHING:
                    rospy.loginfo(f"State: {self.current_state.name} -> SEARCHING (target lost)")
                    self.current_state = RobotState.SEARCHING
                cmd.angular.z = 0.3
                cmd.linear.x = 0.0
        
        # 发布控制命令
        self.cmd_vel_pub.publish(cmd)
        
        # Show windows
        cv2.imshow("Detection Result (RGB)", self.resize(rgb_image))
        if self.detect_mode == 'color' and hsv_result is not None and edges is not None:
            cv2.imshow("HSV Result", self.resize(hsv_result))
            cv2.imshow("Canny Edges", self.resize(edges))
        if self.depth_colormap is not None:
            cv2.imshow("Depth Colormap", self.resize(self.depth_colormap))
        
        cv2.waitKey(1)

    def resize(self,image):
        self.display_scale = rospy.get_param("~display_scale", 0.5)
        return cv2.resize(image, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)

    def compute_angular_cmd(self, horizontal_error, scale=1.0):
        # 过零保护：误差方向翻转时立即停，避免摆头过冲
        if self.prev_horizontal_error is not None:
            if horizontal_error * self.prev_horizontal_error < 0 and abs(horizontal_error) < self.angular_fast_threshold:
                return 0.0

        err = abs(horizontal_error)

        # 小误差死区
        if err < self.angular_slow_threshold or err < self.angular_deadband:
            return 0.0

        # 中误差区：慢速微调（不启用最小角速度）
        if err < self.angular_fast_threshold:
            cmd = -self.angular_kp * 0.5 * scale * horizontal_error
            max_slow = self.max_angular_speed * 0.4
            return max(-max_slow, min(max_slow, cmd))

        # 大误差区：快速转向（启用最小角速度）
        cmd = -self.angular_kp * scale * horizontal_error
        if abs(cmd) < self.min_angular_speed:
            cmd = self.min_angular_speed * (1 if cmd > 0 else -1)
        return max(-self.max_angular_speed, min(self.max_angular_speed, cmd))

    def calculate_3d_coordinates(self, u, v, depth_image):
        """
        基于 ROI (Region of Interest) 中位数滤波的三维坐标计算
        不再只取1个像素点，而是取中心周围的区域过滤噪点，极大提升测距稳定性
        """
        # 1. 确定相机内参
        if self.fx is None:
            fx, fy, cx, cy = 554.25, 554.25, 320.5, 240.5
        else:
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        # 2. 获取深度图的尺寸，防止边界溢出报错
        h, w = depth_image.shape

        # 3. 设定 ROI 区域大小，这里取目标中心点周围 20x20 像素的范围
        box_size = 10  # 半径为 10 像素
        u_min = max(0, u - box_size)
        u_max = min(w - 1, u + box_size)
        v_min = max(0, v - box_size)
        v_max = min(h - 1, v + box_size)

        # 4. 提取该小方块区域内的所有深度数据
        roi = depth_image[v_min:v_max, u_min:u_max]

        # 5. 核心抗噪：过滤掉区域里的无效深度值 (比如背景的0，或者反光造成的 NaN)
        valid_depths = roi[(roi > 0.0) & (~np.isnan(roi))]

        # 6. 取有效深度的“中位数”作为最终的 Z 值
        if len(valid_depths) > 0:
            Z = float(np.median(valid_depths))
        else:
            # 如果整个 20x20 的区域都测不到深度（说明真的在盲区），为了安全返回 0.0
            return 0.0, 0.0, 0.0

        # 7. 根据针孔相机模型反推真实 3D 物理坐标
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return float(X), float(Y), float(Z)

    def publish_point_in_odom(self, X, Y, Z):
        point_cam = PointStamped()
        point_cam.header.frame_id = "camera_rgb_optical_frame"
        point_cam.header.stamp = rospy.Time(0)
        point_cam.point.x = float(X)
        point_cam.point.y = float(Y)
        point_cam.point.z = float(Z)

        try:
            if not self.tf_buffer.can_transform("odom", point_cam.header.frame_id, rospy.Time(0), rospy.Duration(0.2)):
                return
            point_odom = self.tf_buffer.transform(point_cam, "odom", rospy.Duration(0.2))
            self.point_odom_pub.publish(point_odom)
        except Exception as e:
            pass

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    node = PerceptionNode()
    node.run()

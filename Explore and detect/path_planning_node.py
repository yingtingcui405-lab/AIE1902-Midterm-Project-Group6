#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
import numpy as np
import heapq
import math
import tf.transformations

class SimplePathPlanner:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        if not self.test_mode:
            rospy.init_node('eai_path_planner')
            
            # 订阅地图
            self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback_consider_robot)
            # 订阅目标点
            self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
            # 订阅当前位置
            self.pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
            
            # 发布路径
            self.path_pub = rospy.Publisher('/my_planned_path', Path, queue_size=1)
            # 发布控制指令
            self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
            
            # 定时器：控制循环 (10Hz)
            self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        self.map_data = None
        self.current_pose = None
        self.current_goal = None  # 保存当前目标位置
        
        # 路径跟踪相关
        self.current_path = []
        self.is_tracking = False
        self.path_index = 0
        
        # 脱困控制状态
        self.in_escape_mode = False      # 是否处于脱困模式
        self.escape_duration = 0         # 脱困已持续的 control_loop 调用次数（每 0.1s +1）
        self.fallback_state = 0          # 0: 后退, 1: 转向
        self.fallback_counter = 0        # 当前子阶段计数器
    
    def map_callback_consider_robot(self, msg):
        # 直接保存地图数据，不进行膨胀处理
        self.map_data = msg
 
    def map_callback(self, msg):
        self.map_data = msg
        
    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose
        
    def goal_callback(self, msg):
        if self.map_data is None or self.current_pose is None:
            rospy.logwarn("Map or pose not ready")
            return
            
        # ========================================
        # 学生在这里实现自己的路径规划算法
        # ========================================
        rospy.loginfo("Received goal, planning path...")
        
        # 停止当前跟踪
        self.is_tracking = False
        self.cmd_pub.publish(Twist()) # 停车
        
        # 保存当前目标位置
        self.current_goal = msg.pose
        
        path_msg = self.plan_path(self.current_pose, self.current_goal)
        
        if path_msg and len(path_msg.poses) > 0:
            rospy.loginfo("Path found! Publishing...")
            self.path_pub.publish(path_msg)
            
            # 开始跟踪
            self.current_path = path_msg.poses
            self.path_index = 0
            self.is_tracking = True
        else:
            rospy.logwarn("Failed to find a path.")

    def control_loop(self, event):
        if not self.is_tracking or self.current_pose is None or not self.current_path:
            return

        # 检查是否到达目标
        target_pose = self.current_path[-1].pose.position
        dx = target_pose.x - self.current_pose.position.x
        dy = target_pose.y - self.current_pose.position.y
        dist_to_goal = math.sqrt(dx**2 + dy**2)
        
        if dist_to_goal < 0.1:  # 10cm 容差
            rospy.loginfo("Goal reached!")
            self.is_tracking = False
            twist = Twist()  # 停车
            self.cmd_pub.publish(twist)
            return

        # 检查是否接近障碍物
        if self.is_near_obstacle():
            rospy.loginfo("Obstacle detected, using DWA for local planning")
            # 使用DWA算法进行局部路径规划
            twist = self.dwa()
            if twist:
                self.cmd_pub.publish(twist)
            else:
                # DWA失败，使用默认控制
                self.default_control()
        else:
            # 没有障碍物，使用全局路径跟踪
            self.track_global_path()
            
    def track_global_path(self):
        """
        跟踪全局路径
        """
        # 选择当前路径点
        if self.path_index < len(self.current_path):
            target_pose = self.current_path[self.path_index].pose.position
        else:
            target_pose = self.current_path[-1].pose.position

        # 计算到当前目标点的距离
        dx = target_pose.x - self.current_pose.position.x
        dy = target_pose.y - self.current_pose.position.y
        dist_to_target = math.sqrt(dx**2 + dy**2)

        # 如果接近当前目标点，移动到下一个点
        if dist_to_target < 0.2:  # 20cm 容差
            if self.path_index < len(self.current_path) - 1:
                self.path_index += 1
                target_pose = self.current_path[self.path_index].pose.position
                dx = target_pose.x - self.current_pose.position.x
                dy = target_pose.y - self.current_pose.position.y

        # 计算目标角度
        target_angle = math.atan2(dy, dx)

        # 获取当前朝向
        q = self.current_pose.orientation
        _, _, current_yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        # 角度差归一化到 [-π, π]
        angle_diff = target_angle - current_yaw
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        # 控制律
        twist = Twist()
        twist.angular.z = max(-1.0, min(1.0, angle_diff * 2.0))  # P 控制
        twist.linear.x = max(0.0, 0.2 - abs(angle_diff) * 0.5)   # 转弯减速

        self.cmd_pub.publish(twist)
        
    def default_control(self):
        """
        默认控制逻辑，当DWA失败时使用
        """
        # 简单的后退和转向策略
        twist = Twist()
        
        if self.fallback_state == 0:  # 后退
            twist.linear.x = -0.1
            twist.angular.z = 0.0
            self.fallback_counter += 1
            if self.fallback_counter >= 10:  # 后退1秒
                self.fallback_state = 1
                self.fallback_counter = 0
        else:  # 转向
            twist.linear.x = 0.0
            twist.angular.z = 0.5
            self.fallback_counter += 1
            if self.fallback_counter >= 10:  # 转向1秒
                self.fallback_state = 0
                self.fallback_counter = 0
        
        self.cmd_pub.publish(twist)
        
    def plan_path(self, start_pose, goal_pose):
        """
        A*算法路径规划：考虑障碍物，生成最优路径
        """
        start_grid = self.world_to_grid(start_pose.position.x, start_pose.position.y)
        goal_grid = self.world_to_grid(goal_pose.position.x, goal_pose.position.y)
        
        # 检查起点终点是否在地图范围内
        width = self.map_data.info.width
        height = self.map_data.info.height
        if not (0 <= start_grid[0] < width and 0 <= start_grid[1] < height):
             rospy.logwarn("Start is out of map bounds")
             return Path()
        if not (0 <= goal_grid[0] < width and 0 <= goal_grid[1] < height):
             rospy.logwarn("Goal is out of map bounds")
             return Path()

        # 使用A*算法生成路径
        path_points = self.astar(start_grid, goal_grid)
        
        if not path_points:
            rospy.logwarn("Failed to find a path")
            return Path()

        # 重构路径消息
        path_msg = Path()
        path_msg.header.frame_id = "map"
        if not self.test_mode:
            path_msg.header.stamp = rospy.Time.now()
        
        for grid_x, grid_y in path_points:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            if not self.test_mode:
                pose.header.stamp = rospy.Time.now()
            
            wx, wy = self.grid_to_world(grid_x, grid_y)
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0 
            
            path_msg.poses.append(pose)
            
        return path_msg
        
    def astar(self, start, goal):
        """
        A*算法实现
        
        Args:
            start: 起点栅格坐标 (x, y)
            goal: 终点栅格坐标 (x, y)
            
        Returns:
            路径点列表，按顺序从起点到终点
        """
        # 定义移动方向：8个方向，包括对角线
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # 上下左右
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
        ]
        
        # 启发函数：曼哈顿距离
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # 开放列表：使用优先队列
        open_list = []
        heapq.heappush(open_list, (0, start))
        
        # 关闭列表：已访问的节点
        closed_list = set()
        
        # 父节点映射，用于回溯路径
        came_from = {}
        
        # g得分：从起点到当前节点的代价
        g_score = {start: 0}
        
        # f得分：g得分 + 启发函数估计的代价
        f_score = {start: heuristic(start, goal)}
        
        while open_list:
            # 取出f得分最低的节点
            current_f, current = heapq.heappop(open_list)
            
            # 如果到达目标
            if current == goal:
                # 回溯构建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                # 反转路径，使其从起点到终点
                path.reverse()
                return path
            
            # 将当前节点加入关闭列表
            closed_list.add(current)
            
            # 探索所有可能的移动方向
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 检查邻居是否有效
                if not self.is_valid(neighbor[0], neighbor[1]):
                    continue
                
                # 检查邻居是否已在关闭列表中
                if neighbor in closed_list:
                    continue
                
                # 计算从起点到邻居的g得分
                # 对角线移动的代价为√2，约1.414
                if dx != 0 and dy != 0:
                    tentative_g = g_score[current] + 1.414
                else:
                    tentative_g = g_score[current] + 1.0
                
                # 如果邻居不在开放列表中，或新的g得分更低
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # 更新父节点
                    came_from[neighbor] = current
                    # 更新g得分和f得分
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    # 将邻居加入开放列表
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
        
        # 没有找到路径
        return []
        
    def dwa(self):
        """
        DWA (Dynamic Window Approach) 算法实现
        
        Returns:
            Twist: 最佳速度指令
        """
        # DWA参数
        max_speed = 0.3  # 最大线速度 (m/s)
        min_speed = -0.15  # 最小线速度 (m/s)
        max_yaw_rate = 1.5  # 最大角速度 (rad/s)
        max_accel = 0.2  # 最大线加速度 (m/s²)
        max_delta_yaw_rate = 0.8  # 最大角加速度 (rad/s²)
        dt = 0.1  # 时间步长 (s)
        predict_time = 0.5  # 预测时间 (s)
        
        # 目标点：使用当前路径的下一个点
        if len(self.current_path) > self.path_index + 1:
            target_pose = self.current_path[self.path_index + 1].pose.position
        else:
            target_pose = self.current_path[-1].pose.position
        
        # 生成速度窗口
        v_min = min_speed
        v_max = max_speed
        w_min = -max_yaw_rate
        w_max = max_yaw_rate
        
        # 速度分辨率
        v_resolution = 0.05
        w_resolution = 0.1
        
        # 最佳轨迹和评分
        best_score = -float('inf')
        best_twist = None
        
        # 遍历所有可能的速度组合
        v = v_min
        while v <= v_max:
            w = w_min
            while w <= w_max:
                # 模拟轨迹
                trajectory = self.simulate_trajectory(v, w, dt, predict_time)
                
                # 评估轨迹
                score = self.evaluate_trajectory(trajectory, target_pose)
                
                # 更新最佳轨迹
                if score > best_score:
                    best_score = score
                    best_twist = Twist()
                    best_twist.linear.x = v
                    best_twist.angular.z = w
                
                w += w_resolution
            v += v_resolution
        
        return best_twist
        
    def simulate_trajectory(self, v, w, dt, predict_time):
        """
        模拟轨迹
        
        Args:
            v: 线速度
            w: 角速度
            dt: 时间步长
            predict_time: 预测时间
            
        Returns:
            轨迹点列表
        """
        trajectory = []
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        _, _, yaw = tf.transformations.euler_from_quaternion(
            [self.current_pose.orientation.x, self.current_pose.orientation.y, 
             self.current_pose.orientation.z, self.current_pose.orientation.w]
        )
        
        time = 0
        while time < predict_time:
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            yaw += w * dt
            trajectory.append((x, y))
            time += dt
        
        return trajectory
        
    def evaluate_trajectory(self, trajectory, target_pose):
        """
        评估轨迹
        
        Args:
            trajectory: 轨迹点列表
            target_pose: 目标位置
            
        Returns:
            轨迹评分
        """
        # 距离目标的代价
        final_x, final_y = trajectory[-1]
        dist_to_target = math.sqrt(
            (final_x - target_pose.x)**2 + (final_y - target_pose.y)**2
        )
        target_cost = 1.0 / (dist_to_target + 0.001)  # 距离越近，代价越小
        
        # 障碍物代价
        obstacle_cost = 0.0
        min_obstacle_dist = float('inf')
        
        for (x, y) in trajectory:
            # 检查该点是否接近障碍物
            grid = self.world_to_grid(x, y)
            if grid:
                gx, gy = grid
                # 检查该点是否有效（是否是障碍物）
                if not self.is_valid(gx, gy):
                    obstacle_cost = 100.0  # 碰撞惩罚
                    break
                # 计算到障碍物的距离
                dist = self.calculate_obstacle_distance(gx, gy)
                min_obstacle_dist = min(min_obstacle_dist, dist)
        
        if obstacle_cost == 0.0:
            # 平滑的障碍物代价，距离越近代价越大，但不是突变
            obstacle_cost = 1.0 / (min_obstacle_dist + 0.001)
        
        # 速度代价：鼓励较高的速度
        speed_cost = 0.0  # 这里简化处理，不考虑速度代价
        
        # 总评分
        # 权重
        weight_target = 1.0
        weight_obstacle = 5.0  # 减少障碍物代价的权重，使其不那么保守
        weight_speed = 0.1
        
        score = weight_target * target_cost - weight_obstacle * obstacle_cost + weight_speed * speed_cost
        
        return score
        
    def calculate_obstacle_distance(self, grid_x, grid_y):
        """
        计算到最近障碍物的距离
        
        Args:
            grid_x: 栅格x坐标
            grid_y: 栅格y坐标
            
        Returns:
            到最近障碍物的距离
        """
        if not self.map_data:
            return float('inf')
        
        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        
        min_dist = float('inf')
        
        # 搜索周围一定范围内的障碍物
        search_radius = 5  # 搜索半径（栅格）
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    index = ny * width + nx
                    val = self.map_data.data[index]
                    if val > 50 or val == -1:  # 障碍物
                        dist = math.sqrt(dx**2 + dy**2) * resolution
                        min_dist = min(min_dist, dist)
        
        return min_dist
        
    def is_near_obstacle(self):
        """
        检查机器人是否接近障碍物
        
        Returns:
            bool: 如果机器人接近障碍物，返回 True
        """
        if not self.map_data or not self.current_pose:
            return False
        
        # 机器人尺寸估计：直径约 0.4m，半径 0.2m
        # 安全距离：机器人半径 + 0.1m 缓冲区 = 0.3m
        safety_distance = 0.25  # 减小安全距离，减少误触发
        
        # 检查机器人周围的多个点，形成一个安全区域
        # 只检查前方关键点，减少误触发
        check_points = [
            (safety_distance, 0),  # 正前方
        ]
        
        # 障碍物检测计数器
        obstacle_count = 0
        
        for dx, dy in check_points:
            # 转换到地图坐标系
            wx = self.current_pose.position.x + dx
            wy = self.current_pose.position.y + dy
            
            # 转换到栅格坐标
            grid = self.world_to_grid(wx, wy)
            if grid:
                gx, gy = grid
                # 检查该点是否有效（是否是障碍物）
                if not self.is_valid(gx, gy):
                    obstacle_count += 1
        
        # 只有当所有检查点都检测到障碍物时才认为接近障碍物
        return obstacle_count >= len(check_points)
        
    def world_to_grid(self, world_x, world_y):
        if not self.map_data:
            return None
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        resolution = self.map_data.info.resolution
        
        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)
        return (grid_x, grid_y)
        
    def grid_to_world(self, grid_x, grid_y):
        if not self.map_data:
            return None
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        resolution = self.map_data.info.resolution
        
        world_x = grid_x * resolution + origin_x + resolution / 2.0
        world_y = grid_y * resolution + origin_y + resolution / 2.0
        return (world_x, world_y)
        
    def is_valid(self, grid_x, grid_y):
        if not self.map_data:
            return False
        width = self.map_data.info.width
        height = self.map_data.info.height
        
        if grid_x < 0 or grid_x >= width or grid_y < 0 or grid_y >= height:
            return False
            
        # Check occupancy
        # data is row-major: index = y * width + x
        index = grid_y * width + grid_x
        # Occupancy probability [0, 100]. -1 is unknown.
        # Threshold > 50 considered occupied. Treat unknown as free or occupied? 
        # Usually treat unknown as free in exploration, but safe to treat as obstacle here?
        # Let's treat > 50 as obstacle. -1 (unknown) is risky, let's treat it as obstacle for safety.
        val = self.map_data.data[index]
        if val > 50 or val == -1: 
            return False
            
        return True


if __name__ == '__main__':
    try:
        planner = SimplePathPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
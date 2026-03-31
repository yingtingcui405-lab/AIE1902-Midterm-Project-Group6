#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import yaml
import os
import sys
import math

class NBVSearchNode:
    def __init__(self):
        rospy.init_node('nbv_search_node', anonymous=True)
        
        # --- 配置参数 ---
        self.map_name = "final_map" 
        script_dir = os.path.dirname(os.path.realpath(__file__))
        
        # 智能查找工作空间根目录
        ws_root = os.path.abspath(os.path.join(script_dir, '../../../'))
        if os.path.basename(ws_root) != 'catkin_ws':
            ws_root_alt = os.path.abspath(os.path.join(script_dir, '../../'))
            if os.path.basename(ws_root_alt) == 'catkin_ws':
                ws_root = ws_root_alt
        
        search_paths = [
            os.path.join(ws_root, "map", self.map_name),
            os.path.join(script_dir, self.map_name),
            os.path.join(script_dir, "map", self.map_name)
        ]
        
        self.yaml_path = None
        self.pgm_path = None
        
        for base_path in search_paths:
            if os.path.exists(base_path + ".yaml") and os.path.exists(base_path + ".pgm"):
                self.yaml_path = base_path + ".yaml"
                self.pgm_path = base_path + ".pgm"
                break
        
        if not self.yaml_path:
            rospy.logerr("❌ 未找到地图文件！请确认 map/final_map.yaml 存在。")
            sys.exit(1)

        # --- 状态变量 ---
        self.map_data = None       # 原始图像 (numpy)
        self.resolution = 0.0
        self.origin = [0, 0, 0]
        self.width = 0
        self.height = 0
        
        # NBV 核心参数
        self.RAY_COUNT = 36          # 射线数量
        self.RAY_LENGTH_PIXELS = 80  # 射线最大长度 (像素)
        self.UNKNOWN_VAL = 205       # 未知区域
        self.FREE_VAL = 254          # 自由区域
        self.OCCUPIED_VAL = 0        # 障碍物
        
        # 🚀 机器人安全参数 (米)
        # 请根据实际机器人调整：车半径 + 膨胀层半径
        self.ROBOT_SAFE_RADIUS_METERS = 0.5 
        
        # ⚠️ 注意：self.safe_radius_pixels 将在 load_static_map 中计算，
        # 因为此时 self.resolution 还是 0，不能在这里除！
        self.safe_radius_pixels = 0 
        
        # 生成策略参数
        self.MIN_DISTANCE_BETWEEN_POINTS = 1.0 # 米
        self.TOP_N_POINTS = 10                 # 最多保留点数
        self.SAMPLE_STEP = 2                   # 采样步长 (加速用)

        # 先加载地图，获取分辨率后，才能进行后续初始化
        if not self.load_static_map():
            sys.exit(1)
            
        rospy.loginfo(f"✅ 地图加载成功！安全半径设置为: {self.ROBOT_SAFE_RADIUS_METERS}m ({self.safe_radius_pixels} pixels)")
        rospy.loginfo("🚀 开始全局 NBV 计算...")
        self.run_nbv_task()

    def load_static_map(self):
        try:
            with open(self.yaml_path, 'r') as stream:
                meta = yaml.safe_load(stream)
            
            # 关键：确保 resolution 被正确读取且不为 0
            res_val = meta.get('resolution', 0.0)
            if res_val <= 0:
                rospy.logerr("❌ 地图分辨率 resolution 必须大于 0！检查 yaml 文件。")
                return False
                
            self.resolution = float(res_val)
            self.origin = list(meta['origin'])
            
            self.map_data = cv2.imread(self.pgm_path, cv2.IMREAD_GRAYSCALE)
            if self.map_data is None: 
                rospy.logerr("❌ 无法读取地图图片文件 (.pgm)。")
                return False
            
            self.height, self.width = self.map_data.shape
            
            # ✅ 在这里计算像素半径，此时 resolution 已有有效值
            self.safe_radius_pixels = int(math.ceil(self.ROBOT_SAFE_RADIUS_METERS / self.resolution))
            
            rospy.loginfo(f"🗺️ 地图尺寸: {self.width}x{self.height}, 分辨率: {self.resolution:.3f} m/pix")
            rospy.loginfo(f"🛡️ 安全检测圆半径: {self.safe_radius_pixels} 像素")
            return True
        except Exception as e:
            rospy.logerr(f"Error loading map: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_nbv_task(self):
        rospy.loginfo("🔄 第一步：全图扫描，寻找所有【安全】的候选点...")
        
        potential_candidates = []
        
        # 遍历全图自由区域
        for y in range(0, self.height, self.SAMPLE_STEP):
            for x in range(0, self.width, self.SAMPLE_STEP):
                val = self.map_data[y, x]
                if val == self.FREE_VAL or val == 255:
                    potential_candidates.append((x, y))
        
        rospy.loginfo(f"🔍 发现 {len(potential_candidates)} 个自由像素点。开始进行【安全足迹检查】...")
        
        safe_candidates = []
        count_safe = 0
        
        for (x, y) in potential_candidates:
            if self.is_safe_for_robot(x, y):
                safe_candidates.append((x, y))
                count_safe += 1
        
        if not safe_candidates:
            rospy.logwarn("⚠️ 没有找到任何满足安全半径要求的点！")
            rospy.logwarn(f"   当前设置安全半径: {self.ROBOT_SAFE_RADIUS_METERS}m ({self.safe_radius_pixels} pix)")
            rospy.logwarn("   建议：检查地图是否太窄，或减小 ROBOT_SAFE_RADIUS_METERS 参数。")
            self.save_waypoints([])
            return

        rospy.loginfo(f"✅ 通过安全检查的点：{count_safe} 个。开始计算射线得分...")
        
        all_scored_points = []
        
        # 简单的进度提示
        total = len(safe_candidates)
        step_report = max(1, total // 10)
        
        for i, (cx, cy) in enumerate(safe_candidates):
            score = self.cast_rays(cx, cy)
            if score > 0:
                all_scored_points.append((score, cx, cy))
            
            if i % step_report == 0:
                rospy.logdebug(f"   进度: {i}/{total}...")

        if not all_scored_points:
            rospy.logwarn("⚠️ 所有安全点的视线得分均为 0 (可能没有未知区域了)。")
            self.save_waypoints([])
            return

        rospy.loginfo(f"📊 排序前共有 {len(all_scored_points)} 个有效得分点。最高分：{all_scored_points[0][0]}")
        all_scored_points.sort(key=lambda x: x[0], reverse=True)

        final_waypoints = []
        for score, px, py in all_scored_points:
            wx = px * self.resolution + self.origin[0]
            wy = py * self.resolution + self.origin[1]
            
            is_duplicate = False
            for _, ex, ey in final_waypoints:
                dist = math.sqrt((wx - ex)**2 + (wy - ey)**2)
                if dist < self.MIN_DISTANCE_BETWEEN_POINTS:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_waypoints.append((score, wx, wy))
            
            if len(final_waypoints) >= self.TOP_N_POINTS:
                break
        
        rospy.loginfo(f"🎉 计算完成！最终选定 {len(final_waypoints)} 个最佳观测点 (NBV)。")
        self.save_waypoints(final_waypoints)

    def is_safe_for_robot(self, cx, cy):
        """
        🛡️ 核心安全检测：检查圆形区域内是否有障碍物
        """
        r = self.safe_radius_pixels
        x_int, y_int = int(cx), int(cy)
        
        # 1. 边界检查
        if x_int - r < 0 or x_int + r >= self.width or \
           y_int - r < 0 or y_int + r >= self.height:
            return False
        
        # 2. 圆形区域扫描
        # 优化：只遍历正方形内的点，判断是否在圆内
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx*dx + dy*dy > r*r:
                    continue 
                
                val = self.map_data[y_int + dy, x_int + dx]
                
                # 只要遇到一个障碍物 (0)，该点就不安全
                if val == self.OCCUPIED_VAL:
                    return False
        
        return True

    def cast_rays(self, cx, cy):
        """
        👁️ 视线模拟：射线穿过未知区域加分
        """
        score = 0
        angles = np.linspace(0, 2*math.pi, self.RAY_COUNT, endpoint=False)
        
        for angle in angles:
            dx = math.cos(angle)
            dy = math.sin(angle)
            
            for step in range(1, self.RAY_LENGTH_PIXELS):
                rx = int(cx + dx * step)
                ry = int(cy + dy * step)
                
                if not (0 <= rx < self.width and 0 <= ry < self.height):
                    break
                
                val = self.map_data[ry, rx]
                
                if val == self.OCCUPIED_VAL:
                    break # 撞墙停止
                
                if val == self.UNKNOWN_VAL:
                    score += 1
                    # 这里选择继续穿透 (累加面积) 还是停止 (只计看到)
                    # 通常累加更能反映覆盖范围，所以去掉了 break
                    # 如果你想恢复原逻辑 (看到第一个未知就停)，请取消下面这行的注释
                    # break 
        
        return score

    def save_waypoints(self, waypoints):
        filename = "nbv_waypoints.txt"
        script_dir = os.path.dirname(os.path.realpath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                f.write("# NBV Waypoints generated by nbv_search_node\n")
                f.write("# Format: x,y (world coordinates)\n")
                f.write(f"# Total Points: {len(waypoints)}\n")
                for i, (score, wx, wy) in enumerate(waypoints):
                    f.write(f"{wx:.4f},{wy:.4f}\n")
            
            rospy.loginfo(f"💾 waypoints 已保存至: {filepath}")
            
            for i, (score, wx, wy) in enumerate(waypoints):
                rospy.loginfo(f"   📍 Point {i+1}: ({wx:.2f}, {wy:.2f}) | Score: {score}")
            
            param_list = [f"{p[1]},{p[2]}" for p in waypoints]
            rospy.set_param('/nbv_waypoints', param_list)
            rospy.loginfo("📡 已发布参数 /nbv_waypoints")
            
        except Exception as e:
            rospy.logerr(f"❌ 保存文件失败: {e}")

if __name__ == '__main__':
    try:
        NBVSearchNode()
    except rospy.ROSInterruptException:
        rospy.loginfo("👋 节点已关闭。")
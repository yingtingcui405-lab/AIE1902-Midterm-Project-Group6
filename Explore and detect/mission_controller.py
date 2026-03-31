#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import subprocess
import os
import sys
import time
import signal
import actionlib
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_srvs.srv import Empty

# 消息类型兼容
try:
    from lab2_perception.msg import ObjectCoordinates
    MSG_TYPE = ObjectCoordinates
except ImportError:
    from geometry_msgs.msg import PointStamped as ObjectCoordinates
    MSG_TYPE = ObjectCoordinates

# 引入 dynamic_reconfigure 客户端
from dynamic_reconfigure.client import Client

COLOR_MAP = {
    "red":    {"lower": [0, 100, 100],   "upper": [10, 255, 255]},
    
    # 【修改前】[35, 80, 80] -> [85, 255, 255] (误识别砖墙)
    # 【修改后】[45, 180, 150] -> [85, 255, 255] (精准锁定塑料绿方块)
    "green":  {"lower": [45, 180, 150],  "upper": [85, 255, 255]}, 
    
    "blue":   {"lower": [100, 100, 100], "upper": [130, 255, 255]},
    "yellow": {"lower": [20, 100, 100],  "upper": [35, 255, 255]},
    "white":  {"lower": [0, 0, 200],     "upper": [180, 20, 255]},
    "black":  {"lower": [0, 0, 0],       "upper": [180, 255, 50]},
}

class MissionController:
    def __init__(self):
        rospy.init_node('mission_controller', log_level=rospy.INFO)
        
        # 获取当前脚本所在目录
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.waypoints_file = os.path.join(script_dir, "nbv_waypoints.txt")
        # 不再需要 set_color_script 路径，因为直接内部调用
        
        # 【关键】强制指定 perception_node 的全局名称，防止命名空间污染
        self.perception_node_name = "/perception_node" 
        self.perception_node_script = os.path.join(script_dir, "perception_node.py")
        
        # 配置项
        self.nav_goal_tolerance = 1.0
        self.observation_timeout = 30.0
        self.approach_timeout = 10.0
        
        self.target_color = None
        self.perception_proc = None
        self.target_found = False
        self.found_data = None
        
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.clear_costmap_srv = None

    def get_user_color(self):
        print("\n" + "="*40)
        print("🌈 可用颜色:", ', '.join(COLOR_MAP.keys()))
        try:
            color = input("🤖 请输入要寻找的目标颜色: ").strip().lower()
        except EOFError:
            color = ""
        if not color or color not in COLOR_MAP:
            rospy.logerr(f"❌ 无效颜色：{color}")
            return None
        return color

    def kill_perception_process(self):
        """彻底杀死感知节点"""
        if self.perception_proc is None:
            return True
            
        rospy.loginfo("🔴 正在关闭感知节点...")
        try:
            pgid = os.getpgid(self.perception_proc.pid)
            os.killpg(pgid, signal.SIGKILL)
            self.perception_proc.wait(timeout=2.0)
            self.perception_proc = None
            # 停止机器人
            for _ in range(5):
                self.vel_pub.publish(Twist())
                rospy.sleep(0.1)
            rospy.loginfo("   ✅ 感知节点已停止")
            return True
        except Exception as e:
            rospy.logerr(f"   ❌ 关闭进程出错：{e}")
            self.perception_proc = None
            return False

    def start_perception_process(self):
        """启动感知节点"""
        if self.perception_proc and self.perception_proc.poll() is None:
            rospy.logwarn("⚠️ 感知节点已在运行，重启中...")
            self.kill_perception_process()

        if not os.path.exists(self.perception_node_script):
            rospy.logerr(f"❌ 找不到脚本：{self.perception_node_script}")
            return False

        rospy.loginfo("🟢 正在启动 perception_node.py ...")
        cmd = ["python3", self.perception_node_script]
        
        # 【关键】设置环境变量，强制子进程在全局命名空间 '/' 下运行
        env = os.environ.copy()
        env['ROS_NAMESPACE'] = '/'
        
        try:
            self.perception_proc = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                cwd=os.path.dirname(self.perception_node_script),
                env=env  # <--- 应用全局命名空间环境
            )
        except Exception as e:
            rospy.logerr(f"❌ 启动失败：{e}")
            return False

        # 【关键】等待节点启动并注册服务
        # dynamic_reconfigure 服务通常需要 2-3 秒才能完全就绪
        rospy.loginfo("   ⏳ 等待节点初始化及服务注册 (3秒)...")
        time.sleep(3.0)
        
        if self.perception_proc.poll() is not None:
            stdout, _ = self.perception_proc.communicate()
            rospy.logerr("❌ 节点启动后立即崩溃!")
            if stdout:
                for line in stdout.decode('utf-8').splitlines():
                    rospy.logerr(f"   [ERR] {line}")
            return False
            
        rospy.loginfo("   ✅ 感知节点已就绪 (全局命名空间)")
        return True

    def apply_color_config(self, color_name):
        """
        【核心功能】直接使用 dynamic_reconfigure.client 修改参数
        包含重试机制，确保一定成功
        """
        if not color_name or color_name not in COLOR_MAP:
            rospy.logerr(f"❌ 无效颜色：{color_name}")
            return False

        target_lower = COLOR_MAP[color_name]["lower"]
        target_upper = COLOR_MAP[color_name]["upper"]
        
        config_params = {
            'lower_h': int(target_lower[0]),
            'lower_s': int(target_lower[1]),
            'lower_v': int(target_lower[2]),
            'upper_h': int(target_upper[0]),
            'upper_s': int(target_upper[1]),
            'upper_v': int(target_upper[2]),
            'display_scale': 0.5
        }

        rospy.loginfo(f"🎨 正在连接 {self.perception_node_name} 设置颜色：{color_name.upper()}")
        rospy.loginfo(f"   参数：H[{config_params['lower_h']}-{config_params['upper_h']}]")

        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 【关键】使用绝对路径 '/perception_node' 创建客户端
                # timeout=5 秒，如果连不上会抛出异常
                client = Client(self.perception_node_name, timeout=5.0)
                
                rospy.loginfo("   🔗 服务连接成功，正在发送配置...")
                
                # 发送配置
                result = client.update_configuration(config_params)
                
                # 验证结果 (可选：检查返回的配置是否与我们发送的一致)
                if (result['lower_h'] == config_params['lower_h'] and 
                    result['upper_h'] == config_params['upper_h']):
                    rospy.loginfo("✅ 颜色参数设置成功！(直接调用 DR 客户端)")
                    return True
                else:
                    rospy.logwarn("⚠️ 参数已更新，但返回值与预期略有不同，继续执行...")
                    return True
                    
            except Exception as e:
                retry_count += 1
                rospy.logwarn(f"⚠️ 连接/设置失败 (尝试 {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    rospy.loginfo("   🔄 1 秒后重试...")
                    time.sleep(1.0)
                else:
                    rospy.logerr("❌ 多次尝试后仍无法设置颜色参数！")
                    rospy.logerr("   请检查 perception_node 是否正常运行且服务已注册。")
                    return False
        
        return False

    def load_waypoints(self):
        waypoints = []
        if not os.path.exists(self.waypoints_file):
            rospy.logerr(f"❌ 未找到航点文件：{self.waypoints_file}")
            return None
        with open(self.waypoints_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                try:
                    x, y = map(float, line.split(','))
                    waypoints.append((x, y))
                except: pass
        return waypoints if waypoints else None

    def navigate_to_rough(self, x, y):
        """阶段 1：纯导航"""
        self.kill_perception_process()
        time.sleep(1.0)
        self._clear_costmaps()

        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        if not client.wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.logerr("❌ move_base 未连接")
            return False

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = 1.0

        rospy.loginfo(f"🚗 [导航模式] 前往 ({x:.2f}, {y:.2f}) ...")
        client.send_goal(goal)

        if client.wait_for_result(rospy.Duration(120.0)):
            state = client.get_state()
            if state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo("✅ 导航到达")
                return True
            else:
                rospy.logwarn(f"⚠️ 导航状态：{state}. 继续尝试感知...")
                return True 
        else:
            client.cancel_goal()
            rospy.logwarn("⏱️ 导航超时，强制进入感知阶段...")
            return True

    def _clear_costmaps(self):
        if self.clear_costmap_srv is None:
            try:
                rospy.wait_for_service('/move_base/clear_costmaps', timeout=2.0)
                self.clear_costmap_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
            except:
                return
        try:
            self.clear_costmap_srv()
        except:
            pass

    def detect_and_approach_phase(self):
        """阶段 2：启动视觉 -> 设置颜色 -> 搜索靠近"""
        # 1. 确保机器人静止
        for _ in range(5):
            self.vel_pub.publish(Twist())
            rospy.sleep(0.1)
        
        self._clear_costmaps()
        time.sleep(0.5)

        # 2. 启动感知节点 (强制全局命名空间)
        if not self.start_perception_process():
            rospy.logerr("❌ 无法启动感知节点，跳过此点")
            return False

        # 3. 【核心步骤】直接调用 apply_color_config (内部使用 DR Client)
        # 此时节点已启动并等待了 3 秒，且有重试机制
        if not self.apply_color_config(self.target_color):
            rospy.logerr("❌ 颜色配置失败！视觉将使用默认参数（可能找不到目标）")
            # 即使失败也继续尝试，万一默认值碰巧对了呢
        
        # 4. 监听检测结果
        self.target_found = False
        self.found_data = None
        
        def callback(msg):
            if not self.target_found:
                self.target_found = True
                if hasattr(msg, 'x'):
                    self.found_data = (msg.x, msg.y, getattr(msg, 'z', 0.0))
                elif hasattr(msg, 'point'):
                    self.found_data = (msg.point.x, msg.point.y, msg.point.z)
                rospy.loginfo("🚨 [视觉锁定] 目标发现! perception_node 正在自主靠近...")

        sub = rospy.Subscriber('detected_object', MSG_TYPE, callback)
        
        start_time = time.time()
        found_time = None
        
        rospy.loginfo(f"👁️  [搜索模式] 扫描中... (颜色：{self.target_color.upper()})")
        
        while not rospy.is_shutdown():
            elapsed = time.time() - start_time
            
            if self.target_found:
                if found_time is None:
                    found_time = time.time()
                    rospy.loginfo("⏳ 等待靠近过程完成...")
                
                if time.time() - found_time > self.approach_timeout:
                    rospy.loginfo("✅ 靠近动作完成")
                    break
            else:
                if elapsed > self.observation_timeout:
                    rospy.loginfo("⏱️ 扫描超时，未发现目标")
                    break
            
            rospy.sleep(0.1)
        
        sub.unregister()
        
        # 5. 阶段结束，关闭节点
        self.kill_perception_process()
        
        return self.target_found

    def run_mission(self):
        self.target_color = self.get_user_color()
        if not self.target_color: return

        waypoints = self.load_waypoints()
        if not waypoints: return

        self.kill_perception_process()

        rospy.loginfo("\n🚀 === 启动分阶段巡检任务 ===")
        rospy.loginfo("流程：导航 -> 启动视觉(全局命名空间) -> 内部DR改参 -> 搜索靠近\n")

        try:
            for i, (wx, wy) in enumerate(waypoints):
                rospy.loginfo(f"\n>>> 航点 {i+1}/{len(waypoints)}: ({wx:.2f}, {wy:.2f})")
                
                if not self.navigate_to_rough(wx, wy):
                    rospy.logwarn("⚠️ 导航严重失败，跳过")
                    continue
                
                found = self.detect_and_approach_phase()
                
                if found:
                    rospy.loginfo("\n" + "="*40)
                    rospy.loginfo("🎉 任务成功！目标已定位。")
                    rospy.loginfo(f"📍 坐标：{self.found_data}")
                    rospy.loginfo("="*40)
                    return 

                rospy.loginfo("➡️ 此点未找到，前往下一个...")
                time.sleep(1.0)

            rospy.logwarn("\n⚠️ 所有航点巡检完毕，未发现目标。")
            
        except KeyboardInterrupt:
            rospy.loginfo("👋 用户中断")
        finally:
            self.kill_perception_process()
            rospy.loginfo("🏁 清理完成")

if __name__ == '__main__':
    try:
        controller = MissionController()
        controller.run_mission()
    except Exception as e:
        rospy.logerr(f"💥 严重错误：{e}")
        import traceback
        traceback.print_exc()
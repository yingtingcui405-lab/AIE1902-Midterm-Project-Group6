#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import os
import subprocess
import sys
from nav_msgs.srv import GetMap, GetMapRequest
from nav_msgs.msg import OccupancyGrid
import time

class MapSaverTool:
    def __init__(self):
        rospy.init_node('auto_map_saver', anonymous=True)
        
        # --- 配置 ---
        self.output_folder = "map"  # 保存到的文件夹名
        self.file_prefix = "final_map" # 保存的文件名前缀 (最终生成 final_map.pgm 和 final_map.yaml)
        self.wait_time = 2.0 # 等待地图数据稳定的时间(秒)
        
        rospy.loginfo("🗺️ 自动地图保存工具已启动...")
        rospy.loginfo(f"📂 目标文件夹: ./{self.output_folder}/")
        rospy.loginfo(f"📄 目标文件名: {self.file_prefix}.pgm / {self.file_prefix}.yaml")

    def wait_for_map(self):
        """等待 /map 话题有数据"""
        rospy.loginfo("⏳ 正在等待 /map 话题数据...")
        try:
            # 等待消息，超时设为 60 秒
            msg = rospy.wait_for_message("/map", OccupancyGrid, timeout=60.0)
            rospy.loginfo("✅ 接收到地图数据！")
            rospy.loginfo(f"   地图尺寸: {msg.info.width} x {msg.info.height}")
            rospy.loginfo(f"   分辨率: {msg.info.resolution} m/pix")
            return True
        except rospy.ROSException as e:
            rospy.logerr(f"❌ 超时未收到地图数据！请确认 SLAM 或 map_server 正在运行并发布 /map 话题。")
            return False

    def save_map_via_service(self):
        """
        尝试调用 map_server 的 dynamic_map 服务来保存，
        或者直接调用命令行 map_saver (更通用)。
        这里我们使用命令行方式，因为它能同时生成 pgm 和 yaml，且最稳定。
        """
        
        # 1. 确保目标文件夹存在
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            rospy.loginfo(f"📁 已创建文件夹: {self.output_folder}")
        else:
            rospy.loginfo(f"📁 文件夹已存在: {self.output_folder}")

        # 2. 构建命令
        # rosrun map_server map_saver -f <完整路径/前缀>
        # 注意：map_saver 的 -f 参数不需要加后缀，它会自动加 .pgm 和 .yaml
        abs_path = os.path.abspath(os.path.join(self.output_folder, self.file_prefix))
        cmd = ["rosrun", "map_server", "map_saver", "-f", abs_path]
        
        rospy.loginfo(f"💾 正在执行保存命令: {' '.join(cmd)}")
        rospy.loginfo("⚠️ 请确保此时机器人静止，且地图不再剧烈变化。")
        
        # 稍微等待一下，让用户有机会确认机器人停稳，或者让数据缓冲完成
        rospy.sleep(self.wait_time)

        try:
            # 调用系统命令
            # check_call 会在命令失败时抛出异常
            subprocess.check_call(cmd)
            
            # 3. 验证文件是否生成
            pgm_file = abs_path + ".pgm"
            yaml_file = abs_path + ".yaml"
            
            if os.path.exists(pgm_file) and os.path.exists(yaml_file):
                rospy.loginfo("🎉 成功！地图已保存。")
                rospy.loginfo(f"   ✅ {pgm_file}")
                rospy.loginfo(f"   ✅ {yaml_file}")
                rospy.loginfo("------------------------------------------------")
                rospy.loginfo("下一步操作建议：")
                rospy.loginfo("1. 关闭 explore_lite 和 SLAM (可选)")
                rospy.loginfo(f"2. 运行 NBV 脚本，它将自动读取 ./map/{self.file_prefix}.yaml")
                rospy.loginfo("------------------------------------------------")
                return True
            else:
                rospy.logerr("❌ 命令执行完毕，但未找到生成的文件。")
                return False
                
        except subprocess.CalledProcessError as e:
            rospy.logerr(f"❌ 保存命令执行失败！错误代码: {e}")
            rospy.logerr("请确认是否安装了 map_server 包 (sudo apt-get install ros-<your-distro>-map-server)")
            return False
        except Exception as e:
            rospy.logerr(f"❌ 发生未知错误: {str(e)}")
            return False

    def run(self):
        # 主流程
        if self.wait_for_map():
            success = self.save_map_via_service()
            if success:
                rospy.loginfo("✅ 任务完成。节点将在 3 秒后退出。")
                rospy.sleep(3)
                rospy.signal_shutdown("Map saved successfully")
            else:
                rospy.logerr("❌ 保存失败。节点退出。")
                sys.exit(1)
        else:
            sys.exit(1)

if __name__ == '__main__':
    try:
        saver = MapSaverTool()
        saver.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
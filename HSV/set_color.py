#!/usr/bin/env python3
import rospy
import sys
from dynamic_reconfigure.client import Client

# 颜色映射表：颜色名称 -> HSV 范围 (注意：H范围0-180, S/V范围0-255)
COLOR_MAP = {
    "red":    {"lower": [0, 80, 79],   "upper": [50, 255, 255]},
    "green":  {"lower": [35, 80, 80],  "upper": [85, 255, 255]},
    "blue":   {"lower": [114, 80, 80], "upper": [179, 255, 255]},
    "yellow": {"lower": [11, 240, 240],  "upper": [33, 255, 255]},
    "pink":   {"lower": [147, 80, 80], "upper": [179, 255, 255]},
}

def set_color_via_reconfigure(color_name):
    color_name = color_name.lower()
    
    if color_name not in COLOR_MAP:
        print(f"❌ Unsupported color: {color_name}")
        print(f"✅ Supported colors: {', '.join(COLOR_MAP.keys())}")
        return

    # 1. 初始化节点 (必须初始化才能使用 client)
    rospy.init_node('color_commander', anonymous=True)
    
    # 2. 创建客户端，连接到 perception_node 的动态配置服务
    # 服务名通常是：节点名 + "/parameter_updates" 或者直接在 node 路径下
    # 在你的代码中，Server 是在 'perception_node' 里初始化的
    try:
        rospy.loginfo(f"Connecting to the dynamic configuration service of the perception_node...")
        client = Client("perception_node", timeout=10)
        rospy.loginfo("✅Connection successful! ")
    except Exception as e:
        rospy.logerr(f"❌ Update failed: {e}")
        print("Please make sure that the 'perception_node' is already running!")
        return

    # 3. 准备配置数据
    params = COLOR_MAP[color_name]
    new_config = {
        'lower_h': params['lower'][0],
        'lower_s': params['lower'][1],
        'lower_v': params['lower'][2],
        'upper_h': params['upper'][0],
        'upper_s': params['upper'][1],
        'upper_v': params['upper'][2]
    }

    # 4. 发送更新请求
    # 这行代码会触发 perception_node 中的 reconfig_cb 函数
    rospy.loginfo(f"🚀Setting the color to [{color_name.upper()}]: {new_config}")
    try:
        result = client.update_configuration(new_config)
        rospy.loginfo("✅ Parameter update successful! The robot now starts to track this color")
        # 打印返回的实际值（有时边界会被截断）
        # print(f"服务器确认的值: H[{result['lower_h']}-{result['upper_h']}]")
    except Exception as e:
        rospy.logerr(f"❌ Update failed: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        user_input = input("Please enter the color you want to track. (red/green/blue/yellow/pink): ")
    else:
        user_input = sys.argv[1]
    
    set_color_via_reconfigure(user_input)

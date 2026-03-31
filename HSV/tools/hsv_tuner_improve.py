#!/usr/bin/env python3
import cv2
import numpy as np
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

def nothing(x):
    pass


class PerceptionNo:
    def __init__(self):
        rospy.init_node('perception_node')
        
        self.bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.rgb_callback)
    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            return cv_image
        except Exception as e:
            rospy.logerr(f"RGB error: {e}")
def main(img):
    cv2.namedWindow('HSV Tuner')

    # HSV 滑条
    cv2.createTrackbar('H Min', 'HSV Tuner', 35, 179, nothing)
    cv2.createTrackbar('H Max', 'HSV Tuner', 85, 179, nothing)

    cv2.createTrackbar('S Min', 'HSV Tuner', 80, 255, nothing)
    cv2.createTrackbar('S Max', 'HSV Tuner', 255, 255, nothing)

    cv2.createTrackbar('V Min', 'HSV Tuner', 80, 255, nothing)
    cv2.createTrackbar('V Max', 'HSV Tuner', 255, 255, nothing)

    cv2.createTrackbar('Min Area', 'HSV Tuner', 300, 50000, nothing)

    while True:
        # 读取参数
        h_min = cv2.getTrackbarPos('H Min', 'HSV Tuner')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Tuner')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Tuner')
        h_max = cv2.getTrackbarPos('H Max', 'HSV Tuner')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Tuner')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Tuner')
        min_area = cv2.getTrackbarPos('Min Area', 'HSV Tuner')

        results = []
        # === 1️⃣ 图片压缩 ===
        #width x 0.3, height x 0.3
        img = cv2.resize(img, None, fx=0.3, fy=0.3)
        #BGR color space format converts to HSV color space format to do the color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        #Generate a mask, if the hsv values fall in the lower and upper range, it will be white, otherwise black
        mask = cv2.inRange(hsv, lower, upper)
	    #With the help of mask, get the contour sets of the image.
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
	    #copy the img
        result = img.copy()

        # === 2️⃣ 变色目标框 ===
        #Travel through every contour
        for i, cnt in enumerate(contours):
            #If it is not too small, get the (x,y) left up vertice coordinate, 			w=width,h=height
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)

                # 根据目标编号生成颜色 randomly
                color = cv2.cvtColor(
                    np.uint8([[[i * 40 % 180, 255, 255]]]),
                    cv2.COLOR_HSV2BGR
                    )[0][0].tolist()
		    #Generate contour lines
                cv2.rectangle(
                    result,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                    )

            # 图片名
            name = os.path.basename(img_path)
            cv2.putText(
                result,
                name,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            triple = np.hstack((img, mask_bgr, result))
            results.append(triple)

        if results:
            stacked_all = np.vstack(results)
            cv2.imshow('HSV Tuner', stacked_all)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    node = PerceptionNode()
    img=node.run()
    main(img)
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from ultralytics import YOLO
import torch
import subprocess
import pyrealsense2 as rs

# 관절 연결선 정의
skeleton_connections = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIDENCE_THRESHOLD = 0.75
CENTER_X = 640 // 2
GO2_BINARY = "/home/unitree/unitree_sdk2-main/build/bin/go2_follow_human"

class HumanTracker(Node):
    def __init__(self):
        super().__init__('human_tracker')
        self.publisher_ = self.create_publisher(Point, 'human_position', 10)
        self.model = YOLO("yolov8m-pose.pt").to(DEVICE)

        # RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

    def calculate_speed(self, cx, dist_cm):
        vx, vyaw = 0.0, 0.0
        if abs(dist_cm - 200) > 15:
            vx = 0.015 * (dist_cm - 200)
        error_x = cx - CENTER_X
        if abs(error_x) > 30:
            vyaw = -0.003 * error_x
        return vx, 0.0, vyaw

    def run(self):
        try:
            while rclpy.ok():
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_img = np.asanyarray(depth_frame.get_data())
                color_img = np.asanyarray(color_frame.get_data())

                results = self.model(color_img)
                boxes_all = results[0].boxes
                if boxes_all is None or boxes_all.shape[0] == 0:
                    cv2.imshow("Color", color_img)
                    cv2.waitKey(1)
                    continue

                confs = boxes_all.conf.cpu().numpy()
                boxes_xyxy = boxes_all.xyxy.cpu().numpy()
                keypoints_all = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []

                indices = np.where(confs > CONFIDENCE_THRESHOLD)[0]
                boxes = boxes_xyxy[indices]
                keypoints_all = keypoints_all[indices]

                for box, keypoints in zip(boxes, keypoints_all):
                    if len(keypoints) < 11:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    cx = int((keypoints[5][0] + keypoints[6][0]) / 2)
                    cy = int((keypoints[5][1] + keypoints[6][1]) / 2)
                    depth = depth_frame.get_distance(cx, cy)
                    distance_cm = depth * 100

                    # ROS & 이동 명령
                    vx, vy, vyaw = self.calculate_speed(cx, distance_cm)
                    msg = Point()
                    msg.x, msg.y, msg.z = float(vx), float(vy), float(vyaw)
                    self.publisher_.publish(msg)
                    subprocess.run(["sudo", GO2_BINARY, str(vx), str(vy), str(vyaw)])

                    # 시각화
                    cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(color_img, f"{distance_cm:.1f}cm", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(color_img, f"vx:{vx:.2f} vyaw:{vyaw:.2f}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    for i, (x, y) in enumerate(keypoints):
                        if x > 0 and y > 0:
                            cv2.circle(color_img, (int(x), int(y)), 4, (0, 255, 255), -1)

                    for j1, j2 in skeleton_connections:
                        if j1 < len(keypoints) and j2 < len(keypoints):
                            x1, y1 = keypoints[j1]
                            x2, y2 = keypoints[j2]
                            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                                cv2.line(color_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

                cv2.imshow("Color", color_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = HumanTracker()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

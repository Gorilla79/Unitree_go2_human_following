import cv2
import numpy as np
import subprocess
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import pyrealsense2 as rs
from ultralytics import YOLO

# skeleton 연결 정의
skeleton_connections = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

class PersonTracker(Node):
    def __init__(self):
        super().__init__('human_tracker')
        self.publisher_ = self.create_publisher(Point, 'human_position', 10)

        # RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # YOLO pose 모델
        self.model = YOLO("yolov8m-pose.pt").to("cuda")

    def run(self):
        try:
            while rclpy.ok():
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                height, width = color_image.shape[:2]
                cx_center = width // 2

                results = self.model(color_image)
                boxes_data = results[0].boxes
                confidences = boxes_data.conf.cpu().numpy()
                boxes_xyxy = boxes_data.xyxy.cpu().numpy()
                filtered_indices = np.where(confidences > 0.75)[0]
                boxes = boxes_xyxy[filtered_indices]
                keypoints_all = results[0].keypoints.xy.cpu().numpy()[filtered_indices]
                
                if len(boxes) > 0 and len(keypoints_all) > 0:
                    for box, keypoints in zip(boxes, keypoints_all):
                        x1, y1, x2, y2 = map(int, box)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        depth = depth_frame.get_distance(cx, cy)
                        if depth == 0:
                            continue  # 👈 필수 예외처리
                            
                        distance_cm = depth * 100
                        relative_x = cx - cx_center
                        vx = (distance_cm - 200.0) * 0.005
                        vy = 0.0
                        vyaw = -relative_x * 0.002
                        vx = np.clip(vx, -0.4, 0.4)
                        vyaw = np.clip(vyaw, -0.6, 0.6)
                        
                        # ROS2 Topic Publish
                        msg = Point()
                        msg.x = float(relative_x)
                        msg.y = float(distance_cm)
                        msg.z = 0.0
                        self.publisher_.publish(msg)
        
                        # Go2 제어 실행
                        subprocess.run(["sudo", "/home/unitree/unitree_sdk2-main/build/bin/go2_follow_human", str(vx), str(vy), str(vyaw)])

                        # 시각화
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(color_image, f"ID:0, Human", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                        cv2.putText(color_image, f"X:{relative_x}, Depth:{distance_cm:.1f}cm",
                                   (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                        # skeleton 그리기
                        for i, kp in enumerate(keypoints):
                            x, y = map(int, kp)
                            if x > 0 and y > 0:
                                cv2.circle(color_image, (x, y), 3, (0, 255, 255), -1)
                        for start, end in skeleton_connections:
                            if start < len(keypoints) and end < len(keypoints):
                                x1s, y1s = keypoints[start]
                                x2s, y2s = keypoints[end]
                                if x1s > 0 and y1s > 0 and x2s > 0 and y2s > 0:
                                    cv2.line(color_image, (int(x1s), int(y1s)), (int(x2s), int(y2s)),
                                            (0, 255, 0), 2)

                cv2.imshow("Go2 Person Tracking", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = PersonTracker()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

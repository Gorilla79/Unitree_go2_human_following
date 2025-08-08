import cv2
import numpy as np
import subprocess
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import pyrealsense2 as rs
from ultralytics import YOLO
from collections import deque

# 스켈레톤 연결 (pose 시각화)
skeleton_connections = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

class DangerousHumanTracker(Node):
    def __init__(self):
        super().__init__('human_coordinate')
        self.publisher_ = self.create_publisher(Point, 'human_coordinate', 10)

        # RealSense 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # 모델 로드
        self.human_model = YOLO("yolov8m-pose.pt").to("cuda")
        self.weapon_model = YOLO("/home/unitree/dataset/best_bat.pt").to("cuda")  # 무기 감지 모델

        self.track_history = deque(maxlen=5)
        self.target_dangerous_id = None
        self.confidence_threshold = 0.7
        self.weapon_conf_threshold = 0.6
        self.iou_threshold = 0.2

    def predict_direction(self):
        if len(self.track_history) < 2:
            return 0
        dx = self.track_history[-1][0] - self.track_history[0][0]
        return dx

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
                height, width = color_image.shape[:2]
                cx_center = width // 2

                # 사람 탐지
                human_results = self.human_model(color_image)
                weapon_results = self.weapon_model(color_image)

                # 무기 박스 추출 (conf 기준)
                weapon_boxes = []
                if weapon_results[0].boxes:
                    for i, box in enumerate(weapon_results[0].boxes.xyxy.cpu().numpy()):
                        conf = weapon_results[0].boxes.conf[i].cpu().numpy()
                        if conf < self.weapon_conf_threshold:
                            continue
                        wx1, wy1, wx2, wy2 = map(int, box)
                        weapon_boxes.append([wx1, wy1, wx2, wy2])

                dangerous_targets = []
                humans = []

                # keypoints 추출
                if hasattr(human_results[0], 'keypoints') and human_results[0].keypoints is not None:
                    keypoints_all = human_results[0].keypoints.xy.cpu().numpy()
                else:
                    keypoints_all = []

                # 사람 탐지 결과 처리
                if human_results[0].boxes:
                    for i, box in enumerate(human_results[0].boxes.xyxy.cpu().numpy()):
                        conf = human_results[0].boxes.conf[i].cpu().numpy()
                        if conf < self.confidence_threshold:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        depth = depth_frame.get_distance(cx, cy)
                        distance_cm = depth * 100
                        human_id = i

                        # 위험인물 판별(손 좌표와 무기 중심 간의 거리)
                        label = "Human"
                        color = (0, 255, 0)
                        is_dangerous = False

                        # 1. 손-무기 거리 체크
                        if len(keypoints_all) > i:
                            left_hand = keypoints_all[i][9]
                            right_hand = keypoints_all[i][10]
                            for wbox in weapon_boxes:
                                wx1, wy1, wx2, wy2 = wbox
                                weapon_cx = (wx1 + wx2) // 2
                                weapon_cy = (wy1 + wy2) // 2
                                if np.linalg.norm(left_hand - [weapon_cx, weapon_cy]) < 60 or \
                                   np.linalg.norm(right_hand - [weapon_cx, weapon_cy]) < 60:
                                    is_dangerous = True
                                    label = "Dangerous Human"
                                    color = (0, 0, 255)
                                    self.target_dangerous_id = human_id
                                    break

                        # 2. 박스 IoU 판별(추가로 오탐 잡기)
                        if not is_dangerous:
                            for wbox in weapon_boxes:
                                wx1, wy1, wx2, wy2 = wbox
                                iou_x1 = max(x1, wx1)
                                iou_y1 = max(y1, wy1)
                                iou_x2 = min(x2, wx2)
                                iou_y2 = min(y2, wy2)
                                inter_area = max(0, iou_x2 - iou_x1) * max(0, iou_y2 - iou_y1)
                                box_area = (x2-x1)*(y2-y1)
                                weapon_area = (wx2-wx1)*(wy2-wy1)
                                iou = inter_area / float(box_area + weapon_area - inter_area + 1e-6)
                                if iou > self.iou_threshold:
                                    is_dangerous = True
                                    label = "Dangerous Human"
                                    color = (0, 0, 255)
                                    self.target_dangerous_id = human_id
                                    break

                        if is_dangerous:
                            dangerous_targets.append((cx, cy, distance_cm, human_id))
                        else:
                            humans.append((cx, cy, distance_cm, human_id))

                        # 시각화
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(color_image, f"{label}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.circle(color_image, (cx, cy), 5, (0, 255, 255), -1)
                        cv2.putText(color_image, f"X:{cx - cx_center}, D:{distance_cm:.1f}cm",
                                    (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                        # 스켈레톤 표시 (옵션)
                        if len(keypoints_all) > i:
                            for s, e in skeleton_connections:
                                if s < len(keypoints_all[i]) and e < len(keypoints_all[i]):
                                    x_s, y_s = keypoints_all[i][s]
                                    x_e, y_e = keypoints_all[i][e]
                                    if x_s > 0 and y_s > 0 and x_e > 0 and y_e > 0:
                                        cv2.line(color_image, (int(x_s), int(y_s)), (int(x_e), int(y_e)), (255, 255, 0), 2)
                            for j, (x_kp, y_kp) in enumerate(keypoints_all[i]):
                                if x_kp > 0 and y_kp > 0:
                                    cv2.circle(color_image, (int(x_kp), int(y_kp)), 3, (0, 255, 255), -1)

                # 무기만 표시
                for wbox in weapon_boxes:
                    wx1, wy1, wx2, wy2 = wbox
                    cv2.rectangle(color_image, (wx1, wy1), (wx2, wy2), (0, 255, 255), 2)
                    cv2.putText(color_image, "Weapon", (wx1, wy1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # 위험인물 추적 및 타겟팅
                target = None
                if self.target_dangerous_id is not None:
                    for cx, cy, tdist, hid in dangerous_targets:
                        if hid == self.target_dangerous_id:
                            target = (cx, cy, tdist)
                            break
                elif dangerous_targets:
                    cx, cy, tdist, hid = dangerous_targets[0]
                    target = (cx, cy, tdist)
                    self.target_dangerous_id = hid

                if target:
                    tx, ty, tdist = target
                    self.track_history.append((tx, ty))
                    dx = self.predict_direction()

                    relative_x = tx - cx_center
                    vx = (tdist - 300.0) * 0.01  # 3m 유지
                    vyaw = -relative_x * 0.004
                    vx += np.sign(dx) * 0.1

                    vx = np.clip(vx, -0.8, 0.8)
                    vyaw = np.clip(vyaw, -1.2, 1.2)

                    # ROS2 Publish
                    msg = Point()
                    msg.x = float(relative_x)
                    msg.y = float(tdist)
                    msg.z = 0.0
                    self.publisher_.publish(msg)

                    # 로봇 제어
                    subprocess.run(["sudo", "/home/unitree/unitree_sdk2-main/build/bin/go2_follow_human",
                                    str(vx), "0.0", str(vyaw)])

                cv2.imshow("Dangerous Human Tracking", color_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.target_dangerous_id = None
                    self.track_history.clear()

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = DangerousHumanTracker()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

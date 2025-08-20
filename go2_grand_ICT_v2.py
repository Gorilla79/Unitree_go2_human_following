
import cv2
import json
import time
import math
import numpy as np
import subprocess
from collections import defaultdict, deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

import pyrealsense2 as rs
from ultralytics import YOLO

# COCO pose index (YOLOv8 pose): 9=left_wrist, 10=right_wrist
WRIST_IDX = [9, 10]

# 스켈레톤 (시각화 용)
SKELETON = [
    (5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),
    (11,12),(11,13),(13,15),(12,14),(14,16)
]

BIN = "/home/unitree/unitree_sdk2-main/build/bin/go2_grand_ICT_v2"

def now_ns():
    return int(time.time() * 1e9)

class GrandICT(Node):
    def __init__(self):
        super().__init__('grand_ict')

        # Pub/Sub
        self.pub_coord = self.create_publisher(String, 'coordinate', 10)
        self.pub_img   = self.create_publisher(CompressedImage, 'image/compressed', 10)
        self.pub_meta  = self.create_publisher(String, 'image/meta', 10)
        self.sub_cmd   = self.create_subscription(String, 'grand_cmd', self.on_cmd, 10)

        # RealSense
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)

        # Camera intrinsics (fetch once after start)
        prof = self.pipe.get_active_profile()
        cstream = prof.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = cstream.get_intrinsics()  # .fx, .fy, .ppx, .ppy

        # Models
        self.pose_model   = YOLO("yolov8m-pose.pt").to("cuda")
        self.weapon_model = YOLO("/home/unitree/dataset/best_bat.pt").to("cuda")

        # State
        self.cmd_mode = 0          # 0 cancel, 1 follow@3m, 2 approach@1m+hello
        self.target_id = None      # 위험 인물 타겟 id
        self.danger_count = defaultdict(int)
        self.history = dict()      # id -> deque of (u,v,Zmm) for 최근 위치
        self.last_target_pose = None  # (u,v,Zmm) for approach 사용

        # Params
        self.human_conf_th = 0.7
        self.weapon_conf_th = 0.6
        self.iou_th = 0.2
        self.danger_frames = 20
        self.near_px_min = 60
        self.near_px_max = 150

        # 속도 스케일(“빠른” / “느린”)
        # 빠른(전/대각 전후진): go2_follow_human_speed 급
        self.k_v_fast   = 0.012    # 3m 유지 제어계수
        self.k_yaw_fast = 0.006
        # 느린(작은 전후/좌우, 제자리 회전)
        self.k_v_slow   = 0.006
        self.k_yaw_slow = 0.003

        self.max_v_fast = 0.8
        self.max_vyaw_fast = 1.2
        self.max_v_slow = 0.3
        self.max_vyaw_slow = 0.6

        self.mm_target_follow = 3000.0   # 3m
        self.mm_target_approach = 1000.0 # 1m

        # 영상 창
        cv2.namedWindow("Grand-ICT", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Grand-ICT", 1280, 720)
        
        # 위험 인물 판별 결과 전달
        self._prompt_sent = False
        # 위험 인물 최초 확정 알림(Host 프롬프트 트리거) 플래그
        self._danger_announced = False
        
        # 타깃 유실 감속/정지 로직용 카운터(루프마다 0으로 만들지 말고 멤버로 유지)
        self.lost_frames = 0
        self.lost_stop_delay = 3

        # Timer로 루프
        self.timer = self.create_timer(0.005, self.loop)  # as-fast-as-possible
        
        self._last_cmd_ts = 0.0
        self._cmd_period  = 1.0 / 15.0 
        
        self._call_cpp("stop")

    def on_cmd(self, msg: String):
        cmd = msg.data.strip()
        if cmd == "0":
            # Cancel
            self.cmd_mode = 0
            self.target_id = None
            self.danger_count.clear()
            self.history.clear()
            self._danger_announced = False
            self.lost_frames = 0
            self.last_target_pose = None
            self._call_cpp(["stop"])
            self.get_logger().info("[CMD] Cancel: stop & reset")
            
            # ACK 메타 전송
            ack = String(); ack.data = json.dumps({"ack": cmd, "ts": time.time()})
            self.pub_meta.publish(ack)
            
        elif cmd == "1":
            self.cmd_mode = 1
            self.get_logger().info("[CMD] Follow@3m")
            
            # ACK 메타 전송
            ack = String(); ack.data = json.dumps({"ack": cmd, "ts": time.time()})
            self.pub_meta.publish(ack)
            
        elif cmd == "2":
            self.cmd_mode = 2
            self.get_logger().info("[CMD] Approach@1m + Hello")
            
            # ACK 메타 전송
            ack = String(); ack.data = json.dumps({"ack": cmd, "ts": time.time()})
            self.pub_meta.publish(ack)
            
        else:
            self.get_logger().warn(f"[CMD] Unknown: {cmd}")

    def uvZ_to_xy_mm(self, u, v, Zmm):
        # RealSense intrinsics: X = (u-ppx)/fx * Z
        Xmm = ( (u - self.intr.ppx) / self.intr.fx ) * Zmm
        Ymm = Zmm
        return Xmm, Ymm

    def _call_cpp(self, *args, check=False):
        """
            _call_cpp("move", vx, 0.0, vyaw)
            _call_cpp("stop")
            _call_cpp("hello")
            _call_cpp("damp")
        """
        # 숫자는 안전하게 문자열로 변환 + NaN 보호
        argv = []
        for a in args:
            if isinstance(a, (float, int, np.floating, np.integer)):
                if not np.isfinite(a):
                    a = 0.0
                argv.append(f"{float(a):.6f}")
            else:
                argv.append(str(a))

        cmd = ["sudo", BIN] + argv
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=check)
        except Exception as e:
            self.get_logger().error(f"CPP call failed: {e}")

    def _move_fast(self, vx, vy, vyaw):
        vx = float(np.clip(vx, -self.max_v_fast, self.max_v_fast))
        vyaw = float(np.clip(vyaw, -self.max_vyaw_fast, self.max_vyaw_fast))
        self._call_cpp(["move", vx, 0.0, vyaw])

    def _move_slow(self, vx, vy, vyaw):
        vx = float(np.clip(vx, -self.max_v_slow, self.max_v_slow))
        vyaw = float(np.clip(vyaw, -self.max_vyaw_slow, self.max_vyaw_slow))
        self._call_cpp(["move", vx, 0.0, vyaw])

    def loop(self):
        frames = self.pipe.wait_for_frames()
        aligned = self.align.process(frames)
        d = aligned.get_depth_frame()
        c = aligned.get_color_frame()
        if not d or not c: 
            return

        color = np.asanyarray(c.get_data())
        depth = np.asanyarray(d.get_data())
        H, W = color.shape[:2]
        u0 = int(self.intr.ppx)

        # 추론
        pose_res = self.pose_model(color, imgsz=640, verbose=False)
        weapon_res = self.weapon_model(color, imgsz=640, verbose=False)

        # 무기 박스 수집
        weapons = []
        if not weapon_res or weapon_res[0].boxes is None:
            pass
        else:
            boxes = weapon_res[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            for b, cf in zip(xyxy, conf):
                if cf >= self.weapon_conf_th:
                    x1,y1,x2,y2 = map(int, b)
                    cx = (x1+x2)//2; cy=(y1+y2)//2
                    weapons.append(((x1,y1,x2,y2),(cx,cy), float(cf)))

        # 사람 박스 + 포즈
        persons = []
        if not pose_res or pose_res[0].boxes is None:
            pass
        else:
            b = pose_res[0].boxes
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()
            kps  = pose_res[0].keypoints  # .xy (B,P,2)
            kp_xy = kps.xy.cpu().numpy() if kps is not None else None

            idx = 0
            for bb, cf in zip(xyxy, conf):
                if cf < self.human_conf_th: 
                    idx += 1
                    continue
                x1,y1,x2,y2 = map(int, bb)
                cx = (x1+x2)//2; cy=(y1+y2)//2
                Zm = d.get_distance(cx, cy)  # m
                if Zm <= 0: 
                    idx += 1
                    continue
                Zmm = Zm * 1000.0
                if Zmm < 200.0:
                    idx += 1
                    continue
                # 좌표(mm)
                Xmm, Ymm = self.uvZ_to_xy_mm(cx, cy, Zmm)

                # 포즈 키포인트
                kp = None
                if kp_xy is not None and idx < kp_xy.shape[0]:
                    kp = kp_xy[idx]  # (17,2) or similar
                persons.append({
                    "id": idx, "box": (x1,y1,x2,y2),
                    "uvZ": (cx, cy, Zmm),
                    "XYmm": (Xmm, Ymm),
                    "kp": kp
                })
                idx += 1

        # 손목 근처 무기 매칭 → 위험인물 카운트
        # 여러 사람일 때, 각 무기 중심과 가장 가까운 손목을 찾고, 거리(px) 60~150 사이면 카운트
        matched_ids = set()
        for wbox, (wx,wy), wconf in weapons:
            best_id = None; best_dist = 1e9
            for p in persons:
                kp = p["kp"]
                if kp is None: 
                    continue
                for wi in WRIST_IDX:
                    if wi < kp.shape[0]:
                        ux, vy = kp[wi]
                        if ux<=0 or vy<=0: 
                            continue
                        dist = math.hypot(wx - ux, wy - vy)
                        if self.near_px_min <= dist <= self.near_px_max:
                            if dist < best_dist:
                                best_dist = dist
                                best_id = p["id"]
            if best_id is not None:
                self.danger_count[best_id] += 1
                matched_ids.add(best_id)

        # 위험인물 타겟 확정(10프레임 연속 충족 시)
        if self.target_id is None:
            for pid, cnt in self.danger_count.items():
                if cnt >= self.danger_frames:
                    self.target_id = pid
                    self._danger_announced = False  # 새 타깃이 생겼으니 다시 한 번만 알리도록 초기화
                    break
        else:
            # 이미 타겟이면 유지, 단 유실/사라짐 처리
            pass

        # 좌표 퍼블리시 문자열 구성
        coord_msg = String()
        if len(persons) == 0:
            coord_msg.data = "none"
        else:
            if self.target_id is not None:
                # 위험 인물 우선
                sel = [p for p in persons if p["id"] == self.target_id]
                if sel:
                    Xmm, Ymm = sel[0]["XYmm"]
                    coord_msg.data = f"danger_id:{self.target_id}, X:{int(Xmm)}, Y:{int(Ymm)}"
                else:
                    coord_msg.data = "none"
            else:
                parts = []
                for p in persons:
                    Xmm, Ymm = p["XYmm"]
                    parts.append(f"id:{p['id']}, X:{int(Xmm)}, Y:{int(Ymm)}")
                coord_msg.data = ", ".join(parts)
        self.pub_coord.publish(coord_msg)

        # 시각화 + 위험 라벨
        vis = color.copy()
        for p in persons:
            x1,y1,x2,y2 = p["box"]
            is_danger = (self.target_id == p["id"])
            col = (0,0,255) if is_danger else (0,255,0)
            lb  = "dangerous human" if is_danger else f"human_id:{p['id']}"
            cv2.rectangle(vis,(x1,y1),(x2,y2),col,2)
            cv2.putText(vis, lb, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cx,cy,Zmm = p["uvZ"]
            cv2.circle(vis,(cx,cy),4,(0,255,255),-1)
            Xmm,Ymm = p["XYmm"]
            cv2.putText(vis, f"X:{int(Xmm)}mm,Y:{int(Ymm)}mm", (x1, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

            # 포즈 간단 표시
            if p["kp"] is not None:
                for (a,b) in SKELETON:
                    if a < p["kp"].shape[0] and b < p["kp"].shape[0]:
                        ua,va = p["kp"][a]; ub,vb = p["kp"][b]
                        if ua>0 and va>0 and ub>0 and vb>0:
                            cv2.line(vis,(int(ua),int(va)),(int(ub),int(vb)), (0,255,0), 2)

        # 무기 박스
        for (x1,y1,x2,y2),(wx,wy),cf in weapons:
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.putText(vis, f"weapon:{cf:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
        img_msg = CompressedImage()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.format = "jpeg"
        img_msg.data = cv2.imencode(".jpg", vis)[1].tobytes()
        self.pub_img.publish(img_msg)

        # 이미지 전송(캡쳐 조건: 위험인물 최초 확정 순간 or 계속 송신 원하면 매프레임)
        # 여기서는 "타겟이 존재하면" 매 프레임 전송 + 메타 포함
        if self.target_id is not None and not self._danger_announced:
            meta_msg = String()
            meta_msg.data = json.dumps({
                "danger": True,                       # ★ Host가 기다리던 키
                "target_id": int(self.target_id),
                "timestamp": time.time()
            })
            self.pub_meta.publish(meta_msg)
            self._danger_announced = True  # 다시 안 보내게 잠금
        
        # --- 평시 프레임: 필요하면 비어있는 메타로 유지 ---
        elif self.target_id is None:
            nm = String(); nm.data = "none"
            self.pub_meta.publish(nm)

        # 제어 로직 ----------------------------------------------------------
        self.lost_stop_delay = 3
        # 현재 모드에 따라 C++로 명령 호출
        if self.cmd_mode == 0:
            # 이미 Stop 상태 유지
            pass

        elif self.cmd_mode == 1:
            # Follow @ 3m
            if self.target_id is None:
                self._call_cpp("stop")
            else:
                sel = [p for p in persons if p["id"] == self.target_id]
                if sel:
                    (u, v, Zmm) = sel[0]["uvZ"]
                    u0 = int(self.intr.ppx)

                    dist_m = max(0.20, Zmm / 1000.0)  # 최소 0.2m 보호
                    px_err = (u - u0)

                    # 데드존
                    if abs(dist_m - 3.0) < 0.12:
                        depth_term = 0.0
                    else:
                        depth_term = (dist_m - 3.0) * 0.6   # 3m 목표

                    if abs(px_err) < 10:
                        yaw_term = 0.0
                    else:
                        yaw_term = -px_err * 0.002

                    vx   = float(np.clip(depth_term, -0.8, 0.8))
                    vyaw = float(np.clip(yaw_term,   -0.8, 0.8))

                    # 15Hz 스로틀로 전송
                    now = time.time()
                    if now - self._last_cmd_ts >= self._cmd_period:
                        self._call_cpp("move", vx, 0.0, vyaw)
                        self._last_cmd_ts = now

                    self.last_target_pose = (u, v, Zmm)
                else:
                    # 타깃 유실: 잠깐 유지 후 정지
                    self.lost_frames += 1
                    if self.lost_frames <= self.lost_stop_delay:
                        self._call_cpp("move", 0.0, 0.0, 0.0)
                    else:
                        self._call_cpp("stop")

        elif self.cmd_mode == 2:
            base = None
            if self.target_id is not None:
                sel = [p for p in persons if p["id"] == self.target_id]
                if sel: base = sel[0]["uvZ"]
            if base is None:
                base = self.last_target_pose

            if base is None:
                self._call_cpp("stop")
            else:
                (u, v, Zmm) = base
                dist_m = max(0.20, Zmm / 1000.0)
                u0 = int(self.intr.ppx)
                px_err = (u - u0)

                if abs(dist_m - 1.0) < 0.12 and abs(px_err) < 40:
                    self._call_cpp("stop")
                    time.sleep(0.2)
                    self._call_cpp("hello")
                    time.sleep(0.4)
                    self._call_cpp("damp")

                    # 호스트에 완료 통보
                    fin = String()
                    fin.data = json.dumps({"status":"finish", 
                                           "note":"Approach@1m + Hello + Damp done"})
                    self.pub_meta.publish(fin)

                    self.get_logger().info("Finish")
                    rclpy.shutdown()
                    raise SystemExit
                else:
                    depth_term = (dist_m - 1.0) * 0.3      # 천천히 접근
                    yaw_term   = -px_err * 0.002

                    vx   = float(np.clip(depth_term, -0.3, 0.3))
                    vyaw = float(np.clip(yaw_term,   -0.6, 0.6))

                    now = time.time()
                    if now - self._last_cmd_ts >= self._cmd_period:
                        self._call_cpp("move", vx, 0.0, vyaw)
                        self._last_cmd_ts = now

        # -------------------------------------------------------------------

        cv2.imshow("Grand-ICT", vis)
        cv2.waitKey(1)

    def destroy_node(self):
        try:
            self._call_cpp(["stop"])
        except:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = GrandICT()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

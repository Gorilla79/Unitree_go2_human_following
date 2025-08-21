import cv2
import sys, select
import json
import numpy as np
import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

# 좌표계 캔버스(mm -> px 간단 스케일)
SCALE = 0.2  # 1mm -> 0.2px  (1m=1000mm -> 200px)

class HostViewer(Node):
    def __init__(self):
        super().__init__('grand_host')

        self.sub_coord = self.create_subscription(String, 'coordinate', self.on_coord, 10)
        self.sub_img   = self.create_subscription(CompressedImage, 'image/compressed', self.on_img, 10)
        self.sub_meta  = self.create_subscription(String, 'image/meta', self.on_meta, 10)

        self.pub_cmd   = self.create_publisher(String, 'grand_cmd', 10)

        self.last_coords = []  # [(label, Xmm, Ymm), ...]
        self.last_img = None
        self.last_meta = None

        cv2.namedWindow("Grand-Host", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Grand-Host", 1280, 960)
        cv2.namedWindow("Target-Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Target-Image", 640, 360) 

        self.timer = self.create_timer(0.05, self.loop)  # 20Hz UI
        
        self.awaiting_decision = False
        self._prompt_printed = False
        
        self._last_ack_text = ""
        self._last_ack_time = 0.0

    def on_coord(self, msg: String):
        s = msg.data.strip()
        if s == "none" or len(s)==0:
            self.last_coords = []
            return
        # 파싱: "id:0,X:123,Y:456", ...  또는 "danger_id:K,X:...,Y:..."
        parts = [p.strip() for p in s.split(",")]
        out = []
        i = 0
        while i < len(parts):
            token = parts[i]
            if token.startswith("id:") or token.startswith("danger_id:"):
                label = token
                # 다음 두 토큰이 X:?, Y:? 일 것으로 가정
                if i+2 < len(parts):
                    x_tok = parts[i+1]
                    y_tok = parts[i+2]
                    if x_tok.startswith("X:") and y_tok.startswith("Y:"):
                        try:
                            Xmm = int(x_tok.split(":")[1])
                            Ymm = int(y_tok.split(":")[1])
                            out.append((label, Xmm, Ymm))
                            i += 3
                            continue
                        except:
                            pass
            # 혹은 한 사람 블록이 큰 따옴표로 묶여있는 경우 대비(생략)
            i += 1
        self.last_coords = out

    def on_img(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        im = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if im is not None:
            self.last_img = im

    def on_meta(self, msg: String):
        self.last_meta = msg.data
        try:
            meta = json.loads(self.last_meta)
            if meta.get("danger"):
                self.awaiting_decision = True
                self._prompt_printed = False
            if "ack" in meta:
                print(f"[Host] ACK from onboard: {meta['ack']} at {meta.get('ts','')}")
            if meta.get("status") == "finish":
                print("[Host] Onboard reports Finish. Exiting viewer soon.")
        except Exception:
            pass

    def loop(self):
        # 좌표계 캔버스
        canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        # 원점(로봇) 표시 (아래쪽 중앙보다 조금 아래)
        origin = (640, 720)  # (cx, cy) pixel
        cv2.circle(canvas, origin, 6, (0,0,0), -1)
        cv2.putText(canvas, "ROBOT(0,0)", (origin[0]+10, origin[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        # 축
        cv2.line(canvas, (0,origin[1]), (800,origin[1]), (0,0,0), 1)     # X축
        cv2.line(canvas, (origin[0],0), (origin[0],600), (0,0,0), 1)     # Y축

        # 눈금
        for dx in range(-3500, 3501, 500):  # mm
            px = int(origin[0] + dx*SCALE)
            cv2.line(canvas, (px, origin[1]-5), (px, origin[1]+5), (0,0,0), 1)
            cv2.putText(canvas, f"{dx}", (px-10, origin[1]+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
        for dy in range(0, 4001, 500):      # mm (앞방향 +Y)
            py = int(origin[1] - dy*SCALE)
            cv2.line(canvas, (origin[0]-5, py), (origin[0]+5, py), (0,0,0), 1)
            cv2.putText(canvas, f"{dy}", (origin[0]+8, py+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

        # 점 찍기
        for (label, Xmm, Ymm) in self.last_coords:
            px = int(origin[0] + Xmm*SCALE)
            py = int(origin[1] - Ymm*SCALE)
            col = (0,0,255) if label.startswith("danger_id") else (0,0,255)  # 빨간 점 통일
            cv2.circle(canvas, (px,py), 5, col, -1)
            cv2.putText(canvas, label, (px+6, py-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        cv2.imshow("Grand-Host", canvas)

        if self.last_img is not None:
            show = self.last_img.copy()
            # 위험 프롬프트
            if self.awaiting_decision:
                if not self._prompt_printed:
                    print("\n[Host] what is the next? (0: reflash, 1: follow@3m, 2: approach@1m+Hello)")
                    self._prompt_printed = True
                cv2.rectangle(show, (0,0), (show.shape[1], 60), (0,0,0), -1)
                cv2.putText(show, "what is the next? (0: reflash, 1: follow@3m, 2: approach@1m+Hello)",
                    (12,38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    
            # 로봇 ACK 도착시 상단에 잠깐 표시
            if self._last_ack_text and time.time() - self._last_ack_time < 1.5:
                cv2.rectangle(show, (0,60), (show.shape[1], 100), (50,50,50), -1)
                cv2.putText(show, self._last_ack_text,
                    (12,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        
            cv2.imshow("Target-Image", show)
        
        # --- 키보드 입력 처리 ---
        # 창 포커스 있을 때: OpenCV 키
        k = cv2.waitKey(1) & 0xFF
        key_sent = False
        if k == ord('0'):
            self.pub_cmd.publish(String(data="0"))
            self.awaiting_decision = False
            key_sent = True
        elif k == ord('1'):
            self.pub_cmd.publish(String(data="1"))
            self.awaiting_decision = False
            key_sent = True
        elif k == ord('2'):
            self.pub_cmd.publish(String(data="2"))
            self.awaiting_decision = False
            key_sent = True
        elif k == 27:
            rclpy.shutdown()
            return
            
        # 터미널 포커스일 때: 비차단 stdin
        if not key_sent:
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == '0':
                    self.pub_cmd.publish(String(data="0"))
                    self.awaiting_decision = False
                    print("[Host] sent 0 (reflash)")
                elif ch == '1':
                    self.pub_cmd.publish(String(data="1"))
                    self.awaiting_decision = False
                    print("[Host] sent 1 (follow@3m)")
                elif ch == '2':
                    self.pub_cmd.publish(String(data="2"))
                    self.awaiting_decision = False
                    print("[Host] sent 2 (approach@1m+Hello)")

def main():
    rclpy.init()
    node = HostViewer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

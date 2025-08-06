import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import torch

# Define skeleton structure based on the provided diagram
skeleton_connections = [
    (5, 7), (7, 9),  # Left shoulder to left hand
    (6, 8), (8, 10), # Right shoulder to right hand
    (5, 6),           # Shoulder to shoulder
    (5, 11), (6, 12), # Shoulders to hips
    (11, 12),         # Hip to hip
    (11, 13), (13, 15), # Left hip to left foot
    (12, 14), (14, 16)  # Right hip to right foot
]

# Initialize YOLO models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_human = YOLO("yolov8n.pt").to(device)  # YOLOv8 for human detection
model_pose = YOLO("yolov8n-pose.pt").to(device)  # YOLOv8 Pose model
model_weapon_hammer = YOLO("/home/unitree/dataset/best_hammer.pt").to(device)
model_weapon_knife = YOLO("/home/unitree/dataset/best_knife.pt").to(device)
model_weapon_gun = YOLO("/home/unitree/dataset/best_gun.pt").to(device)
model_weapon_bat = YOLO("/home/unitree/dataset/best_bat.pt").to(device)
model_weapon_axe = YOLO("/home/unitree/dataset/best_axe.pt").to(device)
print("YOLOv8 models loaded.")

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Align depth to color stream
align = rs.align(rs.stream.color)

# Confidence threshold
confidence_threshold = 0.6
overlap_threshold = 0.05  # Minimum 10% overlap for dangerous determination
frame_threshold = 20  # Frames for persistent dangerous status

# Dangerous tracking variables
frame_counters = {}
target_dangerous_id = None  # Initialize the variable here
hand_weapon_counters = {}  # Track weapon detections near hands

# Function to update hand weapon counters
def update_hand_weapon_counters(human_id, hand_position, weapon_boxes, threshold=50):
    for weapon_box in weapon_boxes:
        wx1, wy1, wx2, wy2 = weapon_box
        wx_center = (wx1 + wx2) // 2
        wy_center = (wy1 + wy2) // 2
        hx, hy = hand_position

        if abs(wx_center - hx) <= threshold and abs(wy_center - hy) <= threshold:
            if human_id not in hand_weapon_counters:
                hand_weapon_counters[human_id] = 0
            hand_weapon_counters[human_id] += 1
            return True
    return False


# Mouse callback for depth measurement
clicked_point = None
clicked_depth = None

def get_depth_at_click(event, x, y, flags, param):
    global clicked_point, clicked_depth
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cv2.namedWindow("Color and Depth Detection")
cv2.setMouseCallback("Color and Depth Detection", get_depth_at_click)

# Function to calculate overlap ratio
def calculate_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    overlap_x_min = max(x1_min, x2_min)
    overlap_y_min = max(y1_min, y2_min)
    overlap_x_max = min(x1_max, x2_max)
    overlap_y_max = min(y1_max, y2_max)

    if overlap_x_max > overlap_x_min and overlap_y_max > overlap_y_min:
        overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        return overlap_area / box1_area
    return 0

# Adjust text position to stay within image bounds
def adjust_text_position_side(x, y, texts, box, image_shape):
    positions = []
    text_height = 15  # Approximate height of one line of text

    for i, text in enumerate(texts):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_width, _ = text_size

        # Default position
        adjusted_x = x
        adjusted_y = y + i * text_height

        # Check if text exceeds image boundaries
        if adjusted_x + text_width > image_shape[1]:
            # Shift text to the left of the bounding box
            adjusted_x = box[0] - text_width - 10
        if adjusted_y - text_height < 0:
            adjusted_y = text_height + 10 + i * text_height

        positions.append((adjusted_x, adjusted_y))
    return positions

def process_weapon_detections(results_hammer, results_knife, results_gun, results_bat, results_axe, confidence_threshold):
    all_boxes = []
    all_classes = []
    all_confidences = []

    # Collect detections from all models
    models = [
        (results_hammer, "Hammer", 0.7),
        (results_knife, "Knife", 0.7),
        (results_gun, "Gun", 0.7),
        (results_bat, "Bat", 0.7),
        (results_axe, "Axe", 0.9),
    ]

    for result, class_name, threshold in models:
        if result[0].boxes:
            boxes = result[0].boxes.xyxy.cpu().numpy()
            confidences = result[0].boxes.conf.cpu().numpy()
            filtered_indices = np.where(confidences >= threshold)[0]
            boxes = boxes[filtered_indices]
            confidences = confidences[filtered_indices]

            all_boxes.extend(boxes)
            all_classes.extend([class_name] * len(filtered_indices))
            all_confidences.extend(confidences)

    if len(all_boxes) == 0:
        return np.empty((0, 4)), [], []

    # Convert to numpy array
    all_boxes = np.array(all_boxes)
    all_confidences = np.array(all_confidences)
    all_classes = np.array(all_classes)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=all_boxes.tolist(),
        scores=all_confidences.tolist(),
        score_threshold=confidence_threshold,
        nms_threshold=0.5,  # Adjust overlap threshold as needed
    )

    if len(indices) > 0:
        indices = indices.flatten()  # Flatten indices to a list
        all_boxes = all_boxes[indices]
        all_classes = all_classes[indices]
        all_confidences = all_confidences[indices]
    else:
        all_boxes = np.empty((0, 4))
        all_classes = []
        all_confidences = []

    return all_boxes, all_classes, all_confidences

try:
    prev_frame = None
    prev_keypoints = None

    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()

        # Align depth to color
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 이미지 중심 좌표 계산
        height, width = color_image.shape[:2]
        cx, cy = width // 2, height // 2

        # Normalize depth image for display
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # Run YOLO detection
        results_human = model_human.track(color_image, classes=[0], persist=True, device=device)
        results_hammer = model_weapon_hammer(color_image, device=device)
        results_knife = model_weapon_knife(color_image, device=device)
        results_gun = model_weapon_gun(color_image, device=device)
        results_bat = model_weapon_bat(color_image, device=device)
        results_axe = model_weapon_axe(color_image, device=device)
        
        # Process weapon detection results
        boxes_weapon, classes_weapon, confidences_weapon = process_weapon_detections(
            results_hammer, results_knife, results_gun, results_bat, results_axe, confidence_threshold
        )

        print("Detected weapons:", classes_weapon)

        # Extract human detections
        boxes_human = []
        ids_human = []
        if results_human[0].boxes:
            human_boxes = results_human[0].boxes
            human_confidences = human_boxes.conf.cpu().numpy()
            boxes_human = human_boxes.xyxy.cpu().numpy()[human_confidences >= confidence_threshold]
            ids_human = (
                human_boxes.id.cpu().numpy()[human_confidences >= confidence_threshold]
                if human_boxes.id is not None
                else []
            )

        current_frame_human_ids = []

        # Annotate humans and check for dangerous status
        for box, human_id in zip(boxes_human, ids_human):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            depth = depth_frame.get_distance(center_x, center_y)

            # 중심 기준 좌표 계산
            centered_x = center_x - cx
            centered_y = center_y - cy

            is_dangerous = False

            # Check overlap with weapons (hammer or knife or gun or bat or axe)
            for weapon_box, weapon_class in zip(boxes_weapon, classes_weapon):
                overlap_ratio = calculate_overlap(box, weapon_box)
                if overlap_ratio >= overlap_threshold:
                    if human_id not in current_frame_human_ids:
                        current_frame_human_ids.append(human_id)  # 중복 없이 추가
                    if human_id not in frame_counters:
                        frame_counters[human_id] = 0
                    frame_counters[human_id] += 1  # 카운트 증가

                    if frame_counters[human_id] >= frame_threshold:
                        target_dangerous_id = human_id  # 위험 대상으로 갱신
                        is_dangerous = True
                    break

            # Label and color settings based on danger status
            if is_dangerous or human_id == target_dangerous_id:
                label = f"ID:{human_id}, Dangerous"
                color = (0, 0, 255)  # Red for dangerous
            else:
                label = f"ID:{human_id}, Human"
                color = (0, 255, 0)  # Green for non-dangerous
            depth_text = f"X:{centered_x}, Y:{centered_y}, Depth:{depth:.2f}m"

            # Draw bounding box
            cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)

            # Adjust text position and output
            positions = adjust_text_position_side(
                x1, y1 - 2, [label, depth_text], (x1, y1, x2, y2), color_image.shape
            )
            for text, (pos_x, pos_y) in zip([label, depth_text], positions):
                cv2.putText(color_image, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Mark center point
            cv2.circle(color_image, (center_x, center_y), 10, color, -1)

            # Add to depth map
            adjusted_x = min(max(center_x, 10), depth_colormap.shape[1] - 50)
            adjusted_y = min(max(center_y, 20), depth_colormap.shape[0] - 10)

            # 중심 기준 좌표로 변경
            centered_x = adjusted_x - cx
            centered_y = adjusted_y - cy

            cv2.circle(depth_colormap, (adjusted_x, adjusted_y), 10, color, -1)

            depth_label = f"X:{center_x}, Y:{center_y}, Depth:{depth:.2f}m"
            depth_label_positions = adjust_text_position_side(
                adjusted_x + 10, adjusted_y - 10, [depth_label], (adjusted_x, adjusted_y, adjusted_x, adjusted_y), depth_colormap.shape
            )
            for text, (pos_x, pos_y) in zip([depth_label], depth_label_positions):
                cv2.putText(depth_colormap, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Perform pose estimation on the cropped human
            if 0.3 <= depth <= 3.0:
                cropped_human = color_image[y1:y2, x1:x2]
                results_pose = model_pose(cropped_human)
            
                # Draw keypoints and skeleton connections
                for pose_result in results_pose:
                    if pose_result.keypoints:
                        keypoints = pose_result.keypoints.xy.cpu().numpy()
                        for person in keypoints:
                            if len(person) >= 11:  # Ensure there are enough keypoints
                                for i, keypoint in enumerate(person):
                                    x, y = keypoint[:2]
                                    if x > 0 and y > 0:
                                        cv2.circle(color_image, (int(x + x1), int(y + y1)), 5, (0, 0, 255), -1)
                                for joint_start, joint_end in skeleton_connections:
                                    if joint_start < len(person) and joint_end < len(person):
                                        x_start, y_start = person[joint_start][:2]
                                        x_end, y_end = person[joint_end][:2]
                                        if x_start > 0 and y_start > 0 and x_end > 0 and y_end > 0:
                                            if (joint_start, joint_end) in [(5, 7), (7, 9), (6, 8), (8, 10)]:
                                                color = (0, 255, 255)
                                            elif (joint_start, joint_end) in [(11, 13), (13, 15), (12, 14), (14, 16)]:
                                                color = (255, 0, 0)
                                            else:
                                                color = (0, 255, 0)
                                            cv2.line(color_image, (int(x_start + x1), int(y_start + y1)), (int(x_end + x1), int(y_end + y1)), color, 2)
            
                                # Check for weapons near hands
                                left_hand = person[9][:2]
                                right_hand = person[10][:2]
            
                                for hand in [left_hand, right_hand]:
                                    hx, hy = map(int, hand)
                                    if hx > 0 and hy > 0:
                                        hand_region = color_image[hy-20:hy+20, hx-20:hx+20]
                                        if hand_region.size > 0:  # Ensure the region is not empty
                                            hand_region_resized = cv2.resize(hand_region, (224, 224))  # Resize to a standard size
                                        
                                            # Run weapon detection models on hand regions
                                            results_hand_hammer = model_weapon_hammer(hand_region_resized)
                                            results_hand_knife = model_weapon_knife(hand_region_resized)
                                            results_hand_gun = model_weapon_gun(hand_region_resized)
                                            results_hand_bat = model_weapon_bat(hand_region_resized)
                                            results_hand_axe = model_weapon_axe(hand_region_resized)
                                        
                                            # Check if any weapon is detected in the hand region
                                            if any(result[0].boxes for result in [results_hand_hammer, results_hand_knife, results_hand_gun, results_hand_bat, results_hand_axe]):
                                                hand_region = color_image[hy-50:hy+50, hx-50:hx+50]
                                                if hand_region.size > 0:  # Ensure the region is not empty
                                                    hand_region_resized = cv2.resize(hand_region, (224, 224))  # Resize to a standard size
                                                
                                                    results_hand_hammer = model_weapon_hammer(hand_region_resized)
                                                    results_hand_knife = model_weapon_knife(hand_region_resized)
                                                    results_hand_gun = model_weapon_gun(hand_region_resized)
                                                    results_hand_bat = model_weapon_bat(hand_region_resized)
                                                    results_hand_axe = model_weapon_axe(hand_region_resized)
                                                
                                                    if any(result[0].boxes for result in [results_hand_hammer, results_hand_knife, results_hand_gun, results_hand_bat, results_hand_axe]):
                                                        if update_hand_weapon_counters(human_id, (hx, hy), boxes_weapon):
                                                            is_dangerous = True
                                                            target_dangerous_id = human_id
                                                            break

        # Annotate weapon detections
        for box, weapon_class, confidence in zip(boxes_weapon, classes_weapon, confidences_weapon):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            depth = depth_frame.get_distance(center_x, center_y)

            # 중심 기준 좌표 계산
            centered_x = center_x - cx
            centered_y = center_y - cy

            # Draw bounding box
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Annotate class name and confidence
            label = f"{weapon_class} ({confidence:.2f}), Depth:{depth:.2f}m"
            cv2.putText(
                color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )

            # Add to depth map
            adjusted_x = min(max(center_x, 10), depth_colormap.shape[1] - 50)
            adjusted_y = min(max(center_y, 20), depth_colormap.shape[0] - 10)
            cv2.circle(depth_colormap, (adjusted_x, adjusted_y), 10, (0, 255, 255), -1)

            weapon_label = f"{weapon_class} ({confidence:.2f}), Depth:{depth:.2f}m"
            weapon_label_positions = adjust_text_position_side(
                adjusted_x + 10, adjusted_y - 10, [weapon_label], (adjusted_x, adjusted_y, adjusted_x, adjusted_y), depth_colormap.shape
            )
            for text, (pos_x, pos_y) in zip([weapon_label], weapon_label_positions):
                cv2.putText(depth_colormap, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Annotate clicked point
        if clicked_point:
            click_x, click_y = clicked_point
            clicked_depth = depth_frame.get_distance(click_x, click_y)
            
            #중심점 좌표 계산
            centered_click_x = click_x - cx
            centered_click_y = click_y - cy

            adjusted_x = min(max(click_x, 10), depth_colormap.shape[1] - 50)
            adjusted_y = min(max(click_y, 20), depth_colormap.shape[0] - 10)
            cv2.circle(color_image, (click_x, click_y), 10, (0, 255, 255), -1)
            cv2.circle(depth_colormap, (adjusted_x, adjusted_y), 10, (0, 255, 255), -1)

            click_label = f"X:{centered_click_x}, Y:{centered_click_y}, Depth:{clicked_depth:.2f}m"
            click_label_positions = adjust_text_position_side(
                adjusted_x + 10, adjusted_y - 10, [click_label], (adjusted_x, adjusted_y, adjusted_x, adjusted_y), depth_colormap.shape
            )
            for text, (pos_x, pos_y) in zip([click_label], click_label_positions):
                cv2.putText(depth_colormap, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Stack images side-by-side
        combined_image = np.hstack((color_image, depth_colormap))
        cv2.imshow("Color and Depth Detection", combined_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            clicked_point = None
            frame_counters.clear()
            target_dangerous_id = None
            hand_weapon_counters.clear()

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

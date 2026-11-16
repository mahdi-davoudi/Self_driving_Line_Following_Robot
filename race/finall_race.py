import cv2
import numpy as np
import time
import glob
import serial
import math
from typing import Tuple, Dict, Optional


# Servo limits (must match the Arduino sketch)
STEER_CENTER = 90
STEER_MIN = 50
STEER_MAX = 125
FALLBACK_ANGLE = STEER_CENTER + 25   # steering angle used when both lines are lost

# Speed (PWM) limits (the Arduino MAX_SPEED must be >= MAX_SPEED here)
MAX_SPEED = 210              # speed on a straight path
MIN_SPEED = 180              # absolute lower limit
TURN_SPEED_DROP = 15         # speed lost at the sharpest turn (use 10..15)
UNCERTAIN_SPEED_CAP = 185    # max speed while a lane is being held/estimated
ARDUINO_MAX_PWM = 210

# Steering gain parameters
K_MIN = 0.3
K_MAX = 1.5

# Lane detection parameters
CANNY_LOW = 60
CANNY_HIGH = 160
HOUGH_THRESHOLD = 50
MIN_LINE_LENGTH = 30
MAX_LINE_GAP = 60

# Lane-loss handling (applies to BOTH the right/solid and left/dashed lines).
# When a lane is not detected in the current frame, hold the last known position
# (or estimate it from the other lane) for a short window, and only fall back
# to a fixed steering angle once that window is exceeded.
LANE_HOLD_FRAMES = 10        # frames to keep using last-known/estimated position
LANE_WIDTH_EMA_ALPHA = 0.05  # how fast the estimated lane width (px) adapts
LANE_WIDTH_INIT_PX = 140     # initial lane width guess in pixels -- tune to your track

# Frame size
F_Width = 384
F_Height = 216

# ROI for right lane detection (by percentage)
ROI_TOP_RL = 0.65
ROI_BOTTOM_RL = 1.0
ROI_LEFT_RL = 0.65
ROI_RIGHT_RL = 0.9

# ROI for left lane detection (by percentage)
ROI_TOP_LL = 0.65
ROI_BOTTOM_LL = 1.0
ROI_LEFT_LL = 0.0
ROI_RIGHT_LL = 0.4

# Drawing and showing the debug window costs FPS. Keep False while racing.
DEBUG_SHOW = False

# Slope filters of the lane lines (editable live from the dashboard)
RIGHT_MIN_ABS_SLOPE = 0.4
LEFT_MIN_ABS_SLOPE = 0.3


USE_DASHBOARD = True
AUTO_START = True        # False -> the robot waits for the START button in the dashboard
DASHBOARD_PORT = 8000

if USE_DASHBOARD:
    import dashboard

# SERIAL
LAST_CMD = None
def send_command(command: bytes) -> bool:
    global LAST_CMD
    if LAST_CMD == command:
        return False
    LAST_CMD = command

    if serial_connection is not None and serial_connection.is_open:
        serial_connection.write(command)
        return True
    print(f"Serial connection not available for command: {command}")
    return False


def find_port() -> Optional[str]:
    ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
    return ports[0] if ports else None


try:
    port = find_port()
    if port is None:
        raise serial.SerialException("No Arduino port found")
    print("Using serial port:", port)
    serial_connection = serial.Serial(port, 115200, timeout=1)
    time.sleep(2)  
    send_command(b'STOP\r\n')
    send_command(f'SPD{MIN_SPEED}\r\n'.encode())
    send_command(f'S{STEER_CENTER}\r\n'.encode())
except serial.SerialException as e:
    print("Failed to initialize serial port. Robot control disabled.", e)
    serial_connection = None


# CAMERA
class CameraConfig:
    def __init__(self, width: int = F_Width, height: int = F_Height):
        self.width = width
        self.height = height
        self.use_rpi_cam = False
        self.cap = None
        self.picam2 = None
        self._initialize_camera()

    def _initialize_camera(self) -> None:
        try:
            from picamera2 import Picamera2
            self.use_rpi_cam = True
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.set_controls({
                "AeEnable": False,
                "AwbEnable": False,
                "AnalogueGain": 1.0,
                "ExposureTime": 35000,
                "Brightness": 0.4,
                "Contrast": 1.0,
                "Saturation": 1.0,
                "Sharpness": 0.0,
                "NoiseReductionMode": 0,
                "FrameDurationLimits": (50000, 50000),
            })
            self.picam2.start()
            time.sleep(1)
        except ImportError:
            self.use_rpi_cam = False
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def capture_frame(self) -> Optional[np.ndarray]:
        if self.use_rpi_cam:
            frame = self.picam2.capture_array()
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
        # Ensure frame is resized to configured size
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def release(self) -> None:
        if self.use_rpi_cam and self.picam2 is not None:
            self.picam2.stop()
        elif self.cap:
            self.cap.release()


# STEERING
class Steering:
    def __init__(self):
        self.max_deviation = max(STEER_MAX - STEER_CENTER, STEER_CENTER - STEER_MIN)

    def compute_angle(self, error: float, error_max: float, lane_lost: bool) -> float:
        if lane_lost:
            return float(FALLBACK_ANGLE)

        # Adaptive gain based on error magnitude
        error_norm = min(abs(error) / error_max, 1.0)
        gain = K_MIN + (K_MAX - K_MIN) * error_norm
        angle = STEER_CENTER + gain * error
        return max(STEER_MIN, min(STEER_MAX, angle))

    def compute_speed(self, angle: float, lane_uncertain: bool) -> int:

        turn_ratio = min(
            abs(angle - STEER_CENTER) / self.max_deviation,
            1.0
        ) 
        speed = MAX_SPEED - TURN_SPEED_DROP * turn_ratio
        if lane_uncertain:
            speed = min(speed, UNCERTAIN_SPEED_CAP)
        speed = int(round(speed))
        speed = max(MIN_SPEED, min(MAX_SPEED, speed))
        return speed


# ROBOT (serial output only)
class RobotController:
    def __init__(self):
        self.angle_old = None
        self.speed_old = None

    def drive(self, angle: float, speed: int) -> None:

        angle = int(round(angle))

        # Validate PWM before sending it to Arduino
        speed = int(speed)
        speed = max(0, min(ARDUINO_MAX_PWM, speed))

        # Send steering angle
        if angle != self.angle_old:
            send_command(f'S{angle}\r\n'.encode())
            self.angle_old = angle

        # Send speed to Arduino
        if speed != self.speed_old:
            command = f'SPD{speed}\r\n'.encode()

            send_command(command)
            self.speed_old = speed

        # Apply the updated PWM to the motors
        send_command(b'F\r\n')

    def close(self) -> None:
        send_command(b'STOP\r\n')
        if serial_connection is not None and serial_connection.is_open:
            serial_connection.close()


# VISION
class VisionProcessor:
    def __init__(self, config: dict):
        self.config = config
        # Per-side memory: last known center position + how many consecutive
        # frames it has been since that side was actually (raw) detected.
        self.last_center_left_x = None
        self.last_center_right_x = None
        self.left_lost_frames = 0
        self.right_lost_frames = 0

        # Running estimate of lane width in pixels (center_right_x - center_left_x),
        # updated only on frames where BOTH lines are freshly detected.
        self.estimated_lane_width = None

    def detect_lines(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        height, width = frame.shape[:2]
        frame_center = width // 2
        error_max = frame_center  # max possible pixel error

        right_lines_info = []
        left_lines_info = []
        longest_line_length = 0

        # Right lane ROI
        top_rl = int(height * self.config['ROI_TOP_RL'])
        bottom_rl = int(height * self.config['ROI_BOTTOM_RL'])
        left_rl = int(width * self.config['ROI_LEFT_RL'])
        right_rl = int(width * self.config['ROI_RIGHT_RL'])

        # Left lane ROI
        top_ll = int(height * self.config['ROI_TOP_LL'])
        bottom_ll = int(height * self.config['ROI_BOTTOM_LL'])
        left_ll = int(width * self.config['ROI_LEFT_LL'])
        right_ll = int(width * self.config['ROI_RIGHT_LL'])

        # Convert to gray ONCE per frame; both ROIs are slices of this image
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_rl = gray[top_rl:bottom_rl, left_rl:right_rl]
        gray_ll = gray[top_ll:bottom_ll, left_ll:right_ll]

        # Process right lane
        edge_rl = cv2.Canny(gray_rl, self.config['CANNY_LOW'], self.config['CANNY_HIGH'])
        edge_dilated_rl = cv2.dilate(edge_rl, np.ones((3, 3), np.uint8), iterations=1)
        lines_rl = cv2.HoughLinesP(edge_dilated_rl, 1, np.pi / 180, self.config['HOUGH_THRESHOLD'],
                                   minLineLength=self.config['MIN_LINE_LENGTH'], maxLineGap=self.config['MAX_LINE_GAP'])

        # center_right_x_raw / center_left_x_raw: only set when the line is
        # actually found in THIS frame.
        center_right_x_raw = None
        if lines_rl is not None:
            right_lines = []
            roi_height_rl = gray_rl.shape[0]
            for line in lines_rl:
                x1, y1, x2, y2 = line.flatten()
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) > self.config['RIGHT_MIN_ABS_SLOPE']:
                        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        x_bottom = x1 + (roi_height_rl - y1) / slope if abs(slope) > 1e-5 else x1
                        x_bottom = max(0, min(right_rl - left_rl - 1, x_bottom))
                        x_bottom_full = x_bottom + left_rl
                        if x_bottom_full >= frame_center:
                            right_lines.append((x1, y1, x2, y2, x_bottom, length))
                            right_lines_info.append((x1, y1, x2, y2, length))

            if right_lines:
                x_bottom_sum = sum(line[4] + left_rl for line in right_lines)
                center_right_x_raw = x_bottom_sum / len(right_lines)
                longest_line_length = max(longest_line_length, int(max(line[5] for line in right_lines)))

        # Process left lane
        edge_ll = cv2.Canny(gray_ll, self.config['CANNY_LOW'], self.config['CANNY_HIGH'])
        edge_dilated_ll = cv2.dilate(edge_ll, np.ones((3, 3), np.uint8), iterations=1)
        lines_ll = cv2.HoughLinesP(edge_dilated_ll, 1, np.pi / 180, self.config['HOUGH_THRESHOLD'],
                                   minLineLength=self.config['MIN_LINE_LENGTH'], maxLineGap=self.config['MAX_LINE_GAP'])

        center_left_x_raw = None
        if lines_ll is not None:
            left_lines = []
            roi_height_ll = gray_ll.shape[0]
            for line in lines_ll:
                x1, y1, x2, y2 = line.flatten()
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) > self.config['LEFT_MIN_ABS_SLOPE']:
                        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        x_bottom = x1 + (roi_height_ll - y1) / slope if abs(slope) > 1e-5 else x1
                        x_bottom = max(0, min(right_ll - left_ll - 1, x_bottom))
                        x_bottom_full = x_bottom + left_ll
                        if x_bottom_full <= frame_center:
                            left_lines.append((x1, y1, x2, y2, x_bottom, length))
                            left_lines_info.append((x1, y1, x2, y2, length))

            if left_lines:
                x_bottom_sum = sum(line[4] + left_ll for line in left_lines)
                center_left_x_raw = x_bottom_sum / len(left_lines)
                longest_line_length = max(longest_line_length, int(max(line[5] for line in left_lines)))

        right_detected_raw = center_right_x_raw is not None
        left_detected_raw = center_left_x_raw is not None

        # Only trust frames where BOTH lines were freshly detected.
        if right_detected_raw and left_detected_raw:
            current_width = center_right_x_raw - center_left_x_raw
            if current_width > 0:
                if self.estimated_lane_width is None:
                    self.estimated_lane_width = current_width
                else:
                    self.estimated_lane_width = (
                        LANE_WIDTH_EMA_ALPHA * current_width
                        + (1 - LANE_WIDTH_EMA_ALPHA) * self.estimated_lane_width
                    )
        lane_width = self.estimated_lane_width if self.estimated_lane_width is not None else LANE_WIDTH_INIT_PX

        # ---- Resolve RIGHT line position + confidence ----
        # Priority when not detected this frame:
        #   1) estimate from the (raw) left line + known lane width (adapts to turns)
        #   2) hold the last known right position
        #   3) give up -> not confident (triggers the safe fallback steering)
        if right_detected_raw:
            center_right_x = center_right_x_raw
            self.last_center_right_x = center_right_x
            self.right_lost_frames = 0
            right_confident = True
        else:
            self.right_lost_frames += 1
            if self.right_lost_frames <= LANE_HOLD_FRAMES:
                if left_detected_raw:
                    center_right_x = center_left_x_raw + lane_width
                    right_confident = True
                elif self.last_center_right_x is not None:
                    center_right_x = self.last_center_right_x
                    right_confident = True
                else:
                    center_right_x = right_rl + 10
                    right_confident = False
            else:
                center_right_x = right_rl + 10
                right_confident = False

        #  Resolve LEFT line position + confidence (mirrors the right side)
        if left_detected_raw:
            center_left_x = center_left_x_raw
            self.last_center_left_x = center_left_x
            self.left_lost_frames = 0
            left_confident = True
        else:
            self.left_lost_frames += 1
            if self.left_lost_frames <= LANE_HOLD_FRAMES:
                if right_detected_raw:
                    center_left_x = center_right_x_raw - lane_width
                    left_confident = True
                elif self.last_center_left_x is not None:
                    center_left_x = self.last_center_left_x
                    left_confident = True
                else:
                    center_left_x = left_ll
                    left_confident = False
            else:
                center_left_x = left_ll
                left_confident = False

        # True when at least one line was not freshly seen this frame
        # (used to reduce speed while the position is a guess).
        lane_uncertain = not (right_detected_raw and left_detected_raw)

        # True when both sides are lost beyond the hold window.
        lane_lost = not right_confident and not left_confident

        lane_type = ("both" if right_detected_raw and left_detected_raw
                     else "right" if right_detected_raw
                     else "left" if left_detected_raw
                     else "none")

        line_center_x = (center_right_x + center_left_x) / 2
        target_x = frame_center
        error = line_center_x - target_x

        lanes_frame = None
        if DEBUG_SHOW or USE_DASHBOARD:
            lanes_frame = self._draw_debug(
                frame, (left_rl, top_rl, right_rl, bottom_rl), (left_ll, top_ll, right_ll, bottom_ll),
                center_right_x, center_left_x, right_detected_raw, left_detected_raw,
                len(right_lines_info), len(left_lines_info), longest_line_length, lane_type)

        line_info = {
            'line_center_x': line_center_x,
            'target_x': target_x,
            'error': error,
            'error_max': error_max,
            'lane_type': lane_type,
            'center_right_x': center_right_x,
            'center_left_x': center_left_x,
            'lane_uncertain': lane_uncertain,
            'lane_lost': lane_lost,
        }
        return lanes_frame, line_info

    @staticmethod
    def _draw_debug(frame, rl_box, ll_box, center_right_x, center_left_x,
                    right_detected, left_detected, n_right, n_left, longest, lane_type):
        left_rl, top_rl, right_rl, bottom_rl = rl_box
        left_ll, top_ll, right_ll, bottom_ll = ll_box

        lanes_frame = frame.copy()
        cv2.rectangle(lanes_frame, (left_rl, top_rl), (right_rl, bottom_rl), (0, 255, 255), 2)
        cv2.putText(lanes_frame, "RL", (left_rl + 5, top_rl + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.rectangle(lanes_frame, (left_ll, top_ll), (right_ll, bottom_ll), (255, 255, 0), 2)
        cv2.putText(lanes_frame, "LL", (left_ll + 5, top_ll + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Thick line = freshly detected, thin line = held/estimated
        cv2.line(lanes_frame, (int(center_right_x), top_rl), (int(center_right_x), bottom_rl),
                 (255, 0, 0), 2 if right_detected else 1)
        cv2.line(lanes_frame, (int(center_left_x), top_ll), (int(center_left_x), bottom_ll),
                 (0, 0, 255), 2 if left_detected else 1)

        cv2.putText(lanes_frame, f"R Lines: {n_right}", (left_rl + 5, top_rl + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(lanes_frame, f"L Lines: {n_left}", (left_ll + 5, top_ll + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(lanes_frame, f"Len: {longest} px", (left_rl + 5, top_rl + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(lanes_frame, f"Lane: {lane_type}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return lanes_frame


# ============================================================
# MAIN VEHICLE
# ============================================================
class LaneTrackingVehicle:
    def __init__(self):
        self.config = {
            'F_Width': F_Width,
            'F_Height': F_Height,
            'CANNY_LOW': CANNY_LOW,
            'CANNY_HIGH': CANNY_HIGH,
            'HOUGH_THRESHOLD': HOUGH_THRESHOLD,
            'MIN_LINE_LENGTH': MIN_LINE_LENGTH,
            'MAX_LINE_GAP': MAX_LINE_GAP,
            'RIGHT_MIN_ABS_SLOPE': RIGHT_MIN_ABS_SLOPE,
            'LEFT_MIN_ABS_SLOPE': LEFT_MIN_ABS_SLOPE,
            'ROI_TOP_RL': ROI_TOP_RL,
            'ROI_BOTTOM_RL': ROI_BOTTOM_RL,
            'ROI_LEFT_RL': ROI_LEFT_RL,
            'ROI_RIGHT_RL': ROI_RIGHT_RL,
            'ROI_TOP_LL': ROI_TOP_LL,
            'ROI_BOTTOM_LL': ROI_BOTTOM_LL,
            'ROI_LEFT_LL': ROI_LEFT_LL,
            'ROI_RIGHT_LL': ROI_RIGHT_LL,
        }
        self.camera = CameraConfig(self.config['F_Width'], self.config['F_Height'])
        self.vision = VisionProcessor(self.config)
        self.steering = Steering()
        self.robot = RobotController()

        self.running = AUTO_START
        if USE_DASHBOARD:
            dashboard.state.config = self.config
            dashboard.calib.set_params({"GRAY_FROM_RGB": 1})   # this code converts with RGB2GRAY
            dashboard.start(port=DASHBOARD_PORT)
            print(f"[INFO] Dashboard: http://<raspberry-ip>:{DASHBOARD_PORT}")

    def _handle_command(self) -> None:
        cmd = dashboard.state.pop_command()
        if cmd == "stop":
            self.running = False
            send_command(b'STOP\r\n')
        elif cmd == "start":
            self.running = True

    def run(self) -> None:
        print("Robot started. Press Ctrl+C to stop.")
        try:
            last_log, frames, fps = time.time(), 0, 0
            while True:
                t0 = time.time()
                frame = self.camera.capture_frame()
                if time.time() - t0 > 0.3:
                    print(f"[WARN] slow capture: {time.time() - t0:.2f}s")
                if frame is None:
                    print("Failed to capture image")
                    break

                if USE_DASHBOARD:
                    dashboard.state.publish_raw(frame)
                    self._handle_command()

                lanes_frame, line_info = self.vision.detect_lines(frame)

                angle = self.steering.compute_angle(
                    line_info['error'], line_info['error_max'], line_info['lane_lost'])
                speed = self.steering.compute_speed(angle, line_info['lane_uncertain'])
                if self.running:
                    self.robot.drive(angle, speed)

                frames += 1
                if time.time() - last_log >= 1:
                    fps = frames
                    print(f"fps={fps} lane={line_info['lane_type']} "
                          f"angle={int(round(angle))} speed={speed}")
                    frames, last_log = 0, time.time()

                if USE_DASHBOARD:
                    dashboard.state.frames["lanes"] = lanes_frame
                    dashboard.state.telemetry = {
                        "running": self.running,
                        "fps": fps,
                        "lane": line_info['lane_type'],
                        "error_px": round(line_info['error'], 1),
                        "angle": int(round(angle)),
                        "speed": speed,
                    }

                if DEBUG_SHOW and lanes_frame is not None:
                    cv2.imshow("lanes", lanes_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        self.camera.release()
        self.robot.close()
        if DEBUG_SHOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    vehicle = LaneTrackingVehicle()
    vehicle.run()
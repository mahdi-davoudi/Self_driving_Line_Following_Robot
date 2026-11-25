import cv2
import numpy as np
import time
import glob
import serial
import math
from collections import deque
from lane_geometry import detect_side
from typing import Tuple, Dict, Optional


# Servo limits (must match the Arduino sketch)
STEER_CENTER = 90
STEER_MIN = 50
STEER_MAX = 125
FALLBACK_ANGLE = STEER_CENTER - 25   # fallback direction after steering polarity is reversed

# Speed (PWM) limits (the Arduino MAX_SPEED must be >= MAX_SPEED here)
MAX_SPEED = 140              # speed on a straight path
ARDUINO_MAX_PWM = 210
LANE_LOST_STOP_SECONDS = 3

# Steering gain parameters
K_MIN = 0.3
K_MAX = 1.5

# Lane detection parameters
CANNY_LOW = 80
CANNY_HIGH = 160
HOUGH_THRESHOLD = 50
MIN_LINE_LENGTH = 30
MAX_LINE_GAP = 60

# Lane-loss handling (applies to BOTH the right/solid and left/dashed lines).
# When a lane is not detected in the current frame, hold the last known position
# (or estimate it from the other lane) for a short window, and only fall back
# to a fixed steering angle once that window is exceeded.
LANE_HOLD_TIME_SEC = 0.4    # right solid-line short hold
LEFT_DASH_HOLD_INDEFINITE = True  # retain last dashed-line center/slope until next dash is detected
LANE_RECOVERY_STEP_DEG = 10.0     # steer increment toward the right when BOTH raw lines are missing
LANE_RECOVERY_INTERVAL_SEC = 0.5  # seconds between recovery increments
LANE_WIDTH_EMA_ALPHA = 0.05  # how fast the estimated lane width (px) adapts
LANE_WIDTH_INIT_PX = 293     # measured calibration median: about 293 px at 384x216
LANE_CENTER_OFFSET_PX = 0.0   # positive shifts target right in the image; tune from dashboard
LANE_WIDTH_MIN_PX = 250
LANE_WIDTH_MAX_PX = 340
LANE_WIDTH_MAX_CHANGE_PX = 80  # allow initial adoption of measured width
LANE_WIDTH_MAX_STEP_PX = 2.0   # per accepted update, after EMA
LANE_WIDTH_MEDIAN_WINDOW = 7
LINE_MIN_ANGLE_DEG = 24.0
LINE_MAX_ANGLE_DEG = 88.0
LINE_BOTTOM_TOLERANCE_PX = 22
LINE_ANGLE_TOLERANCE_DEG = 18
LINE_EXTRAPOLATION_MARGIN_PX = 25

# Frame size
F_Width = 384
F_Height = 216

# ROI for right lane detection (by percentage)
ROI_TOP_RL = 0.65
ROI_BOTTOM_RL = 1.0
ROI_LEFT_RL = 0.65
ROI_RIGHT_RL = 0.95

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

# SERIAL. A successful write means OS-buffered, not Arduino-executed.
HEARTBEAT_INTERVAL = 0.2
serial_connection = None
serial_error = None
_serial_cache = {}

class SerialControlError(RuntimeError):
    pass


def send_command(command: bytes, *, force=False) -> bool:
    global serial_error, serial_connection
    if serial_error is not None:
        raise SerialControlError(serial_error)
    if serial_connection is None or not serial_connection.is_open:
        serial_error = 'Arduino serial port disconnected'
        raise SerialControlError(serial_error)
    kind = (b'SPD' if command.startswith(b'SPD') else
            b'S' if command.startswith(b'S') and command[1:2].isdigit() else command.rstrip())
    # STOP and HB are never deduplicated. F is retransmitted only if not already active.
    if not force and kind not in (b'STOP', b'HB') and _serial_cache.get(kind) == command:
        return True  # already sent successfully in this session
    try:
        count = serial_connection.write(command)
        if count != len(command):
            raise SerialControlError(f'Partial serial write {count}/{len(command)}')
        if kind == b'STOP':
            _serial_cache.pop(b'F', None)
        else:
            _serial_cache[kind] = command
        return True
    except (serial.SerialException, OSError, SerialControlError) as exc:
        serial_error = f'Serial write failed: {exc}'
        _serial_cache.clear()
        raise SerialControlError(serial_error) from exc


def find_port() -> Optional[str]:
    ports = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
    return ports[0] if ports else None


def init_serial():
    global serial_connection, serial_error
    try:
        port = find_port()
        if not port:
            raise serial.SerialException('No Arduino port found')
        serial_connection = serial.Serial(port, 115200, timeout=0.1, write_timeout=0.2)
        print('Using serial port:', port)
        time.sleep(2)  # reset after opening serial
        send_command(b'STOP\r\n')
        send_command(f'SPD{MAX_SPEED}\r\n'.encode())
        send_command(f'S{STEER_CENTER}\r\n'.encode())
    except (serial.SerialException, OSError, SerialControlError) as exc:
        serial_error = f'Serial initialization failed: {exc}'
        print('[FAULT]', serial_error)
        if serial_connection is not None:
            try: serial_connection.close()
            except Exception: pass
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
                "ExposureTime": 30000,
                "Brightness": 0.0,
                "Contrast": 1.0,
                "Saturation": 1.0,
                "Sharpness": 0.0,
                "NoiseReductionMode": 0,
                "FrameDurationLimits": (40000, 40000),
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
        # Adaptive gain based on error magnitude
        error_norm = min(abs(error) / error_max, 1.0)
        gain = K_MIN + (K_MAX - K_MIN) * error_norm
        # Positive error means the detected lane center is right of image center.
        # Based on the observed servo/robot response, reduce the servo angle to steer left.
        angle = STEER_CENTER + gain * error
        return max(STEER_MIN, min(STEER_MAX, angle))

    def compute_speed(self, angle: float, lane_type: str) -> int:
        """Return constant motor PWM; steering angle does not affect speed."""
        return MAX_SPEED


# ROBOT (serial output only)
class RobotController:
    def __init__(self):
        self.angle_old = None
        self.speed_old = None

    def drive(self, angle: float, speed: int) -> None:
        angle = max(STEER_MIN, min(STEER_MAX, int(round(angle))))
        speed = max(0, min(ARDUINO_MAX_PWM, int(speed)))
        if angle != self.angle_old:
            if send_command(f'S{angle}\r\n'.encode()):
                self.angle_old = angle
        if speed != self.speed_old:
            if send_command(f'SPD{speed}\r\n'.encode()):
                self.speed_old = speed
        send_command(b'F\r\n')

    def stop(self) -> None:
        send_command(b'STOP\r\n')

    def close(self) -> None:
        if serial_connection is not None and serial_connection.is_open:
            try:
                if serial_error is None:
                    send_command(b'STOP\r\n')
            except SerialControlError:
                pass
            finally:
                try: serial_connection.close()
                except (OSError, serial.SerialException): pass


# VISION
class VisionProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.last_center_left_x = None
        self.last_left_slope_dx_dy = None
        self.last_center_right_x = None
        self.last_seen_left = None
        self.last_seen_right = None
        self.estimated_lane_width = float(LANE_WIDTH_INIT_PX)
        self.width_history = deque(maxlen=LANE_WIDTH_MEDIAN_WINDOW)

    def detect_lines(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        h, w = frame.shape[:2]
        frame_center = w // 2
        p = self.config
        rb = (int(h*p['ROI_TOP_RL']), int(h*p['ROI_BOTTOM_RL']),
              int(w*p['ROI_LEFT_RL']), int(w*p['ROI_RIGHT_RL']))
        lb = (int(h*p['ROI_TOP_LL']), int(h*p['ROI_BOTTOM_LL']),
              int(w*p['ROI_LEFT_LL']), int(w*p['ROI_RIGHT_LL']))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        right = detect_side(gray, rb, 'right', p)
        left = detect_side(gray, lb, 'left', p)
        rx, lx = right['center_x'], left['center_x']
        raw_right, raw_left = rx is not None, lx is not None
        now = time.monotonic()
        if raw_right and raw_left:
            measured = rx - lx
            prev = self.estimated_lane_width
            if (p['LANE_WIDTH_MIN_PX'] <= measured <= p['LANE_WIDTH_MAX_PX'] and
                    abs(measured - prev) <= p['LANE_WIDTH_MAX_CHANGE_PX']):
                self.width_history.append(measured)
                median = float(np.median(self.width_history))
                delta = p['LANE_WIDTH_EMA_ALPHA'] * (median - prev)
                delta = max(-p['LANE_WIDTH_MAX_STEP_PX'], min(p['LANE_WIDTH_MAX_STEP_PX'], delta))
                self.estimated_lane_width = max(p['LANE_WIDTH_MIN_PX'],
                                                min(p['LANE_WIDTH_MAX_PX'], prev + delta))
        lane_width = self.estimated_lane_width
        if raw_right:
            self.last_seen_right, self.last_center_right_x = now, rx
        if raw_left:
            self.last_seen_left, self.last_center_left_x = now, lx
            self.last_left_slope_dx_dy = left.get('slope_dx_dy')
        hold = float(p['LANE_HOLD_TIME_SEC'])
        valid_r = raw_right or (self.last_seen_right is not None and now - self.last_seen_right <= hold)
        valid_l = raw_left or (self.last_seen_left is not None and now - self.last_seen_left <= hold)
        if not raw_right:
            if valid_r and raw_left:
                rx = lx + lane_width
            elif valid_r:
                rx = self.last_center_right_x
            else:
                rx = None
        if not raw_left:
            if LEFT_DASH_HOLD_INDEFINITE and self.last_center_left_x is not None:
                # Dashed LEFT boundary: keep the last detected dash position/slope
                # until a new dash is detected; do not replace it with rx-lane_width.
                lx = self.last_center_left_x
            elif valid_l and self.last_center_left_x is not None:
                lx = self.last_center_left_x
            elif raw_right:
                # Before the first dash has ever been seen, estimate from solid RIGHT.
                lx = rx - lane_width
            else:
                lx = None
        # One visible side permits brief estimation even if opposite side has never been observed.
        if rx is None and raw_left:
            rx = lx + lane_width
        if lx is None and raw_right:
            lx = rx - lane_width
        # The solid RIGHT boundary is the primary tracking reference.
        # The dashed LEFT boundary is used only to update the lane-width estimate.
        # If LEFT disappears, infer its position from RIGHT and the last stable width.
        # If RIGHT disappears briefly, retain its last valid position; if only LEFT
        # remains, estimate RIGHT from LEFT + the stable width.
        lane_lost = not (raw_right or raw_left)
        lane_type = ('both' if raw_right and raw_left else 'right' if raw_right else
                     'left' if raw_left else 'none')

        if rx is not None and self.last_center_left_x is not None:
            # RIGHT is the live primary boundary; LEFT uses the last detected dash
            # (center + slope model) until the next dashed segment is detected.
            base_center_x = (float(rx) + float(lx)) / 2.0
        elif rx is not None:
            # Before the first left dash is seen, use the calibrated lane width.
            base_center_x = float(rx) - (lane_width / 2.0)
        elif lx is not None:
            # If RIGHT is temporarily missing, estimate it from the retained LEFT dash.
            estimated_right_x = float(lx) + lane_width
            base_center_x = (estimated_right_x + float(lx)) / 2.0
            rx = estimated_right_x
        else:
            base_center_x = float(frame_center)

        offset = float(p.get('LANE_CENTER_OFFSET_PX', LANE_CENTER_OFFSET_PX))
        line_center_x = base_center_x + offset
        lane_status = {
            'both': 'BOTH LINES DETECTED',
            'right': 'RIGHT ONLY - CENTER ESTIMATED',
            'left': 'LEFT ONLY - RIGHT ESTIMATED',
            'none': 'NO LINE DETECTED'
        }[lane_type]
        info = {'line_center_x': line_center_x, 'target_x': frame_center,
                'error': line_center_x-frame_center, 'error_max': frame_center,
                'lane_type': lane_type, 'lane_status': lane_status,
                'center_right_x': rx, 'center_left_x': lx,
                'lane_uncertain': not (raw_right and raw_left), 'lane_lost': lane_lost,
                'lane_width_px': lane_width, 'lane_center_offset_px': offset}
        lanes_frame = None
        if DEBUG_SHOW or USE_DASHBOARD:
            lanes_frame = self._draw_debug(frame, (rb[2], rb[0], rb[3], rb[1]),
                       (lb[2], lb[0], lb[3], lb[1]), rx, lx, raw_right, raw_left,
                       len(right['kept']), len(left['kept']),
                       max(right['longest'], left['longest']), lane_type,
                       self.last_left_slope_dx_dy)
        return lanes_frame, info

    @staticmethod
    def _draw_debug(frame, rl_box, ll_box, center_right_x, center_left_x,
                    right_detected, left_detected, n_right, n_left, longest, lane_type,
                    left_slope_dx_dy=None):
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
        if center_right_x is not None:
            cv2.line(lanes_frame, (int(center_right_x), top_rl), (int(center_right_x), bottom_rl),
                     (255, 0, 0), 2 if right_detected else 1)
        if center_left_x is not None:
            slope = float(left_slope_dx_dy or 0.0)
            x_bottom = int(round(center_left_x))
            x_top = int(round(center_left_x - (bottom_ll - top_ll) * slope))
            cv2.line(lanes_frame, (x_top, top_ll), (x_bottom, bottom_ll),
                     (0, 0, 255), 2 if left_detected else 1)

        cv2.putText(lanes_frame, f"R Lines: {n_right}", (left_rl + 5, top_rl + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(lanes_frame, f"L Lines: {n_left}", (left_ll + 5, top_ll + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(lanes_frame, f"Len: {longest} px", (left_rl + 5, top_rl + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        display_lane = {'both': 'BOTH', 'right': 'RIGHT ONLY - ESTIMATED CENTER',
                        'left': 'LEFT ONLY - ESTIMATED RIGHT', 'none': 'NO LINE'}[lane_type]
        cv2.putText(lanes_frame, f"Tracking: {display_lane}", (10, 70),
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
            'LANE_HOLD_TIME_SEC': LANE_HOLD_TIME_SEC,
            'LANE_WIDTH_EMA_ALPHA': LANE_WIDTH_EMA_ALPHA,
            'LANE_WIDTH_INIT_PX': LANE_WIDTH_INIT_PX,
            'LANE_CENTER_OFFSET_PX': LANE_CENTER_OFFSET_PX,
            'LANE_WIDTH_MIN_PX': LANE_WIDTH_MIN_PX,
            'LANE_WIDTH_MAX_PX': LANE_WIDTH_MAX_PX,
            'LANE_WIDTH_MAX_CHANGE_PX': LANE_WIDTH_MAX_CHANGE_PX,
            'LANE_WIDTH_MAX_STEP_PX': LANE_WIDTH_MAX_STEP_PX,
            'LINE_MIN_ANGLE_DEG': LINE_MIN_ANGLE_DEG,
            'LINE_MAX_ANGLE_DEG': LINE_MAX_ANGLE_DEG,
            'LINE_BOTTOM_TOLERANCE_PX': LINE_BOTTOM_TOLERANCE_PX,
            'LINE_ANGLE_TOLERANCE_DEG': LINE_ANGLE_TOLERANCE_DEG,
            'LINE_EXTRAPOLATION_MARGIN_PX': LINE_EXTRAPOLATION_MARGIN_PX,

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
        self.running = AUTO_START and serial_error is None
        self.fault = serial_error
        self.stopped_for_lane = False
        self.lane_missing_since = None
        self.last_heartbeat = 0.0
        self.lane_recovery_angle = float(STEER_CENTER)
        self.last_recovery_step = 0.0
        self.last_drive_angle = float(STEER_CENTER)
        
        if USE_DASHBOARD:
            dashboard.state.config = self.config
            dashboard.calib.set_params({"GRAY_FROM_RGB": 1})   # this code converts with RGB2GRAY
            dashboard.start(port=DASHBOARD_PORT)
            print(f"[INFO] Dashboard: http://<raspberry-ip>:{DASHBOARD_PORT}")

    def _handle_command(self) -> None:
        cmd = dashboard.state.pop_command()
        if cmd == "stop":
            self.running = False
            if self.fault is None:
                self.robot.stop()
        elif cmd == "start" and self.fault is None:
            send_command(b'ARM\r\n', force=True)
            self.robot.angle_old = None
            self.robot.speed_old = None
            _serial_cache.pop(b'F', None)
            self.running = True

    def run(self) -> None:
        print("Robot started. Press Ctrl+C to stop.")
        try:
            if self.running:
                send_command(b'ARM\r\n', force=True)
                send_command(b'HB\r\n', force=True)
                self.last_heartbeat = time.monotonic()
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
                    line_info['error'],
                    line_info['error_max'],
                    line_info['lane_lost']
                )
                now = time.monotonic()
                lane_type = line_info['lane_type']

                if lane_type == "none":
                    if self.lane_missing_since is None:
                        self.lane_missing_since = now
                        self.lane_recovery_angle = self.last_drive_angle
                        self.last_recovery_step = now
                    elif now - self.last_recovery_step >= LANE_RECOVERY_INTERVAL_SEC:
                        # Increasing servo angle steers RIGHT on this robot.
                        self.lane_recovery_angle = min(
                            STEER_MAX,
                            self.lane_recovery_angle + LANE_RECOVERY_STEP_DEG
                        )
                        self.last_recovery_step = now
                    angle = self.lane_recovery_angle
                else:
                    self.lane_missing_since = None
                    self.lane_recovery_angle = angle
                    self.last_drive_angle = angle
                
                if self.running and self.fault is None and now - self.last_heartbeat >= HEARTBEAT_INTERVAL:
                    if send_command(b'HB\r\n'):
                        self.last_heartbeat = now

                now = time.monotonic()
                lost_duration = (
                    now - self.lane_missing_since
                    if self.lane_missing_since is not None else 0.0
                )

                if lost_duration >= LANE_LOST_STOP_SECONDS:
                    speed = 0
                    if self.running:
                        self.robot.stop()
                        self.stopped_for_lane = True
                        self.running = False  # require explicit dashboard START to resume

                else:
                    # Fixed full speed: no turn-based or uncertainty-based speed reduction.
                    speed = MAX_SPEED
                    if self.running:
                        self.robot.drive(angle, speed)
                        self.stopped_for_lane = False

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
                        "fault": self.fault,
                        "lane_width_px": round(line_info["lane_width_px"], 1),
                        "fps": fps,
                        "lane": line_info['lane_type'],
                        "lane_status": line_info['lane_status'],
                        "left_dash_held": bool(self.vision.last_center_left_x is not None and line_info['lane_type'] == 'right'),
                        "recovery_angle": int(round(self.lane_recovery_angle)),
                        "lane_status": line_info['lane_status'],
                        "lane_center_offset_px": round(line_info["lane_center_offset_px"], 1),
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
        except (SerialControlError, OSError, cv2.error) as exc:
            self.fault = str(exc)
            self.running = False
            print('[FAULT] Robot halted:', self.fault)
            if USE_DASHBOARD:
                dashboard.state.publish_telemetry({'running': False, 'fault': self.fault})
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        self.robot.close()
        self.camera.release()
        if DEBUG_SHOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    init_serial()
    vehicle = LaneTrackingVehicle()
    vehicle.run()

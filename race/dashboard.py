"""
dashboard.py - web dashboard for the robot (FastAPI + MJPEG + WebSocket)
with a built-in lane CALIBRATION tab.

Open http://<raspberry-ip>:8000 from any browser on the same network.

Two ways to use it
------------------
1) Inside the race loop (race.py / city.py / fix_lost_right_line.py):

    import dashboard

    # once, in __init__:
    dashboard.state.config = self.config      # live config dict, editable from the browser
    dashboard.start(port=8000)                # web server runs in a background thread

    # every loop iteration:
    dashboard.state.publish_raw(frame)        # the raw camera frame (needed by the Calibration tab)
    dashboard.state.frames["lanes"] = lanes_frame
    dashboard.state.telemetry = {...}         # any dict of values you want to see
    cmd = dashboard.state.pop_command()       # "start" / "stop" / None

2) Standalone calibration (no robot, no serial) - the dashboard opens the camera itself:

    python dashboard.py

   Edit USE_DROIDCAM / DROIDCAM_URL below first. Do NOT use this while the race code is
   running: a Picamera2 / webcam can only be opened by one program.

Design rule: the control loop only *writes* references into `state`.
All JPEG encoding happens in the web thread and only while a browser is watching,
and all calibration maths runs in its own thread that sleeps unless the Calibration
tab is open in a visible browser tab, so the web server can never slow down (or block)
the control loop.
"""
import asyncio
import json
import math
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, Response, StreamingResponse

JPEG_QUALITY = 70        # lower = less CPU / bandwidth
STREAM_FPS_LIMIT = 20    # max frames per second sent to each browser
TELEMETRY_HZ = 10        # how often telemetry is pushed to each browser

# ---------------------------------------------------------------------------
# Standalone camera (only used by `python dashboard.py`)
# ---------------------------------------------------------------------------
USE_DROIDCAM = True                              # True: phone over Wi-Fi, False: Picamera2 / webcam
DROIDCAM_URL = "http://10.98.50.94:4747/video"   # phone IP changes -> update it
F_Width = 384                                    # must match the race code
F_Height = 216

# ---------------------------------------------------------------------------
# Calibration settings
# ---------------------------------------------------------------------------
CALIB_MAX_FPS = 15            # cap for the calibration thread (0 = unlimited). Saves CPU on the Pi.
STATS_HISTORY = 20000         # max stored width/offset samples
MIN_SAMPLES_WIDTH = 30        # frames with BOTH lines needed before trusting the width
MIN_STREAK_EVENTS = 3         # lost-line episodes needed before suggesting LANE_HOLD_FRAMES
HOLD_SAFETY_FACTOR = 1.5      # suggested hold = longest observed gap * this factor
WIDTH_STD_WARN_PX = 12        # width std above this -> measurement is unstable
OFFSET_WARN_PX = 6            # |center offset| above this -> camera/robot not centered
CALIB_IDLE_TIMEOUT = 3.0      # seconds without a browser poll -> calibration thread goes idle

REJECT_COLOR = (0, 215, 255)  # amber: Hough lines that were rejected (slope / wrong side)

# key: (default, min, max, step, group)
# Names match the constants of the race code, so values can be copied 1:1.
PARAM_SPEC = {
    "CANNY_LOW":           (60, 0, 255, 1, "Edges (Canny)"),
    "CANNY_HIGH":          (160, 0, 255, 1, "Edges (Canny)"),
    "HOUGH_THRESHOLD":     (50, 1, 200, 1, "Hough lines"),
    "MIN_LINE_LENGTH":     (30, 1, 200, 1, "Hough lines"),
    "MAX_LINE_GAP":        (60, 0, 200, 1, "Hough lines"),
    "RIGHT_MIN_ABS_SLOPE": (0.4, 0.0, 3.0, 0.05, "Slope filter"),
    "LEFT_MIN_ABS_SLOPE":  (0.3, 0.0, 3.0, 0.05, "Slope filter"),
    "ROI_TOP_RL":          (0.65, 0.0, 1.0, 0.01, "Right lane ROI"),
    "ROI_BOTTOM_RL":       (1.0, 0.0, 1.0, 0.01, "Right lane ROI"),
    "ROI_LEFT_RL":         (0.65, 0.0, 1.0, 0.01, "Right lane ROI"),
    "ROI_RIGHT_RL":        (0.9, 0.0, 1.0, 0.01, "Right lane ROI"),
    "ROI_TOP_LL":          (0.65, 0.0, 1.0, 0.01, "Left lane ROI"),
    "ROI_BOTTOM_LL":       (1.0, 0.0, 1.0, 0.01, "Left lane ROI"),
    "ROI_LEFT_LL":         (0.0, 0.0, 1.0, 0.01, "Left lane ROI"),
    "ROI_RIGHT_LL":        (0.4, 0.0, 1.0, 0.01, "Left lane ROI"),
    "GRAY_FROM_RGB":       (0, 0, 1, 1, "Input (1 = frames are RGB, like Picamera2 in the race code)"),
}


class SharedState:
    """Data shared between the control loop (writer) and the web server (reader)."""

    def __init__(self):
        self.frames = {}       # name -> latest numpy image (BGR or gray). Publish a NEW array each frame.
        self.telemetry = {}    # latest values to display
        self.config = {}       # live reference to the vision config dict
        self.raw_frame = None  # latest raw camera frame (input of the Calibration tab)
        self.raw_id = 0        # incremented for every published raw frame
        self.standalone = False
        self._commands = queue.Queue()

    def publish_raw(self, frame) -> None:
        """Call once per loop iteration with the raw camera frame. Publish a NEW array each time."""
        self.raw_frame = frame
        self.raw_id += 1

    def push_command(self, name: str) -> None:
        self._commands.put(name)

    def pop_command(self):
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None


state = SharedState()
app = FastAPI()


def _to_jsonable(obj):
    """Fallback serializer so numpy values / anything odd never break the telemetry stream."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


# ===========================================================================
# Lane detection -- identical logic to VisionProcessor.detect_lines() of the race code
# ===========================================================================
def _roi_bounds(height, width, top, bottom, left, right):
    return int(height * top), int(height * bottom), int(width * left), int(width * right)


def _detect_side(gray, bounds, side, p):
    """Returns a dict: center_x (full-frame x or None), kept/rejected lines (full-frame coords), edges image."""
    top, bottom, left, right = bounds
    frame_center = gray.shape[1] // 2
    out = {"center_x": None, "kept": [], "rejected": [], "edges": None}

    if bottom - top < 4 or right - left < 4:      # degenerate ROI -> nothing to analyse
        out["edges"] = np.zeros((8, 8), np.uint8)
        return out

    roi = gray[top:bottom, left:right]
    edges = cv2.Canny(roi, p["CANNY_LOW"], p["CANNY_HIGH"])
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, int(p["HOUGH_THRESHOLD"]),
                            minLineLength=int(p["MIN_LINE_LENGTH"]), maxLineGap=int(p["MAX_LINE_GAP"]))
    out["edges"] = edges

    min_slope = p["RIGHT_MIN_ABS_SLOPE"] if side == "right" else p["LEFT_MIN_ABS_SLOPE"]
    roi_h = roi.shape[0]
    bottoms = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = (int(v) for v in line.flatten())
            seg = (x1 + left, y1 + top, x2 + left, y2 + top)
            if x2 == x1:                               # vertical lines are skipped, same as the race code
                out["rejected"].append(seg)
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) <= min_slope:
                out["rejected"].append(seg)
                continue
            x_bottom = x1 + (roi_h - y1) / slope if abs(slope) > 1e-5 else x1
            x_bottom = max(0, min(right - left - 1, x_bottom))
            x_bottom_full = x_bottom + left
            on_correct_side = (x_bottom_full >= frame_center) if side == "right" else (x_bottom_full <= frame_center)
            if on_correct_side:
                bottoms.append(x_bottom_full)
                out["kept"].append(seg)
            else:
                out["rejected"].append(seg)

    out["center_x"] = float(np.mean(bottoms)) if bottoms else None
    return out


def analyze_frame(frame, p):
    h, w = frame.shape[:2]
    # Gray conversion is done ONCE per frame; both ROIs are slices of this image.
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY if p["GRAY_FROM_RGB"] else cv2.COLOR_BGR2GRAY)

    rl = _roi_bounds(h, w, p["ROI_TOP_RL"], p["ROI_BOTTOM_RL"], p["ROI_LEFT_RL"], p["ROI_RIGHT_RL"])
    ll = _roi_bounds(h, w, p["ROI_TOP_LL"], p["ROI_BOTTOM_LL"], p["ROI_LEFT_LL"], p["ROI_RIGHT_LL"])
    right = _detect_side(gray, rl, "right", p)
    left = _detect_side(gray, ll, "left", p)
    return {
        "right": right, "left": left,
        "right_x": right["center_x"], "left_x": left["center_x"],
        "rl_bounds": rl, "ll_bounds": ll,
        "frame_center": w // 2,
    }


def draw_overlay(frame, res, frozen):
    vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]

    for (top, bottom, left, right), color, name in (
        (res["rl_bounds"], (0, 255, 255), "RL"),
        (res["ll_bounds"], (255, 255, 0), "LL"),
    ):
        cv2.rectangle(vis, (left, top), (right, bottom), color, 2)
        cv2.putText(vis, name, (left + 4, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    for side in (res["right"], res["left"]):
        for x1, y1, x2, y2 in side["rejected"]:
            cv2.line(vis, (x1, y1), (x2, y2), REJECT_COLOR, 1)
    for x1, y1, x2, y2 in res["right"]["kept"]:
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for x1, y1, x2, y2 in res["left"]["kept"]:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    fc = res["frame_center"]
    cv2.line(vis, (fc, 0), (fc, h), (0, 255, 0), 1)   # image center

    r, l = res["right_x"], res["left_x"]
    if r is not None:
        cv2.line(vis, (int(r), res["rl_bounds"][0]), (int(r), res["rl_bounds"][1]), (255, 0, 0), 2)
    if l is not None:
        cv2.line(vis, (int(l), res["ll_bounds"][0]), (int(l), res["ll_bounds"][1]), (0, 0, 255), 2)
    if r is not None and l is not None:
        mid = int((r + l) / 2)
        cv2.line(vis, (mid, 0), (mid, h), (0, 165, 255), 1)   # lane midpoint

    if r is not None and l is not None:
        text, color = "BOTH", (0, 255, 0)
    elif r is not None:
        text, color = "RIGHT ONLY", (0, 255, 255)
    elif l is not None:
        text, color = "LEFT ONLY", (0, 255, 255)
    else:
        text, color = "NONE", (0, 0, 255)
    cv2.putText(vis, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if r is not None and l is not None:
        cv2.putText(vis, f"width {r - l:.0f}px  off {((r + l) / 2 - fc):+.0f}px", (8, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if frozen:
        cv2.putText(vis, "FROZEN", (w - 80, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    return vis


def edge_view(side, bounds, color):
    """Edge image of one ROI with the Hough lines drawn on it (accepted = color, rejected = amber)."""
    top, bottom, left, right = bounds
    img = cv2.cvtColor(side["edges"], cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in side["rejected"]:
        cv2.line(img, (x1 - left, y1 - top), (x2 - left, y2 - top), REJECT_COLOR, 1)
    for x1, y1, x2, y2 in side["kept"]:
        cv2.line(img, (x1 - left, y1 - top), (x2 - left, y2 - top), color, 2)
    return img


# ===========================================================================
# Statistics
# ===========================================================================
class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        now = time.time()
        self.t_start = now
        self.t_paused = now
        self.running = False
        self.frames = 0          # analysed samples
        self.src_frames = 0      # camera / race-loop frames those samples stand for
        self.right_hits = 0
        self.left_hits = 0
        self.both_hits = 0
        self.widths = deque(maxlen=STATS_HISTORY)
        self.offsets = deque(maxlen=STATS_HISTORY)
        self.streaks = {"left": [], "right": []}
        self.cur_streak = {"left": 0, "right": 0}

    def set_running(self, flag):
        """Pause the clock while nothing is analysed (tab closed / frozen) so FPS stays correct."""
        now = time.time()
        if flag and not self.running:
            self.t_start += now - self.t_paused
            self.cur_streak = {"left": 0, "right": 0}
            self.running = True
        elif not flag and self.running:
            self.t_paused = now
            self.running = False

    def update(self, res, n=1):
        """n = how many source frames this sample stands for (>1 when the thread skipped frames)."""
        r = res["right_x"] is not None
        l = res["left_x"] is not None
        self.frames += 1
        self.src_frames += n
        self.right_hits += r
        self.left_hits += l

        if r and l:
            self.both_hits += 1
            self.widths.append(res["right_x"] - res["left_x"])
            self.offsets.append((res["right_x"] + res["left_x"]) / 2 - res["frame_center"])

        # A side counts as "lost" only while the OTHER side is still visible
        # (so the camera is still on the track). If both vanish, the episode is dropped.
        for side, mine, other in (("left", l, r), ("right", r, l)):
            if mine:
                if self.cur_streak[side] > 0:
                    self.streaks[side].append(self.cur_streak[side])
                self.cur_streak[side] = 0
            elif other:
                self.cur_streak[side] += n
            else:
                self.cur_streak[side] = 0

    @property
    def fps(self):
        end = time.time() if self.running else self.t_paused
        dt = end - self.t_start
        return self.src_frames / dt if dt > 0.2 else 0.0

    def suggest(self):
        out = {"width": None, "width_std": None, "width_mean": None, "width_min": None, "width_max": None,
               "offset": None, "hold": None, "longest": None, "gap_events": 0}
        if len(self.widths) >= MIN_SAMPLES_WIDTH:
            w = np.array(self.widths)
            out["width"] = float(np.median(w))
            out["width_std"] = float(np.std(w))
            out["width_mean"] = float(w.mean())
            out["width_min"] = float(w.min())
            out["width_max"] = float(w.max())
            out["offset"] = float(np.median(self.offsets))
        gaps = self.streaks["left"] + self.streaks["right"] + [v for v in self.cur_streak.values() if v > 0]
        out["gap_events"] = len(gaps)
        if len(gaps) >= MIN_STREAK_EVENTS:
            out["longest"] = max(gaps)
            out["hold"] = max(3, int(math.ceil(out["longest"] * HOLD_SAFETY_FACTOR)))
        return out

    def summary(self):
        n = max(self.frames, 1)
        s = self.suggest()
        out = []
        out.append(f"{self.src_frames} frames, {self.fps:.1f} FPS  ({self.frames} analysed)")
        out.append(f"Right line detected : {100 * self.right_hits / n:5.1f} %")
        out.append(f"Left  line detected : {100 * self.left_hits / n:5.1f} %")
        out.append(f"Both detected       : {100 * self.both_hits / n:5.1f} %  ({self.both_hits} frames)")
        out.append("")

        if s["width"] is not None:
            out.append(f"Lane width (px)     : median {s['width']:.1f} | mean {s['width_mean']:.1f} | "
                       f"std {s['width_std']:.1f} | min {s['width_min']:.0f} | max {s['width_max']:.0f}")
            out.append(f"  -> LANE_WIDTH_INIT_PX = {int(round(s['width']))}")
            if s["width_std"] > WIDTH_STD_WARN_PX:
                out.append(f"  !! std > {WIDTH_STD_WARN_PX}px: width is unstable. Keep the robot straight and "
                           "still, or reset and re-measure.")
            off = s["offset"]
            out.append(f"Center offset (px)  : median {off:+.1f}   (lane midpoint - image center)")
            if abs(off) > OFFSET_WARN_PX:
                side = "RIGHT" if off > 0 else "LEFT"
                out.append(f"  !! Offset is large. If the robot is truly centered between the lines, the camera "
                           f"is shifted/rotated toward the {side}; the robot will drift unless you compensate "
                           "(shift target_x by this offset, or re-mount the camera).")
            else:
                out.append("  -> Camera looks well centered (offset is small).")
        else:
            out.append(f"Lane width          : not enough data ({len(self.widths)}/{MIN_SAMPLES_WIDTH} frames "
                       "with both lines). Put BOTH lines inside their ROI boxes and hold still.")
        out.append("")

        out.append(f"Lost-line episodes  : left {len(self.streaks['left'])} "
                   f"(max {max(self.streaks['left'], default=0)} frames) | "
                   f"right {len(self.streaks['right'])} (max {max(self.streaks['right'], default=0)} frames)")
        if s["hold"] is not None:
            fps = self.fps
            secs = f"{s['longest'] / fps:.2f} s at {fps:.1f} FPS" if fps > 0 else "? s"
            out.append(f"  longest gap = {s['longest']} frames (~{secs})")
            out.append(f"  -> LANE_HOLD_FRAMES = {s['hold']}   (longest gap x {HOLD_SAFETY_FACTOR})")
            out.append("  NOTE: valid only if you moved along the track at about the real robot speed.")
        else:
            out.append(f"  not enough gaps seen ({s['gap_events']}/{MIN_STREAK_EVENTS}). Move the camera along "
                       "the dashed line (dash -> gap -> dash) while the solid right line stays visible.")
        return "\n".join(out)


# ===========================================================================
# Calibration worker (own thread, sleeps unless the Calibration tab is being watched)
# ===========================================================================
def _num(v):
    if isinstance(v, float) and not v.is_integer():
        return f"{v:g}"
    return str(int(v))


class CalibWorker:
    def __init__(self, shared):
        self.shared = shared
        self.lock = threading.Lock()
        self.params = {k: spec[0] for k, spec in PARAM_SPEC.items()}
        self.version = 0                 # incremented on every parameter change
        self.stats = Stats()
        self.frames = {}                 # name -> latest image, streamed by /calib/stream/{name}
        self.live = {}
        self.frozen_frame = None
        self.last_seen = 0.0
        self.synced = False
        self._thread = None

    # ---- life cycle -------------------------------------------------------
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def touch(self):
        self.last_seen = time.time()

    def active(self):
        return (time.time() - self.last_seen) < CALIB_IDLE_TIMEOUT

    # ---- parameters -------------------------------------------------------
    def params_json(self):
        with self.lock:
            return {k: {"value": self.params[k], "min": spec[1], "max": spec[2],
                        "step": spec[3], "group": spec[4]} for k, spec in PARAM_SPEC.items()}

    def set_params(self, update):
        with self.lock:
            for k, v in update.items():
                if k not in PARAM_SPEC:
                    continue
                default, lo, hi, _step, _group = PARAM_SPEC[k]
                try:
                    v = type(default)(v)
                except (TypeError, ValueError):
                    continue
                self.params[k] = max(lo, min(hi, v))
            self.version += 1

    def load_from_robot(self, config):
        loaded = []
        with self.lock:
            for k, spec in PARAM_SPEC.items():
                if k in config:
                    try:
                        self.params[k] = max(spec[1], min(spec[2], type(spec[0])(config[k])))
                        loaded.append(k)
                    except (TypeError, ValueError):
                        pass
            self.version += 1
        return loaded

    def sync_once(self, config):
        if not self.synced:
            self.synced = True
            self.load_from_robot(config)

    def apply_to_robot(self, config):
        applied, skipped = [], []
        with self.lock:
            items = dict(self.params)
        for k, v in items.items():
            if k in config:
                try:
                    config[k] = type(config[k])(v)
                    applied.append(k)
                except (TypeError, ValueError):
                    skipped.append(k)
            else:
                skipped.append(k)
        return applied, skipped

    # ---- freeze / reset ---------------------------------------------------
    def set_frozen(self, on):
        if on:
            f = self.shared.raw_frame
            if f is not None:
                self.frozen_frame = f.copy()
        else:
            self.frozen_frame = None

    def reset_stats(self):
        with self.lock:
            self.stats.reset()

    def current_raw(self):
        return self.frozen_frame if self.frozen_frame is not None else self.shared.raw_frame

    # ---- status for the browser ------------------------------------------
    def status(self):
        self.touch()          # every poll doubles as a heartbeat
        with self.lock:
            st = self.stats
            n = max(st.frames, 1)
            return {
                "has_frame": self.shared.raw_frame is not None,
                "frozen": self.frozen_frame is not None,
                "live": dict(self.live),
                "stats": {
                    "samples": st.frames,
                    "fps": round(st.fps, 1),
                    "right_pct": round(100 * st.right_hits / n, 1),
                    "left_pct": round(100 * st.left_hits / n, 1),
                    "both_pct": round(100 * st.both_hits / n, 1),
                },
                "summary": st.summary(),
                "constants": self._constants_text(st.suggest()),
            }

    def _constants_text(self, sug):
        lines = []
        group = None
        for k, spec in PARAM_SPEC.items():
            if k == "GRAY_FROM_RGB":
                continue
            if spec[4] != group:
                if group is not None:
                    lines.append("")
                group = spec[4]
                lines.append(f"# {group}")
            lines.append(f"{k} = {_num(self.params[k])}")
        measured = []
        if sug["width"] is not None:
            measured.append(f"LANE_WIDTH_INIT_PX = {int(round(sug['width']))}")
        if sug["hold"] is not None:
            measured.append(f"LANE_HOLD_FRAMES = {sug['hold']}")
        if measured:
            lines.append("")
            lines.append("# Measured")
            lines.extend(measured)
        return "\n".join(lines)

    # ---- worker thread ----------------------------------------------------
    def _run(self):
        last_src_id = None
        last_key = None
        period = 1.0 / CALIB_MAX_FPS if CALIB_MAX_FPS > 0 else 0.0
        while True:
            frozen = self.frozen_frame
            active = self.active()
            with self.lock:
                self.stats.set_running(active and frozen is None)

            if not active:
                last_src_id = None
                last_key = None
                time.sleep(0.1)
                continue

            if frozen is not None:
                # Re-analyse the same still image whenever a parameter changes.
                key = (id(frozen), self.version)
                if key != last_key:
                    last_key = key
                    self._analyze(frozen, 1, update_stats=False)
                last_src_id = None
                time.sleep(0.03)
                continue

            last_key = None
            frame, src_id = self.shared.raw_frame, self.shared.raw_id
            if frame is None or src_id == last_src_id:
                time.sleep(0.005)
                continue

            t0 = time.time()
            n = 1 if last_src_id is None else max(1, min(10, src_id - last_src_id))
            last_src_id = src_id
            self._analyze(frame, n, update_stats=True)
            rest = period - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)

    def _analyze(self, frame, n, update_stats):
        try:
            with self.lock:
                p = dict(self.params)
            res = analyze_frame(frame, p)
            frozen = self.frozen_frame is not None

            self.frames["overlay"] = draw_overlay(frame, res, frozen)
            self.frames["edges_rl"] = edge_view(res["right"], res["rl_bounds"], (255, 0, 0))
            self.frames["edges_ll"] = edge_view(res["left"], res["ll_bounds"], (0, 0, 255))

            r, l = res["right_x"], res["left_x"]
            both = r is not None and l is not None
            live = {
                "state": "BOTH" if both else "RIGHT ONLY" if r is not None else "LEFT ONLY" if l is not None else "NONE",
                "right_x": round(r, 1) if r is not None else None,
                "left_x": round(l, 1) if l is not None else None,
                "width": round(r - l, 1) if both else None,
                "offset": round((r + l) / 2 - res["frame_center"], 1) if both else None,
                "right_kept": len(res["right"]["kept"]),
                "right_all": len(res["right"]["kept"]) + len(res["right"]["rejected"]),
                "left_kept": len(res["left"]["kept"]),
                "left_all": len(res["left"]["kept"]) + len(res["left"]["rejected"]),
            }
            with self.lock:
                self.live = live
                if update_stats:
                    self.stats.update(res, n)
        except Exception as e:
            print("[calib] analysis error:", e)
            time.sleep(0.2)


calib = CalibWorker(state)


# ===========================================================================
# Video: MJPEG streams
#   /stream/<name>        frames published by the control loop (state.frames)
#   /calib/stream/<name>  frames produced by the calibration tab
# ===========================================================================
async def _mjpeg(request: Request, get_frames, name: str):
    last = None
    delay = 1.0 / STREAM_FPS_LIMIT
    while not await request.is_disconnected():
        frame = get_frames().get(name)
        if frame is None or frame is last:       # nothing new yet
            await asyncio.sleep(0.02)
            continue
        last = frame
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        await asyncio.sleep(delay)


@app.get("/stream/{name}")
async def stream(request: Request, name: str):
    return StreamingResponse(
        _mjpeg(request, lambda: state.frames, name),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/calib/stream/{name}")
async def calib_stream(request: Request, name: str):
    return StreamingResponse(
        _mjpeg(request, lambda: calib.frames, name),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/streams")
async def list_streams():
    return list(state.frames.keys())


@app.get("/api/info")
async def info():
    return {"standalone": state.standalone}


# ----------------------------------------------------------------------------
# Telemetry: pushed over a WebSocket
# ----------------------------------------------------------------------------
@app.websocket("/ws")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_text(json.dumps(state.telemetry, default=_to_jsonable))
            await asyncio.sleep(1.0 / TELEMETRY_HZ)
    except Exception:
        pass  # client closed the page - nothing to clean up


# ----------------------------------------------------------------------------
# Live tuning + commands (robot config)
# ----------------------------------------------------------------------------
@app.get("/api/config")
async def get_config():
    return state.config


@app.post("/api/config")
async def set_config(update: dict):
    for key, value in update.items():
        if key in state.config:
            try:
                state.config[key] = type(state.config[key])(value)   # keep original type (int/float)
            except (TypeError, ValueError):
                pass
    return state.config


@app.post("/api/cmd/{name}")
async def command(name: str):
    if name not in ("start", "stop"):
        return {"ok": False}
    state.push_command(name)
    return {"ok": True}


# ----------------------------------------------------------------------------
# Calibration API
# ----------------------------------------------------------------------------
@app.get("/api/calib/params")
async def calib_get_params():
    calib.sync_once(state.config)        # first call: start from the robot's current values
    return calib.params_json()


@app.post("/api/calib/params")
async def calib_set_params(update: dict):
    calib.set_params(update)
    return calib.params_json()


@app.post("/api/calib/load")
async def calib_load():
    loaded = calib.load_from_robot(state.config)
    return {"loaded": loaded}


@app.post("/api/calib/apply")
async def calib_apply():
    applied, skipped = calib.apply_to_robot(state.config)
    return {"applied": applied, "skipped": skipped}


@app.post("/api/calib/reset")
async def calib_reset():
    calib.reset_stats()
    return {"ok": True}


@app.post("/api/calib/freeze")
async def calib_freeze(body: dict):
    calib.set_frozen(bool(body.get("on")))
    return {"frozen": calib.frozen_frame is not None}


@app.get("/api/calib/status")
async def calib_status():
    return calib.status()


def _png_response(img, filename):
    if img is None:
        return Response(status_code=404)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return Response(status_code=500)
    return Response(content=buf.tobytes(), media_type="image/png",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/calib/snapshot.png")
async def calib_snapshot():
    return _png_response(calib.frames.get("overlay"), "calib_snapshot.png")


@app.get("/calib/raw.png")
async def calib_raw():
    return _png_response(calib.current_raw(), "calib_raw.png")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


def start(host: str = "0.0.0.0", port: int = 8000):
    """Start the web server (and the calibration thread) in daemon threads and return immediately."""
    calib.start()
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return server


# ===========================================================================
# Standalone camera source (only used by `python dashboard.py`)
# ===========================================================================
def _init_camera():
    if USE_DROIDCAM:
        cap = cv2.VideoCapture(DROIDCAM_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open DroidCam stream at {DROIDCAM_URL} -- check the phone IP, "
                "that both devices are on the same Wi-Fi, and that DroidCam is running."
            )
        print(f"[INFO] Using DroidCam: {DROIDCAM_URL}")
        return "cv2", cap

    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (F_Width, F_Height), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1)
        print("[INFO] Using Picamera2")
        return "picam2", picam2
    except ImportError:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, F_Width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, F_Height)
        print("[INFO] Using cv2.VideoCapture(0)")
        return "cv2", cap


def _capture(cam_type, cam):
    if cam_type == "picam2":
        frame = cam.capture_array()
    else:
        ret, frame = cam.read()
        if not ret:
            return None
    return cv2.resize(frame, (F_Width, F_Height), interpolation=cv2.INTER_AREA)


def start_camera():
    """Open the camera and publish its frames into state (standalone calibration mode)."""
    cam_type, cam = _init_camera()
    if cam_type == "picam2":
        calib.set_params({"GRAY_FROM_RGB": 1})   # the race code converts Picamera2 frames with RGB2GRAY

    def loop():
        while True:
            frame = _capture(cam_type, cam)
            if frame is None:
                time.sleep(0.05)
                continue
            state.publish_raw(frame)

    threading.Thread(target=loop, daemon=True).start()


# ===========================================================================
# Front-end (single page, no build step, no external files)
# ===========================================================================
HTML_PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot dashboard</title>
<style>
  body { font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 12px; }
  h3 { margin: 0 0 6px; font-size: 14px; color: #9ab; }
  .card { background: #1c1c1c; border-radius: 8px; padding: 8px; }

  /* tabs */
  #tabs { display: flex; gap: 6px; margin-bottom: 10px; }
  .btn { font-size: 14px; padding: 6px 14px; border: 0; border-radius: 6px; color: #fff; background: #345;
         cursor: pointer; text-decoration: none; display: inline-block; }
  .btn.tab { background: #262626; font-size: 15px; padding: 8px 20px; }
  .btn.tab.active { background: #36a; }
  .btn.warn { background: #a62; }
  .btn.on { background: #a3a; }

  /* left: controls + video / right: telemetry + config */
  .layout { display: flex; gap: 12px; align-items: flex-start; }
  .main { flex: 1 1 auto; min-width: 0; }
  .side { flex: 0 0 320px; position: sticky; top: 12px; max-height: calc(100vh - 24px);
          overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
  .side.wide { flex-basis: 400px; }

  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 10px; }
  .streams { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start; }
  .streams .card img { display: block; width: 768px; max-width: 100%; image-rendering: pixelated; }
  .streams .card img.small { width: 376px; }

  table { border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 13px; }
  td { padding: 2px 10px 2px 0; }
  td:first-child { color: #8c8; }
  .row { display: flex; justify-content: space-between; gap: 10px; margin: 2px 0; font-size: 13px; }
  .row input { width: 80px; background: #222; color: #eee; border: 1px solid #444; }

  button { font-size: 16px; padding: 8px 18px; border: 0; border-radius: 6px; color: #fff; cursor: pointer; }
  #start { background: #2a7; } #stop { background: #c33; }
  #speedbox { display: none; align-items: center; gap: 8px; margin-left: 12px; padding: 4px 10px;
              background: #1c1c1c; border-radius: 6px; font-size: 14px; }
  #speedbox button { background: #345; padding: 4px 14px; }
  #speedVal { min-width: 3ch; text-align: center; font-family: ui-monospace, monospace; font-size: 16px; }

  /* calibration */
  .prow { display: grid; grid-template-columns: 150px 1fr 64px; gap: 6px; align-items: center;
          font-size: 12px; margin: 3px 0; font-family: ui-monospace, monospace; }
  .prow input[type=number] { width: 100%; box-sizing: border-box; background: #222; color: #eee; border: 1px solid #444; }
  .prow input[type=range] { width: 100%; }
  #csummary { margin: 0; font-size: 12px; white-space: pre-wrap; font-family: ui-monospace, monospace; }
  #cconst { width: 100%; box-sizing: border-box; height: 230px; background: #222; color: #cde; border: 1px solid #444;
            font-family: ui-monospace, monospace; font-size: 12px; margin-bottom: 6px; }
  #cnoframe { background: #5a3a10; border-radius: 6px; padding: 8px; margin-bottom: 10px; font-size: 13px; }
  #cmsg { font-size: 13px; color: #9c9; }
  .legend { font-size: 12px; color: #9ab; margin-bottom: 8px; }

  @media (max-width: 900px) {
    .layout { flex-direction: column; align-items: stretch; }
    .side, .side.wide { flex: none; position: static; max-height: none; }
  }
</style>
</head>
<body>
<div id="tabs">
  <button class="btn tab active" data-tab="live">Live</button>
  <button class="btn tab" data-tab="calib">Calibration</button>
</div>

<!-- ===================== LIVE ===================== -->
<div id="view-live">
  <div class="layout">
    <div class="main">
      <div class="controls">
        <button id="start">START</button>
        <button id="stop">STOP</button>
        <div id="speedbox">
          Max speed
          <button id="spdDn">-</button>
          <span id="speedVal">-</span>
          <button id="spdUp">+</button>
        </div>
      </div>
      <div class="streams" id="streams"></div>
    </div>

    <div class="side">
      <div class="card"><h3>Telemetry</h3><table id="telemetry"></table></div>
      <div class="card"><h3>Config (live)</h3><div id="config"></div></div>
    </div>
  </div>
</div>

<!-- ===================== CALIBRATION ===================== -->
<div id="view-calib" style="display:none">
  <div class="layout">
    <div class="main">
      <div class="controls">
        <button class="btn" id="cfreeze">Freeze frame</button>
        <button class="btn" id="creset">Reset stats</button>
        <button class="btn" id="cload">Load from robot</button>
        <button class="btn warn" id="capply">Apply to robot</button>
        <a class="btn" href="/calib/snapshot.png">Save overlay PNG</a>
        <a class="btn" href="/calib/raw.png">Save raw PNG</a>
        <span id="cmsg"></span>
      </div>
      <div id="cnoframe" style="display:none">
        No camera frames yet. In the race loop call <code>dashboard.state.publish_raw(frame)</code>
        every iteration, or stop the race code and run <code>python dashboard.py</code> for standalone calibration.
      </div>
      <div class="legend">
        Overlay: blue = accepted right lines, red = accepted left lines, amber = rejected lines (slope or wrong side),
        green = image centre, orange = lane midpoint. Use <b>Freeze frame</b> to tune the parameters on one still image.
      </div>
      <div class="streams">
        <div class="card"><h3>overlay</h3><img data-group="calib" data-src="/calib/stream/overlay"></div>
        <div class="card"><h3>edges - right ROI</h3><img class="small" data-group="calib" data-src="/calib/stream/edges_rl"></div>
        <div class="card"><h3>edges - left ROI</h3><img class="small" data-group="calib" data-src="/calib/stream/edges_ll"></div>
      </div>
      <div class="card" style="margin-top:10px"><h3>Measured values</h3><pre id="csummary"></pre></div>
    </div>

    <div class="side wide">
      <div class="card"><h3>Live detection</h3><table id="clive"></table></div>
      <div class="card">
        <h3>Parameters (separate copy - the robot is not affected until you press "Apply to robot")</h3>
        <div id="cparams"></div>
      </div>
      <div class="card">
        <h3>Copy into race code</h3>
        <textarea id="cconst" readonly></textarea>
        <button class="btn" id="ccopy">Copy</button>
      </div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const fmt = v => typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : JSON.stringify(v);
const val = v => (v === null || v === undefined) ? '-' : fmt(v);
const post = (url, body) => fetch(url, {
  method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})
});
const BLANK = 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==';

// ------------------------------------------------------------------ tabs
let tab = 'live';

function showTab(name) {
  tab = name;
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  $('view-live').style.display = name === 'live' ? '' : 'none';
  $('view-calib').style.display = name === 'calib' ? '' : 'none';
  location.hash = name;
  syncStreams();
  if (name === 'calib') { loadCalibParams(); startCalibPolling(); } else { stopCalibPolling(); }
}
document.querySelectorAll('.tab').forEach(b => b.onclick = () => showTab(b.dataset.tab));

// Only the streams of the visible tab are connected, so the Pi never encodes JPEGs nobody looks at.
function syncStreams() {
  document.querySelectorAll('img[data-src]').forEach(img => {
    const on = img.dataset.group === tab && !document.hidden;
    if (on) {
      if (!img.dataset.on) { img.src = img.dataset.src + '?t=' + Date.now(); img.dataset.on = '1'; }
    } else if (img.dataset.on) {
      img.src = BLANK; img.dataset.on = '';
    }
  });
}
document.addEventListener('visibilitychange', () => {
  syncStreams();
  if (tab === 'calib') { document.hidden ? stopCalibPolling() : startCalibPolling(); }
});

// ------------------------------------------------------------------ LIVE tab
$('start').onclick = () => post('/api/cmd/start');
$('stop').onclick  = () => post('/api/cmd/stop');

// video streams: a card is added automatically for every frame name the robot publishes
const known = new Set();
async function refreshStreams() {
  const names = await (await fetch('/api/streams')).json();
  for (const n of names) {
    if (known.has(n)) continue;
    known.add(n);
    const d = document.createElement('div');
    d.className = 'card';
    d.innerHTML = `<h3>${n}</h3><img data-group="live" data-src="/stream/${n}">`;
    $('streams').appendChild(d);
  }
  syncStreams();
}
setInterval(refreshStreams, 2000);
refreshStreams();

// telemetry over WebSocket (auto-reconnect)
function connectWS() {
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.onmessage = e => {
    const t = JSON.parse(e.data);
    $('telemetry').innerHTML = Object.entries(t)
      .map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`).join('');
  };
  ws.onclose = () => setTimeout(connectWS, 1000);
}
connectWS();

// speed buttons (they edit SPEED_MAX in the live config; hidden if the robot has no such key)
const SPEED_KEY = 'SPEED_MAX', SPEED_STEP = 20, SPEED_LO = 100, SPEED_HI = 500;

function showSpeed(cfg) {
  const has = SPEED_KEY in cfg;
  $('speedbox').style.display = has ? 'flex' : 'none';
  if (has) $('speedVal').textContent = cfg[SPEED_KEY];
}

async function nudgeSpeed(delta) {
  const cfg = await (await fetch('/api/config')).json();
  if (!(SPEED_KEY in cfg)) return;
  const v = Math.min(SPEED_HI, Math.max(SPEED_LO, cfg[SPEED_KEY] + delta));
  await post('/api/config', {[SPEED_KEY]: v});
  loadConfig();
}
$('spdUp').onclick = () => nudgeSpeed(SPEED_STEP);
$('spdDn').onclick = () => nudgeSpeed(-SPEED_STEP);

// live config editor (ROI, Canny, Hough, speed, ...)
async function loadConfig() {
  const cfg = await (await fetch('/api/config')).json();
  showSpeed(cfg);
  $('config').innerHTML = '';
  for (const [k, v] of Object.entries(cfg)) {
    const step = (Number.isInteger(v) && Math.abs(v) > 1) ? 1 : 0.05;
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = `<label>${k}</label><input type="number" step="${step}" value="${v}">`;
    row.querySelector('input').onchange = async e => {
      const r = await post('/api/config', {[k]: Number(e.target.value)});
      showSpeed(await r.json());
    };
    $('config').appendChild(row);
  }
}
loadConfig();

// ------------------------------------------------------------------ CALIBRATION tab
let pollTimer = null, pollBusy = false;

function startCalibPolling() {
  stopCalibPolling();
  pollCalib();
  pollTimer = setInterval(pollCalib, 250);   // every poll is also the heartbeat that keeps the worker awake
}
function stopCalibPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}
async function pollCalib() {
  if (pollBusy) return;
  pollBusy = true;
  try { renderCalib(await (await fetch('/api/calib/status')).json()); } catch (e) {}
  pollBusy = false;
}

function setText(el, txt) { if (el.textContent !== txt) el.textContent = txt; }

function renderCalib(s) {
  $('cnoframe').style.display = s.has_frame ? 'none' : 'block';
  $('cfreeze').textContent = s.frozen ? 'Unfreeze' : 'Freeze frame';
  $('cfreeze').classList.toggle('on', s.frozen);

  const L = s.live || {}, S = s.stats;
  const color = {'BOTH': '#4d4', 'RIGHT ONLY': '#ec4', 'LEFT ONLY': '#ec4', 'NONE': '#f55'}[L.state] || '#eee';
  const rows = [
    ['lane', `<span style="color:${color}">${L.state || '-'}</span>`],
    ['right x', val(L.right_x)], ['left x', val(L.left_x)],
    ['width px', val(L.width)], ['offset px', val(L.offset)],
    ['right lines kept/all', `${val(L.right_kept)} / ${val(L.right_all)}`],
    ['left lines kept/all', `${val(L.left_kept)} / ${val(L.left_all)}`],
    ['source fps', val(S.fps)], ['samples', val(S.samples)],
    ['right found %', val(S.right_pct)], ['left found %', val(S.left_pct)], ['both found %', val(S.both_pct)],
  ];
  $('clive').innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');

  setText($('csummary'), s.summary);
  if ($('cconst').value !== s.constants) $('cconst').value = s.constants;   // keeps your text selection
}

// parameter sliders (changes go to the calibration thread immediately; the robot is untouched)
const pending = {};
let sendTimer = null;
function queueParam(k, v) {
  pending[k] = Number(v);
  if (!sendTimer) sendTimer = setTimeout(flushParams, 80);
}
async function flushParams() {
  sendTimer = null;
  const body = {...pending};
  for (const k in pending) delete pending[k];
  await post('/api/calib/params', body);
}

async function loadCalibParams() {
  const params = await (await fetch('/api/calib/params')).json();
  const box = $('cparams');
  box.innerHTML = '';
  let group = null;
  for (const [k, p] of Object.entries(params)) {
    if (p.group !== group) {
      group = p.group;
      const h = document.createElement('h3');
      h.textContent = group;
      h.style.marginTop = '10px';
      box.appendChild(h);
    }
    const row = document.createElement('div');
    row.className = 'prow';
    row.innerHTML = `<label>${k}</label>
      <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.value}">
      <input type="number" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.value}">`;
    const [rng, num] = row.querySelectorAll('input');
    rng.oninput = () => { num.value = rng.value; queueParam(k, rng.value); };
    num.onchange = () => { rng.value = num.value; queueParam(k, num.value); };
    box.appendChild(row);
  }
}

function flash(msg) {
  $('cmsg').textContent = msg;
  setTimeout(() => { if ($('cmsg').textContent === msg) $('cmsg').textContent = ''; }, 8000);
}

$('cfreeze').onclick = async () => {
  const frozen = $('cfreeze').classList.contains('on');
  await post('/api/calib/freeze', {on: !frozen});
  pollCalib();
};
$('creset').onclick = async () => { await post('/api/calib/reset'); pollCalib(); };
$('cload').onclick = async () => {
  const r = await (await post('/api/calib/load')).json();
  await loadCalibParams();
  flash(r.loaded.length ? `Loaded ${r.loaded.length} values from the robot config` : 'Robot config has no matching keys');
};
$('capply').onclick = async () => {
  if (!confirm('Write these values into the LIVE robot config?')) return;
  const r = await (await post('/api/calib/apply')).json();
  flash(`Applied ${r.applied.length} values to the robot` +
        (r.skipped.length ? `. Not in the robot config (copy by hand): ${r.skipped.join(', ')}` : ''));
};
$('ccopy').onclick = () => {
  const ta = $('cconst');
  ta.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch (e) {}
  if (!ok && navigator.clipboard) { navigator.clipboard.writeText(ta.value); ok = true; }
  flash(ok ? 'Copied' : 'Select the text and press Ctrl+C');
};

// ------------------------------------------------------------------ start
(async () => {
  let standalone = false;
  try { standalone = (await (await fetch('/api/info')).json()).standalone; } catch (e) {}
  showTab(standalone || location.hash === '#calib' ? 'calib' : 'live');
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    state.standalone = True
    try:
        start_camera()
    except Exception as e:      # the page still comes up, it just shows "No camera frames yet"
        print("[WARN] camera not available:", e)
    start(port=8000)
    print("Standalone calibration running. Open http://localhost:8000  (Ctrl+C to quit)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped")
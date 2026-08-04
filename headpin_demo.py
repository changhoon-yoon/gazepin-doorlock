# HeadPIN 4-way door-unlock demo
# Base-4 GazePIN protocol driven by head turns (LEFT/RIGHT/UP/DOWN).
# - Real 0-9 PIN (10^4 keyspace), 2 rounds/digit -> 8 gestures per 4-digit PIN.
# - Digits sit on a FIXED phone-style keypad; only each tile's direction badge
#   (arrow + tint) re-randomizes every round. Partitions are generated from
#   public state only - the PIN is never an input to layout generation.
# - LONG BLINK (~0.5s) cancels the CURRENT DIGIT's rounds and restarts that
#   digit with fresh partitions (during final verification it cancels the last
#   digit instead). Only 10 of the 16 possible answer pairs are valid, so many
#   input errors are self-detected and the digit auto-restarts.
#
# Usage: python headpin_demo.py [--pin 1234] [--source 0]
# Keys : c = recalibrate, r = restart entry, x = flip yaw sign, q = quit
import argparse
import math
import random
import time
import cv2
import numpy as np
from headtracker import HeadTracker
import lr_gaze_fast as g   # beep
import gazepin_demo as gp  # shared drawing helpers / colors / canvas size

SYMBOLS = [str(i) for i in range(10)]
DIRECTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]
ROUNDS = 2               # ceil(log4(10))
PIN_LEN = 4
DEBOUNCE_N = 3
MAX_ATTEMPTS = 3
LOCKOUT_S = 30.0
INACTIVITY_S = 25.0
FINAL_CHECK_DELAY = 1.4
LONG_BLINK_FRAMES = 15   # ~0.5s of closed eyes = cancel current digit
# presence-gated calibration: calibrate only after the visitor stands still
MIN_FACE_FRAC = 0.14     # face width >= 14% of frame width = close enough
STAND_FRAMES = 25        # ~0.8s of stable presence required
STABLE_TOL = 0.02        # bbox-center wander tolerance (fraction of frame width)
JUMP_TOL = 0.012         # per-frame movement that restarts calibration
CALIB_FRAMES = 18
FACE_LOST_RESET_S = 2.0  # visitor gone this long during entry -> session reset
# mask detection (decided ONCE per session, during the hold-still window):
# masked -> DOWN gestures are unreliable, switch to 3 directions / 3 rounds
LOWER_IDS = [0, 13, 14, 17, 152, 200]      # lips / chin landmarks
UPPER_IDS = [33, 133, 263, 362, 159, 386]  # eye-region landmarks
MASK_COLOR_DIST = 43.0    # glabella vs philtrum mean-HSV distance
# (measured: bare face 22-31, masked 56-88 -> threshold at the gap midpoint.
#  landmark jitter proved non-discriminative and is logged for info only)
MASKED_DIRS = ["LEFT", "RIGHT", "UP"]
MASKED_ROUNDS = 3         # ceil(log3(10))

DIR_COLORS = {
    "LEFT": (80, 200, 255),   # orange
    "RIGHT": (255, 200, 80),  # light blue
    "UP": (110, 220, 110),    # green
    "DOWN": (235, 120, 200),  # violet
}
DIR_GLYPH = {"LEFT": "<", "RIGHT": ">", "UP": "^", "DOWN": "v"}


def draw_arrow(canvas, d, cx, cy, s, color):
    """Filled triangle arrow — consistent size for all four directions
    (the '^' text glyph renders tiny in Hershey fonts)."""
    if d == "LEFT":
        p = [(cx - s, cy), (cx + int(s * 0.7), cy - int(s * 0.8)),
             (cx + int(s * 0.7), cy + int(s * 0.8))]
    elif d == "RIGHT":
        p = [(cx + s, cy), (cx - int(s * 0.7), cy - int(s * 0.8)),
             (cx - int(s * 0.7), cy + int(s * 0.8))]
    elif d == "UP":
        p = [(cx, cy - s), (cx - int(s * 0.8), cy + int(s * 0.7)),
             (cx + int(s * 0.8), cy + int(s * 0.7))]
    else:  # DOWN
        p = [(cx, cy + s), (cx - int(s * 0.8), cy - int(s * 0.7)),
             (cx + int(s * 0.8), cy - int(s * 0.7))]
    cv2.fillPoly(canvas, [np.array(p, np.int32)], color)

# whole-canvas background per PIN digit stage (1st..4th) so the current
# position is always obvious at a glance
STAGE_BGS = [(52, 34, 22), (24, 46, 26), (46, 28, 46), (24, 42, 54)]

KEYPAD = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], [None, "0", None]]
KEY_CX = {0: 400, 1: 480, 2: 560}
KEY_YC = [200, 263, 326, 389]


def make_partition(cells, dirs=None):
    """Assign every symbol to one of the active directions; split each cell as
    evenly as possible, balancing group sizes. Uses only public state + randomness."""
    dirs = dirs or DIRECTIONS
    groups = {d: set() for d in dirs}
    order = cells[:]
    random.shuffle(order)
    for cell in order:
        c = list(cell)
        random.shuffle(c)
        s = len(c)
        base, rem = divmod(s, len(dirs))
        sizes = [base + 1] * rem + [base] * (len(dirs) - rem)
        ordered = sorted(dirs, key=lambda d: (len(groups[d]), random.random()))
        i = 0
        for d, k in zip(ordered, sorted(sizes, reverse=True)):
            groups[d].update(c[i:i + k])
            i += k
    return groups


def split_cells(cells, groups, dirs=None):
    dirs = dirs or DIRECTIONS
    out = []
    for cell in cells:
        for d in dirs:
            part = [s for s in cell if s in groups[d]]
            if part:
                out.append(part)
    return out


class PinEntry4:
    def __init__(self, dirs=None, rounds=None):
        self.dirs = list(dirs or DIRECTIONS)
        self.rounds = rounds or ROUNDS
        self.entered = []
        self.reset_symbol()

    def reset_symbol(self):
        """(Re)start the current digit from round 1 with fresh partitions."""
        self.cells = [SYMBOLS[:]]
        self.vectors = {s: [] for s in SYMBOLS}
        self.bits = []
        self._new_round()

    def _new_round(self):
        self.groups = make_partition(self.cells, self.dirs)
        for s in SYMBOLS:
            for d in self.dirs:
                if s in self.groups[d]:
                    self.vectors[s].append(d)
                    break

    def answer(self, direction):
        """Returns 'round' | 'symbol' | 'done' | 'invalid'."""
        self.bits.append(direction)
        self.cells = split_cells(self.cells, self.groups, self.dirs)
        if len(self.bits) == self.rounds:
            matches = [s for s in SYMBOLS if self.vectors[s] == self.bits]
            if not matches:            # error self-detected: no digit fits
                self.reset_symbol()
                return "invalid"
            self.entered.append(matches[0])
            if len(self.entered) == PIN_LEN:
                return "done"
            self.reset_symbol()
            return "symbol"
        self._new_round()
        return "round"

    @property
    def symbol_idx(self):
        return len(self.entered)

    @property
    def round_idx(self):
        return len(self.bits)


def draw_board(canvas, entry, live, flash, armed, dirs=None):
    """Fixed keypad + per-round direction badges + edge command targets."""
    dirs = dirs or DIRECTIONS
    # edge targets
    tgt = {
        "LEFT": ((40, 160), (170, 430), (105, 295), 30),
        "RIGHT": ((790, 160), (920, 430), (855, 295), 30),
        "UP": ((330, 118), (630, 164), (480, 141), 17),
        "DOWN": ((330, 428), (630, 474), (480, 451), 17),
    }
    for d, (p0, p1, tp, sz) in tgt.items():
        if d not in dirs:
            continue
        fl = (flash == d)
        fill = tuple(int(c * 0.5 + f * 0.5) for c, f in zip(gp.PANEL, gp.OK_COLOR)) if fl else gp.PANEL
        cv2.rectangle(canvas, p0, p1, fill, -1)
        border = DIR_COLORS[d] if (live == d or fl) else (85, 85, 85)
        cv2.rectangle(canvas, p0, p1, border, 3 if live == d else 1)
        draw_arrow(canvas, d, tp[0], tp[1], sz, DIR_COLORS[d])
    # keypad frame doubles as the arming indicator
    cv2.rectangle(canvas, (350, 168), (610, 423), gp.OK_COLOR if armed else (85, 85, 85),
                  2 if armed else 1)
    # fixed keypad with direction badges
    for r, row in enumerate(KEYPAD):
        for cidx, s in enumerate(row):
            if s is None:
                continue
            cx, yc = KEY_CX[cidx], KEY_YC[r]
            d = entry.vectors[s][-1] if entry.vectors[s] else "LEFT"
            # uniform tiles/digits: direction is carried by the arrow color only
            cv2.rectangle(canvas, (cx - 36, yc - 27), (cx + 36, yc + 27), gp.PANEL, -1)
            cv2.rectangle(canvas, (cx - 36, yc - 27), (cx + 36, yc + 27), (95, 95, 95), 1)
            gp.put_center(canvas, s, cx - 6, yc + 12, 1.0, gp.TXT, 2, cv2.FONT_HERSHEY_DUPLEX)
            draw_arrow(canvas, d, cx + 23, yc - 12, 9, DIR_COLORS[d])


# landmark/pose-driven cartoon CAT avatar for demo videos (--avatar):
# anonymizes the face while head turns, blinks and mouth stay visible.
# Pure drawing, ~1-2ms.
def draw_avatar(img, lms, bbox, st=None):
    h, w = img.shape[:2]
    xs = [lm.x for lm in lms]
    ys = [lm.y for lm in lms]
    cx, cy = int(np.mean(xs) * w), int(np.mean(ys) * h)
    fw = max((max(xs) - min(xs)) * w, 1.0)
    R = int(fw * 0.80)

    # pixelate an enlarged face box first (hides hair/ears/chin)
    if bbox:
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        x0 = max(0, x0 - bw // 3); y0 = max(0, y0 - bh // 2)
        x1 = min(w, x1 + bw // 3); y1 = min(h, y1 + bh // 4)
        roi = img[y0:y1, x0:x1]
        if roi.size:
            small = cv2.resize(roi, (12, 12), interpolation=cv2.INTER_LINEAR)
            img[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0),
                                           interpolation=cv2.INTER_NEAREST)

    yaw = st.yaw if st else 0.0
    pitch = st.pitch if st else 0.0
    dx = int(-np.clip(yaw, -25, 25) * R * 0.016)   # feature parallax with pose
    dy = int(-np.clip(pitch, -25, 25) * R * 0.016)
    pl, pr = lms[33], lms[263]
    roll = math.atan2((pr.y - pl.y) * h, (pr.x - pl.x) * w)
    ca, sa = math.cos(roll), math.sin(roll)

    def rot(px, py, ox=0, oy=0):
        return (int(cx + ox + px * ca - py * sa), int(cy + oy + px * sa + py * ca))

    HEAD = (90, 190, 255)   # warm orange
    DARK = (55, 95, 150)
    PINK = (150, 130, 250)
    # ears
    for sx in (-1, 1):
        outer = np.array([rot(sx * R * 0.78, -R * 0.50), rot(sx * R * 0.25, -R * 0.90),
                          rot(sx * R * 0.82, -R * 1.22)])
        cv2.fillPoly(img, [outer], HEAD)
        cv2.polylines(img, [outer], True, DARK, 3)
        inner = np.array([rot(sx * R * 0.65, -R * 0.66), rot(sx * R * 0.42, -R * 0.86),
                          rot(sx * R * 0.68, -R * 1.02)])
        cv2.fillPoly(img, [inner], PINK)
    # head
    cv2.circle(img, (cx, cy), R, HEAD, -1)
    cv2.circle(img, (cx, cy), R, DARK, 3)
    # big eyes (blink-aware) with pose-following pupils
    eye_off, eye_r = R * 0.36, int(R * 0.22)
    closed = st is not None and st.blink_score > 0.35
    for sx in (-1, 1):
        ex, ey = rot(sx * eye_off, -R * 0.08, dx, dy)
        if closed:
            cv2.ellipse(img, (ex, ey), (eye_r, int(eye_r * 0.55)),
                        math.degrees(roll), 20, 160, (40, 40, 40), 4)
        else:
            cv2.circle(img, (ex, ey), eye_r, (255, 255, 255), -1)
            cv2.circle(img, (ex, ey), eye_r, DARK, 2)
            px_, py_ = ex + dx // 2, ey + dy // 2
            cv2.circle(img, (px_, py_), int(eye_r * 0.52), (45, 40, 40), -1)
            cv2.circle(img, (px_ - eye_r // 4, py_ - eye_r // 4),
                       max(2, eye_r // 5), (255, 255, 255), -1)
    # blush
    for sx in (-1, 1):
        cv2.circle(img, rot(sx * R * 0.58, R * 0.30, dx // 2, dy // 2),
                   int(R * 0.13), (170, 160, 255), -1)
    # nose + omega mouth (opens with the real mouth)
    nose = np.array([rot(-R * 0.07, R * 0.16, dx, dy), rot(R * 0.07, R * 0.16, dx, dy),
                     rot(0, R * 0.27, dx, dy)])
    cv2.fillPoly(img, [nose], PINK)
    open_amt = abs(lms[14].y - lms[13].y) * h / fw
    if open_amt > 0.06:
        mx, my_ = rot(0, R * 0.48, dx, dy)
        cv2.ellipse(img, (mx, my_), (int(R * 0.15), int(R * 0.18)),
                    math.degrees(roll), 0, 360, (60, 60, 160), -1)
    else:
        for sx in (-1, 1):
            mx, my_ = rot(sx * R * 0.10, R * 0.38, dx, dy)
            cv2.ellipse(img, (mx, my_), (int(R * 0.10), int(R * 0.08)),
                        math.degrees(roll), 20, 160, DARK, 3)
    # whiskers
    for sx in (-1, 1):
        for wy in (-0.02, 0.10):
            cv2.line(img, rot(sx * R * 0.55, R * (0.22 + wy), dx // 2, dy // 2),
                     rot(sx * R * 1.02, R * (0.16 + wy * 2), dx // 2, dy // 2), DARK, 2)


def _region_hist(img, x0, y0, x1, y1):
    roi = img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def mask_color_score(view, landmarks, face_px):
    """Mean-HSV distance between two landmark-anchored patches:
    glabella (between the eyebrows - always skin) vs philtrum (fabric when
    masked). Mean color is robust where tiny-patch histograms are pure noise.
    Returns (distance, up_hsv, lo_hsv) or None."""
    h, w = view.shape[:2]
    r = max(8, int(face_px * 0.12))

    def patch_mean(idx):
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        roi = view[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        if roi.size == 0:
            return None
        return cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(axis=0)

    up = patch_mean(168)   # glabella
    lo = patch_mean(164)   # philtrum
    if up is None or lo is None:
        return None
    dh = min(abs(up[0] - lo[0]), 180 - abs(up[0] - lo[0])) * 2.0  # circular hue
    ds = abs(up[1] - lo[1])
    dv = abs(up[2] - lo[2])
    return float(dh + ds + 0.5 * dv), up, lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default="1234")
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--model", default="models/face_landmarker.task")
    ap.add_argument("--avatar", action="store_true",
                    help="anonymize the camera view with a landmark-driven cartoon face")
    ap.add_argument("--mask-detect", action="store_true",
                    help="EXPERIMENTAL: auto-switch to 3-way mode when a mask is detected "
                         "(current color heuristic is lighting-sensitive, off by default)")
    args = ap.parse_args()
    pin = args.pin
    if len(pin) != PIN_LEN or any(ch not in SYMBOLS for ch in pin):
        raise SystemExit(f"PIN must be {PIN_LEN} digits 0-9, got: {pin}")
    print(f"[HeadPIN-4way] demo PIN = {pin} (console only, never shown on screen)")

    # up/down thresholds raised well above the natural head tilt of reading the
    # on-screen keypad (session log showed DOWN false-firing while reading)
    tracker = HeadTracker(args.model, thr_deg=12.0, up_thr_deg=13.0, down_thr_deg=17.0)
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    from collections import deque
    W, H = gp.CANVAS_W, gp.CANVAS_H
    app = "WAIT"
    calib_frames = 0
    centers = deque(maxlen=STAND_FRAMES)
    grace_until = 0.0
    face_lost_at = None
    wait_close = False
    masked = False
    active_dirs = list(DIRECTIONS)
    active_rounds = ROUNDS
    mask_low, mask_up, mask_bhat = [], [], []
    mask_patches = None
    state, cand, cand_n = "CENTER", None, 0
    prev_state = "CENTER"
    armed = False
    entry = None
    attempts = 0
    until = 0.0
    flash_side, flash_until = None, 0.0
    last_event = time.time()
    info, info_until = "", 0.0
    pending_at = None
    closed_frames = 0
    blink_latched = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()
        st = tracker.update(frame)
        view = cv2.flip(frame, 1)
        if st.ok and st.bbox and not args.avatar:
            cv2.rectangle(view, st.bbox[:2], st.bbox[2:], gp.OK_COLOR, 1)

        # debounced state (directions outside the active set are ignored,
        # e.g. DOWN in mask mode)
        raw = st.direction if st.ok else "CENTER"
        if raw != "CENTER" and raw not in active_dirs:
            raw = "CENTER"
        if raw != state:
            if raw == cand:
                cand_n += 1
            else:
                cand, cand_n = raw, 1
            if cand_n >= DEBOUNCE_N:
                state, cand, cand_n = raw, None, 0
        else:
            cand, cand_n = None, 0

        # long-blink detector (cancel current digit)
        long_blink = False
        if st.ok and st.blink_score > 0.35:
            closed_frames += 1
            if closed_frames >= LONG_BLINK_FRAMES and not blink_latched:
                long_blink = True
                blink_latched = True
        elif st.blink_score < 0.25:
            closed_frames = 0
            blink_latched = False

        # presence tracking (normalized bbox center + face size)
        nface = 0.0
        if st.ok and st.bbox:
            fw = view.shape[1]
            centers.append(((st.bbox[0] + st.bbox[2]) / 2 / fw,
                            (st.bbox[1] + st.bbox[3]) / 2 / fw))
            nface = st.face_px / fw
        else:
            centers.clear()

        # visitor left mid-entry -> reset the session for the next person
        if app in ("ENTER", "CONFIRM"):
            if not st.ok:
                if face_lost_at is None:
                    face_lost_at = now
                elif now - face_lost_at > FACE_LOST_RESET_S:
                    app = "WAIT"
                    centers.clear()
                    tracker.reset_calibration()
                    entry = PinEntry4(active_dirs, active_rounds)
                    armed = False
                    pending_at = None
                    face_lost_at = None
                    info, info_until = "Visitor left - session reset", now + 2.5
                    print("[presence] face lost - session reset")
            else:
                face_lost_at = None

        # ---- app state machine ----
        if app == "WAIT":
            wait_close = st.ok and nface >= MIN_FACE_FRAC
            stable = (len(centers) == centers.maxlen
                      and max(c[0] for c in centers) - min(c[0] for c in centers) < STABLE_TOL
                      and max(c[1] for c in centers) - min(c[1] for c in centers) < STABLE_TOL)
            if wait_close and stable:
                app = "CALIB"
                calib_frames = 0
                mask_low, mask_up, mask_bhat = [], [], []
                grace_until = now + 0.6
                g.beep(600, 80)
                print("[presence] visitor standing still - starting calibration")

        elif app == "CALIB":
            if not st.ok:
                app = "WAIT"
                centers.clear()
            else:
                # collect mask-detection evidence during the hold-still window
                if st.landmarks is not None:
                    mask_low.append([(st.landmarks[i].x, st.landmarks[i].y) for i in LOWER_IDS])
                    mask_up.append([(st.landmarks[i].x, st.landmarks[i].y) for i in UPPER_IDS])
                if st.landmarks is not None and st.face_px:
                    cs = mask_color_score(view, st.landmarks, st.face_px)
                    if cs is not None:
                        mask_bhat.append(cs[0])
                        mask_patches = (cs[1], cs[2])
                moved = (len(centers) >= 2
                         and (abs(centers[-1][0] - centers[-2][0]) > JUMP_TOL
                              or abs(centers[-1][1] - centers[-2][1]) > JUMP_TOL))
                if moved:
                    calib_frames = 0
                elif now >= grace_until:
                    calib_frames += 1
                if calib_frames >= CALIB_FRAMES and tracker.calibrate():
                    # decide mask mode ONCE per session
                    ratio, bhat = 0.0, 0.0
                    if len(mask_low) >= 8:
                        low = np.array(mask_low)
                        up = np.array(mask_up)
                        jl = float(np.mean(np.std(low, axis=0)))
                        ju = float(np.mean(np.std(up, axis=0)))
                        ratio = jl / max(ju, 1e-6)
                    if mask_bhat:
                        bhat = float(np.median(mask_bhat))
                    # landmark-anchored mean-color distance is the sole signal;
                    # opt-in only: hue is unstable at low saturation/lighting
                    masked = args.mask_detect and bhat > MASK_COLOR_DIST
                    active_dirs = list(MASKED_DIRS) if masked else list(DIRECTIONS)
                    active_rounds = MASKED_ROUNDS if masked else ROUNDS
                    print(f"[mask] jitter_ratio={ratio:.2f} color_dist={bhat:.1f} -> "
                          f"{'MASKED: 3-way mode' if masked else 'no mask: 4-way mode'}")
                    if mask_patches:
                        u, l = mask_patches
                        print(f"[mask-debug] glabella HSV=({u[0]:.0f},{u[1]:.0f},{u[2]:.0f}) "
                              f"philtrum HSV=({l[0]:.0f},{l[1]:.0f},{l[2]:.0f})")
                    entry = PinEntry4(active_dirs, active_rounds)
                    attempts = 0
                    armed = False
                    state = "CENTER"
                    last_event = now
                    app = "ENTER"
                    g.beep(800, 120)
                    print("[calibrated] neutral head pose locked")

        elif app == "ENTER":
            if state == "CENTER" and pending_at is None:
                armed = True
            if long_blink:
                if pending_at is not None:
                    pending_at = None
                    entry.entered.pop()
                    entry.reset_symbol()
                    info, info_until = "Last digit cancelled - re-enter it", now + 2.5
                    g.beep(500, 250)
                    print("[blink] last digit cancelled")
                elif entry.bits:
                    entry.reset_symbol()
                    info, info_until = "Digit rounds cancelled - restart this digit", now + 2.5
                    g.beep(500, 250)
                    print("[blink] current digit rounds cancelled")
                last_event = now
            elif armed and prev_state == "CENTER" and state in DIRECTIONS and pending_at is None:
                armed = False
                last_event = now
                flash_side, flash_until = state, now + 0.35
                res = entry.answer(state)
                print(f"[select] {state}  -> {res}")
                if res == "round":
                    g.beep(1000, 70)
                elif res == "symbol":
                    g.beep(700, 130)
                elif res == "invalid":
                    info, info_until = "Input error detected - digit restarted", now + 2.5
                    g.beep(300, 300)
                else:
                    g.beep(700, 130)
                    pending_at = now + FINAL_CHECK_DELAY
            if pending_at is not None and now >= pending_at:
                pending_at = None
                if "".join(entry.entered) == pin:
                    app = "CONFIRM"
                    until = now + 10.0
                    armed = False
                    g.beep(1100, 150)
                    print("[PIN OK] asking whether to open the door")
                else:
                    attempts += 1
                    print(f"[wrong PIN] attempt {attempts}/{MAX_ATTEMPTS}")
                    if attempts >= MAX_ATTEMPTS:
                        app = "LOCKOUT"
                        until = now + LOCKOUT_S
                        g.beep(250, 700)
                    else:
                        app = "FAIL"
                        until = now + 2.5
                        g.beep(300, 400)
            if now - last_event > INACTIVITY_S and (entry.entered or entry.bits):
                entry = PinEntry4(active_dirs, active_rounds)
                info, info_until = "Timed out - entry restarted", now + 3.0
                last_event = now
                g.beep(300, 150)

        elif app == "CONFIRM":
            if state == "CENTER":
                armed = True
            if armed and prev_state == "CENTER" and state in ("LEFT", "RIGHT"):
                armed = False
                flash_side, flash_until = state, now + 0.35
                if state == "RIGHT":
                    app = "SUCCESS"
                    until = now + 4.0
                    g.beep(1400, 350)
                    print("[door opened]")
                else:
                    app = "DECLINED"
                    until = now + 2.5
                    g.beep(400, 250)
                    print("[declined] door stays locked")
            elif now >= until:
                app = "DECLINED"
                until = now + 2.5
                g.beep(400, 250)
                print("[confirm timeout] door stays locked")

        elif app in ("SUCCESS", "DECLINED"):
            if now >= until:
                attempts = 0
                entry = PinEntry4(active_dirs, active_rounds)
                armed = False
                pending_at = None
                centers.clear()
                tracker.reset_calibration()
                last_event = now
                app = "WAIT"

        elif app == "FAIL":
            if now >= until:
                entry = PinEntry4(active_dirs, active_rounds)
                armed = False
                pending_at = None
                last_event = now
                app = "ENTER"

        elif app == "LOCKOUT":
            if now >= until:
                attempts = 0
                entry = PinEntry4(active_dirs, active_rounds)
                armed = False
                pending_at = None
                last_event = now
                app = "ENTER"

        prev_state = state

        # ---- draw ----
        bg = gp.BG
        if app in ("ENTER", "FAIL") and entry is not None:
            bg = STAGE_BGS[min(entry.symbol_idx, PIN_LEN - 1)]
        canvas = np.full((H, W, 3), bg, np.uint8)
        gp.put(canvas, "HeadPIN 4-way Door Lock", (30, 45), 0.9, gp.TXT, 2, cv2.FONT_HERSHEY_DUPLEX)
        gp.put(canvas, f"attempts {attempts}/{MAX_ATTEMPTS}", (W - 180, 45), 0.55, gp.DIM)

        if app == "WAIT":
            gp.put_center(canvas, "Stand in front of the door", W // 2, 290, 1.1, gp.TXT, 2,
                          cv2.FONT_HERSHEY_DUPLEX)
            if not st.ok:
                gp.put_center(canvas, "waiting for a visitor...", W // 2, 350, 0.7, gp.DIM, 2)
            elif not wait_close:
                gp.put_center(canvas, "Come closer", W // 2, 350, 0.8, gp.ACCENT["LEFT"], 2)
            else:
                frac = len(centers) / centers.maxlen
                gp.put_center(canvas, "Hold still...", W // 2, 350, 0.8, gp.OK_COLOR, 2)
                cv2.rectangle(canvas, (W // 2 - 120, 380), (W // 2 + 120, 394), (70, 70, 70), 1)
                cv2.rectangle(canvas, (W // 2 - 120, 380),
                              (W // 2 - 120 + int(240 * frac), 394), gp.OK_COLOR, -1)
        elif app == "CALIB":
            ex, ey = W // 2, 290
            accent = gp.ACCENT["LEFT"]
            # rings converging onto the eye -> pulls the gaze inward
            phase = (now * 1.1) % 1.0
            ring_r = int(95 - 60 * phase)
            ring_col = tuple(int(c * (0.25 + 0.75 * phase)) for c in accent)
            cv2.circle(canvas, (ex, ey), ring_r, ring_col, 2)
            # eye icon: almond outline + iris + pupil (pupil dilates with progress)
            prog = min(1.0, calib_frames / CALIB_FRAMES)
            cv2.ellipse(canvas, (ex, ey), (50, 28), 0, 0, 360, gp.TXT, 2)
            cv2.circle(canvas, (ex, ey), 16, accent, -1)
            cv2.circle(canvas, (ex, ey), 6 + int(5 * prog), (25, 25, 25), -1)
            cv2.circle(canvas, (ex - 5, ey - 5), 2, (245, 245, 245), -1)
            gp.put_center(canvas, "LOOK HERE", ex, ey + 68, 0.7, accent, 2, cv2.FONT_HERSHEY_DUPLEX)
            gp.put_center(canvas, "DO NOT MOVE", W // 2, 400, 0.85, gp.TXT, 2,
                          cv2.FONT_HERSHEY_DUPLEX)
            gp.put_center(canvas, f"calibrating {int(100 * prog)}%", W // 2, 435, 0.7, gp.DIM, 2)
        elif app == "CONFIRM":
            gp.put_center(canvas, "PIN OK - Open the door?", W // 2, 160, 1.1, gp.TXT, 2,
                          cv2.FONT_HERSHEY_DUPLEX)
            for side, x0, x1, label, col in (("LEFT", 120, 420, "< NO", gp.BAD_COLOR),
                                             ("RIGHT", 540, 840, "YES >", gp.OK_COLOR)):
                live_s = (state == side)
                fl = (flash_side == side and now < flash_until)
                cv2.rectangle(canvas, (x0, 220), (x1, 420), gp.PANEL, -1)
                cv2.rectangle(canvas, (x0, 220), (x1, 420),
                              col if (live_s or fl) else (95, 95, 95), 4 if live_s else 1)
                gp.put_center(canvas, label, (x0 + x1) // 2, 335, 1.6, col, 4, cv2.FONT_HERSHEY_DUPLEX)
            hint = "turn your head:  LEFT = no   RIGHT = yes" if armed else "face forward first"
            gp.put_center(canvas, hint, W // 2, 470, 0.65, gp.OK_COLOR if armed else gp.TXT, 1)
            gp.put_center(canvas, f"auto-cancel in {max(0, int(until - now) + 1)}s",
                          W // 2, 505, 0.55, gp.DIM, 1)
        elif app == "DECLINED":
            cv2.rectangle(canvas, (0, 90), (W, H), (40, 40, 48), -1)
            gp.put_center(canvas, "Door stays locked", W // 2, 320, 1.6, gp.TXT, 3,
                          cv2.FONT_HERSHEY_DUPLEX)
        elif app == "SUCCESS":
            cv2.rectangle(canvas, (0, 90), (W, H), (35, 70, 35), -1)
            gp.put_center(canvas, "UNLOCKED", W // 2, 320, 2.6, gp.OK_COLOR, 6, cv2.FONT_HERSHEY_DUPLEX)
            gp.put_center(canvas, "Welcome!", W // 2, 400, 1.0, gp.TXT, 2)
        elif app == "LOCKOUT":
            cv2.rectangle(canvas, (0, 90), (W, H), (30, 30, 70), -1)
            gp.put_center(canvas, "LOCKED OUT", W // 2, 320, 2.2, gp.BAD_COLOR, 5, cv2.FONT_HERSHEY_DUPLEX)
            gp.put_center(canvas, f"try again in {int(until - now) + 1}s", W // 2, 400, 0.9, gp.TXT, 2)
        else:
            # PIN slots: the current slot is split into ROUNDS half-cells and
            # fills one half per head turn — "2 turns complete 1 digit" is
            # conveyed by the structure itself, no round counter needed
            slot_w, slot_h = 62, 56
            for i in range(PIN_LEN):
                x0 = W // 2 - 156 + i * 80
                y0 = 58
                if i < entry.symbol_idx:
                    cv2.rectangle(canvas, (x0, y0), (x0 + slot_w, y0 + slot_h), gp.PANEL, -1)
                    cv2.rectangle(canvas, (x0, y0), (x0 + slot_w, y0 + slot_h), gp.OK_COLOR, 2)
                    gp.put_center(canvas, "*", x0 + slot_w // 2, y0 + 45, 1.4, gp.OK_COLOR, 3,
                                  cv2.FONT_HERSHEY_DUPLEX)
                elif i == entry.symbol_idx:
                    for k in range(entry.rounds):
                        hx0 = x0 + k * (slot_w // entry.rounds)
                        filled = k < entry.round_idx
                        cv2.rectangle(canvas, (hx0 + 3, y0 + 3),
                                      (hx0 + slot_w // entry.rounds - 3, y0 + slot_h - 3),
                                      gp.OK_COLOR if filled else gp.PANEL, -1)
                    cv2.rectangle(canvas, (x0, y0), (x0 + slot_w, y0 + slot_h), gp.TXT, 2)
                else:
                    cv2.rectangle(canvas, (x0, y0), (x0 + slot_w, y0 + slot_h), gp.PANEL, -1)
                    cv2.rectangle(canvas, (x0, y0), (x0 + slot_w, y0 + slot_h), (85, 85, 85), 1)
            gp.put(canvas, f"{entry.rounds} turns", (W // 2 + 180, 82), 0.5, gp.DIM, 1)
            gp.put(canvas, "= 1 digit", (W // 2 + 180, 104), 0.5, gp.DIM, 1)
            if masked:
                gp.put(canvas, "MASK MODE - 3 directions", (30, 70), 0.55, gp.ACCENT["LEFT"], 2)

            live = state if state in DIRECTIONS else None
            fl = flash_side if now < flash_until else None
            draw_board(canvas, entry, live, fl, armed, active_dirs)

            # mini head cursor over the keypad: shows where the head turn is
            # being read, right where the user is already looking
            if st.ok:
                mcx, mcy = 480, 296
                mx = mcx - int(np.clip(st.yaw, -25, 25) / 25 * 95)    # +yaw = LEFT
                my_ = mcy - int(np.clip(st.pitch, -25, 25) / 25 * 95)  # +pitch = UP
                mcol = DIR_COLORS.get(state, (200, 200, 200))
                cv2.circle(canvas, (mx, my_), 5, mcol, -1)
                cv2.circle(canvas, (mx, my_), 8, mcol, 1)

            if entry.symbol_idx >= PIN_LEN:
                gp.put_center(canvas, "Verifying...  (long blink cancels the last digit)",
                              W // 2, 490, 0.6, gp.TXT, 1)
            elif armed:
                gp.put_center(canvas, "turn your head toward your digit's arrow  |  long blink = redo digit",
                              W // 2, 490, 0.58, gp.OK_COLOR, 1)
            else:
                gp.put_center(canvas, "face forward to arm", W // 2, 490, 0.58, gp.TXT, 1)

        # 2D head joystick pad
        if st.ok and app in ("ENTER", "FAIL", "CONFIRM"):
            px0, py0, sz = 80, 495, 150
            cv2.rectangle(canvas, (px0, py0), (px0 + sz, py0 + sz), (70, 70, 70), 1)
            cv2.line(canvas, (px0 + sz // 2, py0), (px0 + sz // 2, py0 + sz), (55, 55, 55), 1)
            cv2.line(canvas, (px0, py0 + sz // 2), (px0 + sz, py0 + sz // 2), (55, 55, 55), 1)
            jx = px0 + sz // 2 - int(np.clip(st.yaw, -25, 25) / 25 * (sz // 2 - 6))   # +yaw = LEFT
            jy = py0 + sz // 2 - int(np.clip(st.pitch, -25, 25) / 25 * (sz // 2 - 6))  # +pitch = UP
            col = DIR_COLORS.get(state, gp.OK_COLOR)
            cv2.circle(canvas, (jx, jy), 7, col, -1)
            gp.put(canvas, f"{state}", (px0, py0 + sz + 22), 0.55, col, 2)
            if closed_frames > 4:
                gp.put(canvas, f"blink {min(closed_frames, LONG_BLINK_FRAMES)}/{LONG_BLINK_FRAMES}",
                       (px0, py0 - 10), 0.5, (80, 200, 255), 1)

        if now < info_until:
            gp.put_center(canvas, info, W // 2, 615, 0.6, (80, 200, 255), 2)
        gp.put(canvas, "c=recalibrate  r=restart  x=flip-yaw  q=quit", (300, H - 20), 0.5, gp.DIM)

        # avatar is applied only now, AFTER mask-detection sampled real colors
        if args.avatar and st.ok and st.landmarks is not None:
            draw_avatar(view, st.landmarks, st.bbox, st)
        inset = cv2.resize(view, (240, 180))
        canvas[H - 200:H - 20, W - 260:W - 20] = inset
        cv2.rectangle(canvas, (W - 260, H - 200), (W - 20, H - 20), (90, 90, 90), 1)

        cv2.imshow("HeadPIN Door Lock", canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c'):
            app = "WAIT"
            calib_frames = 0
            centers.clear()
            tracker.reset_calibration()
        elif k == ord('x'):
            tracker.flip_yaw()
            print("[yaw sign flipped]")
        elif k == ord('r') and app == "ENTER":
            entry = PinEntry4(active_dirs, active_rounds)
            pending_at = None
            info, info_until = "Entry restarted", now + 2.0
            last_event = now

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()

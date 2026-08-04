# GazePIN door-unlock demo
# 8 symbols (1-8), 4-symbol PIN, 3 rounds/symbol = 12 gaze selections.
# Every round all 8 symbols are shown split 4/4; the user answers which side
# holds their symbol by glancing LEFT or RIGHT (edge-triggered, must return to
# center between rounds). The screen never reveals candidate narrowing, and
# partitions are generated from public state only — never from the PIN.
#
# Usage: python gazepin_demo.py [--pin 1234] [--source 0]
# Keys : c = recalibrate, r = restart entry, q = quit
import argparse
import math
import random
import time
import cv2
import numpy as np
from collections import deque
from uniface import RetinaFace
from onnx_inference import GazeEstimationONNX
import lr_gaze_fast as g  # tuned FSM, thresholds, beep

SYMBOLS = [str(i) for i in range(1, 9)]
ROUNDS = 3            # ceil(log2(8))
PIN_LEN = 4
DETECT_EVERY = 10
CALIB_SAMPLES = 40
MAX_ATTEMPTS = 3
LOCKOUT_S = 30.0
INACTIVITY_S = 25.0
UNDO_HOLD_FRAMES = 32   # keep gazing sideways ~1.1s after a selection -> undo it
FINAL_CHECK_DELAY = 1.4  # grace window to undo the 12th selection
ARM_FRAMES = 1           # a single confirmed center glance arms the round (FSM already debounces)
CANVAS_W, CANVAS_H = 960, 680

BG = (28, 26, 24)
PANEL = (52, 48, 44)
ACCENT = {"LEFT": (80, 200, 255), "RIGHT": (255, 200, 80)}
OK_COLOR = (90, 210, 90)
BAD_COLOR = (70, 70, 230)
TXT = (235, 235, 235)
DIM = (150, 150, 150)
# fixed public digit->color mapping (pre-attentive search aid; no security impact)
DIGIT_COLORS = {
    "1": (80, 80, 240),    # red
    "2": (50, 160, 255),   # orange
    "3": (70, 220, 255),   # yellow
    "4": (110, 215, 110),  # green
    "5": (220, 210, 90),   # teal
    "6": (255, 170, 90),   # light blue
    "7": (235, 120, 180),  # violet
    "8": (200, 100, 245),  # magenta
}


def make_partition(cells):
    """Split every cell evenly, random side per cell. Never reads the PIN."""
    left = set()
    for cell in cells:
        c = list(cell)
        random.shuffle(c)
        half = c[: len(c) // 2]
        left.update(half if random.random() < 0.5 else [s for s in c if s not in half])
    return left


def split_cells(cells, left):
    out = []
    for cell in cells:
        a = [s for s in cell if s in left]
        b = [s for s in cell if s not in left]
        if a:
            out.append(a)
        if b:
            out.append(b)
    return out


class PinEntry:
    def __init__(self):
        self.entered = []
        self._hist = []
        self._new_symbol()

    def _new_symbol(self):
        self.cells = [SYMBOLS[:]]
        self.vectors = {s: [] for s in SYMBOLS}
        self.bits = []
        self._new_round()

    def _new_round(self):
        self.left = make_partition(self.cells)
        for s in SYMBOLS:
            self.vectors[s].append(s in self.left)

    def answer(self, is_left):
        """Returns 'round' | 'symbol' | 'done'."""
        self._hist.append(([c[:] for c in self.cells],
                           {k: v[:] for k, v in self.vectors.items()},
                           self.bits[:], self.entered[:], set(self.left)))
        self.bits.append(is_left)
        self.cells = split_cells(self.cells, self.left)
        if len(self.bits) == ROUNDS:
            sym = next(s for s in SYMBOLS if self.vectors[s] == self.bits)
            self.entered.append(sym)
            if len(self.entered) == PIN_LEN:
                return "done"
            self._new_symbol()
            return "symbol"
        self._new_round()
        return "round"

    def undo(self):
        """Revert the last selection (same partition is shown again)."""
        if not self._hist:
            return False
        self.cells, self.vectors, self.bits, self.entered, self.left = self._hist.pop()
        return True

    @property
    def symbol_idx(self):
        return len(self.entered)

    @property
    def round_idx(self):
        return len(self.bits)


def put(canvas, text, org, scale=0.6, color=TXT, thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(canvas, text, (int(org[0]), int(org[1])), font, scale, color, thick, cv2.LINE_AA)


def put_center(canvas, text, cx, y, scale=0.6, color=TXT, thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    put(canvas, text, (cx - tw // 2, y), scale, color, thick, font)


def draw_entry_board(canvas, left_syms, right_syms, live, flash, armed, center_frames):
    """Digits sit in a FIXED central grid (position never changes, like a keypad,
    so finding your digit is instant). What re-randomizes every round is only the
    side badge on each tile (< blue / > orange) — i.e. the partition rendering.
    Command zone = big arrow targets at the screen edges; a center glance arms."""
    # edge command targets
    for side, x0, x1, acx in (("LEFT", 40, 170, 105), ("RIGHT", 790, 920, 855)):
        fl = (flash == side)
        fill = tuple(int(c * 0.5 + f * 0.5) for c, f in zip(PANEL, OK_COLOR)) if fl else PANEL
        cv2.rectangle(canvas, (x0, 160), (x1, 430), fill, -1)
        border = ACCENT[side] if (live == side or fl) else (85, 85, 85)
        cv2.rectangle(canvas, (x0, 160), (x1, 430), border, 3 if live == side else 1)
        put_center(canvas, "<" if side == "LEFT" else ">", acx, 315, 3.0, ACCENT[side], 6,
                   cv2.FONT_HERSHEY_DUPLEX)
    # fixed central digit grid: 2 cols x 4 rows, row-major 1..8, positions permanent
    for i, s in enumerate(SYMBOLS):
        cx = 415 if i % 2 == 0 else 545
        yc = 205 + (i // 2) * 67
        side = "LEFT" if s in left_syms else "RIGHT"
        tint = tuple(int(p * 0.72 + a * 0.28) for p, a in zip(PANEL, ACCENT[side]))
        cv2.rectangle(canvas, (cx - 42, yc - 27), (cx + 42, yc + 27), tint, -1)
        cv2.rectangle(canvas, (cx - 42, yc - 27), (cx + 42, yc + 27), ACCENT[side], 1)
        if side == "LEFT":
            put_center(canvas, "<", cx - 24, yc + 10, 0.9, ACCENT[side], 2, cv2.FONT_HERSHEY_DUPLEX)
            put_center(canvas, s, cx + 10, yc + 12, 1.1, DIGIT_COLORS[s], 3, cv2.FONT_HERSHEY_DUPLEX)
        else:
            put_center(canvas, s, cx - 10, yc + 12, 1.1, DIGIT_COLORS[s], 3, cv2.FONT_HERSHEY_DUPLEX)
            put_center(canvas, ">", cx + 24, yc + 10, 0.9, ACCENT[side], 2, cv2.FONT_HERSHEY_DUPLEX)
    # arming dot between the grid columns
    dot = (480, 305)
    if armed:
        cv2.circle(canvas, dot, 9, OK_COLOR, -1)
    else:
        cv2.circle(canvas, dot, 9, (90, 90, 90), 2)
        frac = min(1.0, center_frames / ARM_FRAMES)
        if frac > 0:
            cv2.ellipse(canvas, dot, (15, 15), -90, 0, int(360 * frac), OK_COLOR, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default="1234")
    ap.add_argument("--source", type=int, default=0)
    args = ap.parse_args()
    pin = args.pin
    if len(pin) != PIN_LEN or any(ch not in SYMBOLS for ch in pin):
        raise SystemExit(f"PIN must be {PIN_LEN} symbols from 1-8, got: {pin}")
    print(f"[GazePIN] demo PIN = {pin} (console only, never shown on screen)")

    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    engine = GazeEstimationONNX(model_path=g.MODEL)
    detector = RetinaFace()

    app = "CALIB"
    calib = []
    baseline = None
    yaw_buf = deque(maxlen=g.SMOOTH_N)
    fsm = g.GazeStateMachine()
    prev_state = "CENTER"
    armed = False
    entry = None
    attempts = 0
    until = 0.0
    flash_side, flash_until = None, 0.0
    last_event = time.time()
    info, info_until = "", 0.0
    bbox = None
    frame_i = 0
    undo_side, undo_frames = None, 0
    pending_at = None
    center_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()

        # ---- face + gaze ----
        if frame_i % DETECT_EVERY == 0 or bbox is None:
            faces = detector.detect(frame)
            bbox = None
            for face in faces:
                b = face["bbox"] if isinstance(face, dict) else face.bbox
                bbox = list(map(int, b[:4]))
                break
        frame_i += 1

        yaw_deg = None
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            x0, y0 = max(0, x0), max(0, y0)
            crop = frame[y0:y1, x0:x1]
            if crop.size > 0:
                yaw, _ = engine.estimate(crop)
                yaw_deg = math.degrees(yaw)
                cv2.rectangle(frame, (x0, y0), (x1, y1), OK_COLOR, 1)

        yaw_rel = None
        yaw_s = None
        state = "CENTER"
        if yaw_deg is not None:
            yaw_buf.append(yaw_deg)
            yaw_s = float(np.median(yaw_buf))
            if baseline is not None:
                yaw_rel = -(yaw_s - baseline)
                state = fsm.update(yaw_rel)

        # ---- app state machine ----
        if app == "CALIB":
            if yaw_deg is not None:
                calib.append(float(np.median(yaw_buf)))
            if len(calib) >= CALIB_SAMPLES:
                baseline = float(np.median(calib))
                fsm = g.GazeStateMachine()
                entry = PinEntry()
                attempts = 0
                armed = False
                last_event = now
                app = "ENTER"
                g.beep(800, 120)
                print(f"[calibrated] baseline yaw = {baseline:+.1f}")

        elif app == "ENTER":
            # Midas-touch gate: a round arms only after a sustained center fixation
            if state == "CENTER":
                center_frames += 1
            else:
                center_frames = 0
            if not armed and center_frames >= ARM_FRAMES and pending_at is None:
                armed = True
            if armed and center_frames >= 3 and yaw_s is not None:
                # drift correction only during a confirmed center fixation
                baseline = (1 - g.BASELINE_EMA) * baseline + g.BASELINE_EMA * yaw_s
            if armed and prev_state == "CENTER" and state in ("LEFT", "RIGHT") and pending_at is None:
                armed = False
                center_frames = 0
                last_event = now
                flash_side, flash_until = state, now + 0.35
                undo_side, undo_frames = state, 0
                res = entry.answer(state == "LEFT")
                print(f"[select] {state}  -> {res}")
                if res == "round":
                    g.beep(1000, 70)
                elif res == "symbol":
                    g.beep(700, 130)
                else:  # done: judge after a short grace window (undo still possible)
                    g.beep(700, 130)
                    pending_at = now + FINAL_CHECK_DELAY
            elif undo_side is not None:
                if state == undo_side:
                    undo_frames += 1
                    if undo_frames >= UNDO_HOLD_FRAMES:
                        if entry.undo():
                            pending_at = None
                            info, info_until = "Selection cancelled - answer this round again", now + 2.5
                            g.beep(500, 250)
                            print("[undo] last selection cancelled")
                        undo_side = None
                        last_event = now
                else:
                    undo_side = None
            if pending_at is not None and now >= pending_at:
                pending_at = None
                undo_side = None
                if "".join(entry.entered) == pin:
                    app = "SUCCESS"
                    until = now + 4.0
                    g.beep(1400, 350)
                    print("[UNLOCKED]")
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
                entry = PinEntry()
                info, info_until = "Timed out - entry restarted", now + 3.0
                last_event = now
                g.beep(300, 150)

        elif app in ("SUCCESS", "FAIL"):
            if now >= until:
                if app == "SUCCESS":
                    attempts = 0
                entry = PinEntry()
                armed = False
                undo_side, pending_at = None, None
                last_event = now
                app = "ENTER"

        elif app == "LOCKOUT":
            if now >= until:
                attempts = 0
                entry = PinEntry()
                armed = False
                undo_side, pending_at = None, None
                last_event = now
                app = "ENTER"

        prev_state = state

        # ---- draw UI ----
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
        put(canvas, "GazePIN Door Lock", (30, 45), 0.9, TXT, 2, cv2.FONT_HERSHEY_DUPLEX)
        put(canvas, f"attempts {attempts}/{MAX_ATTEMPTS}", (CANVAS_W - 180, 45), 0.55, DIM)

        if app == "CALIB":
            cv2.circle(canvas, (CANVAS_W // 2, 300), 14, ACCENT["LEFT"], -1)
            cv2.circle(canvas, (CANVAS_W // 2, 300), 22, ACCENT["LEFT"], 2)
            put_center(canvas, "Look at the dot to calibrate", CANVAS_W // 2, 380, 0.8, TXT, 2)
            pct = int(100 * len(calib) / CALIB_SAMPLES)
            put_center(canvas, f"{pct}%", CANVAS_W // 2, 420, 0.7, DIM, 2)
            if yaw_deg is None:
                put_center(canvas, "NO FACE DETECTED", CANVAS_W // 2, 470, 0.7, BAD_COLOR, 2)
        elif app == "SUCCESS":
            cv2.rectangle(canvas, (0, 90), (CANVAS_W, CANVAS_H), (35, 70, 35), -1)
            put_center(canvas, "UNLOCKED", CANVAS_W // 2, 320, 2.6, OK_COLOR, 6, cv2.FONT_HERSHEY_DUPLEX)
            put_center(canvas, "Welcome!", CANVAS_W // 2, 400, 1.0, TXT, 2)
        elif app == "LOCKOUT":
            cv2.rectangle(canvas, (0, 90), (CANVAS_W, CANVAS_H), (30, 30, 70), -1)
            put_center(canvas, "LOCKED OUT", CANVAS_W // 2, 320, 2.2, BAD_COLOR, 5, cv2.FONT_HERSHEY_DUPLEX)
            put_center(canvas, f"try again in {int(until - now) + 1}s", CANVAS_W // 2, 400, 0.9, TXT, 2)
        else:  # ENTER or FAIL overlay
            # PIN progress
            for i in range(PIN_LEN):
                cx = CANVAS_W // 2 - 90 + i * 60
                if i < entry.symbol_idx:
                    cv2.circle(canvas, (cx, 85), 12, OK_COLOR, -1)
                else:
                    cv2.circle(canvas, (cx, 85), 12, DIM, 2)
                    if i == entry.symbol_idx:
                        cv2.circle(canvas, (cx, 85), 15, TXT, 1)
            # round dots
            for r in range(ROUNDS):
                cx = CANVAS_W // 2 - 30 + r * 30
                col = OK_COLOR if r < entry.round_idx else DIM
                cv2.circle(canvas, (cx, 120), 6, col, -1 if r < entry.round_idx else 1)

            live = state if state in ("LEFT", "RIGHT") else None
            fl = flash_side if now < flash_until else None
            left_syms = [s for s in SYMBOLS if s in entry.left]
            right_syms = [s for s in SYMBOLS if s not in entry.left]
            draw_entry_board(canvas, left_syms, right_syms, live, fl, armed, center_frames)

            if entry.symbol_idx >= PIN_LEN:
                put_center(canvas, "Verifying...  (keep gazing sideways to cancel last selection)",
                           CANVAS_W // 2, 470, 0.62, TXT, 1)
            else:
                step = (f"Symbol {entry.symbol_idx + 1}/{PIN_LEN}  round {entry.round_idx + 1}/{ROUNDS}  -  ")
                if armed:
                    put_center(canvas, step + "glance at the < or > target on YOUR digit's side",
                               CANVAS_W // 2, 470, 0.62, OK_COLOR, 1)
                else:
                    put_center(canvas, step + "find your digit, then look at the center dot",
                               CANVAS_W // 2, 470, 0.62, TXT, 1)
            if undo_side is not None and undo_frames > 8:
                frac = min(1.0, undo_frames / UNDO_HOLD_FRAMES)
                cv2.rectangle(canvas, (CANVAS_W // 2 - 120, 486), (CANVAS_W // 2 + 120, 500), (70, 70, 70), 1)
                cv2.rectangle(canvas, (CANVAS_W // 2 - 120, 486),
                              (CANVAS_W // 2 - 120 + int(240 * frac), 500), (80, 200, 255), -1)
                put_center(canvas, "hold to UNDO", CANVAS_W // 2, 520, 0.5, (80, 200, 255), 1)
            elif not armed and state != "CENTER":
                put_center(canvas, "return to center...", CANVAS_W // 2, 500, 0.55, DIM, 1)

            if app == "FAIL":
                cv2.rectangle(canvas, (200, 240), (760, 360), (30, 30, 70), -1)
                put_center(canvas, "WRONG PIN", CANVAS_W // 2, 315, 1.4, BAD_COLOR, 3, cv2.FONT_HERSHEY_DUPLEX)

        # gaze meter
        if yaw_rel is not None and app in ("ENTER", "FAIL"):
            cx, my = 300, 560
            cv2.line(canvas, (cx - 150, my), (cx + 150, my), (90, 90, 90), 2)
            for t in (-g.ENTER_LEFT_DEG, g.ENTER_RIGHT_DEG):
                tx = cx + int(np.clip(t, -30, 30) / 30 * 150)
                cv2.line(canvas, (tx, my - 8), (tx, my + 8), DIM, 1)
            px = cx + int(np.clip(yaw_rel, -30, 30) / 30 * 150)
            col = ACCENT.get(state, OK_COLOR)
            cv2.circle(canvas, (px, my), 8, col, -1)
            put(canvas, state, (cx - 150, my + 35), 0.55, col, 2)

        # transient info + hotkeys
        if now < info_until:
            put_center(canvas, info, CANVAS_W // 2, 615, 0.6, (80, 200, 255), 2)
        put(canvas, "c=recalibrate  r=restart  q=quit", (30, CANVAS_H - 20), 0.5, DIM)

        # camera inset
        inset = cv2.resize(frame, (240, 180))
        canvas[CANVAS_H - 200:CANVAS_H - 20, CANVAS_W - 260:CANVAS_W - 20] = inset
        cv2.rectangle(canvas, (CANVAS_W - 260, CANVAS_H - 200), (CANVAS_W - 20, CANVAS_H - 20), (90, 90, 90), 1)

        cv2.imshow("GazePIN Door Lock", canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c'):
            app = "CALIB"
            calib = []
            baseline = None
            yaw_buf.clear()
        elif k == ord('r') and app == "ENTER":
            entry = PinEntry()
            undo_side, pending_at = None, None
            info, info_until = "Entry restarted", now + 2.0
            last_event = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

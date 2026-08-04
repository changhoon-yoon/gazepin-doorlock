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

DIR_COLORS = {
    "LEFT": (80, 200, 255),   # orange
    "RIGHT": (255, 200, 80),  # light blue
    "UP": (110, 220, 110),    # green
    "DOWN": (235, 120, 200),  # violet
}
DIR_GLYPH = {"LEFT": "<", "RIGHT": ">", "UP": "^", "DOWN": "v"}

KEYPAD = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], [None, "0", None]]
KEY_CX = {0: 400, 1: 480, 2: 560}
KEY_YC = [200, 263, 326, 389]


def make_partition(cells):
    """Assign every symbol to one of 4 directions; split each cell as evenly as
    possible, balancing group sizes. Uses only public state + randomness."""
    groups = {d: set() for d in DIRECTIONS}
    order = cells[:]
    random.shuffle(order)
    for cell in order:
        c = list(cell)
        random.shuffle(c)
        s = len(c)
        base, rem = divmod(s, 4)
        sizes = [base + 1] * rem + [base] * (4 - rem)
        dirs = sorted(DIRECTIONS, key=lambda d: (len(groups[d]), random.random()))
        i = 0
        for d, k in zip(dirs, sorted(sizes, reverse=True)):
            groups[d].update(c[i:i + k])
            i += k
    return groups


def split_cells(cells, groups):
    out = []
    for cell in cells:
        for d in DIRECTIONS:
            part = [s for s in cell if s in groups[d]]
            if part:
                out.append(part)
    return out


class PinEntry4:
    def __init__(self):
        self.entered = []
        self.reset_symbol()

    def reset_symbol(self):
        """(Re)start the current digit from round 1 with fresh partitions."""
        self.cells = [SYMBOLS[:]]
        self.vectors = {s: [] for s in SYMBOLS}
        self.bits = []
        self._new_round()

    def _new_round(self):
        self.groups = make_partition(self.cells)
        for s in SYMBOLS:
            for d in DIRECTIONS:
                if s in self.groups[d]:
                    self.vectors[s].append(d)
                    break

    def answer(self, direction):
        """Returns 'round' | 'symbol' | 'done' | 'invalid'."""
        self.bits.append(direction)
        self.cells = split_cells(self.cells, self.groups)
        if len(self.bits) == ROUNDS:
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


def draw_board(canvas, entry, live, flash, armed):
    """Fixed keypad + per-round direction badges + 4 edge command targets."""
    # edge targets
    tgt = {
        "LEFT": ((40, 160), (170, 430), (105, 310), 3.0),
        "RIGHT": ((790, 160), (920, 430), (855, 310), 3.0),
        "UP": ((330, 128), (630, 160), (480, 153), 1.1),
        "DOWN": ((330, 430), (630, 462), (480, 456), 1.1),
    }
    for d, (p0, p1, tp, sc) in tgt.items():
        fl = (flash == d)
        fill = tuple(int(c * 0.5 + f * 0.5) for c, f in zip(gp.PANEL, gp.OK_COLOR)) if fl else gp.PANEL
        cv2.rectangle(canvas, p0, p1, fill, -1)
        border = DIR_COLORS[d] if (live == d or fl) else (85, 85, 85)
        cv2.rectangle(canvas, p0, p1, border, 3 if live == d else 1)
        gp.put_center(canvas, DIR_GLYPH[d], tp[0], tp[1], sc, DIR_COLORS[d],
                      4 if sc > 2 else 2, cv2.FONT_HERSHEY_DUPLEX)
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
            tint = tuple(int(p * 0.72 + a * 0.28) for p, a in zip(gp.PANEL, DIR_COLORS[d]))
            cv2.rectangle(canvas, (cx - 36, yc - 27), (cx + 36, yc + 27), tint, -1)
            cv2.rectangle(canvas, (cx - 36, yc - 27), (cx + 36, yc + 27), DIR_COLORS[d], 1)
            gp.put_center(canvas, s, cx - 6, yc + 12, 1.0, gp.TXT, 2, cv2.FONT_HERSHEY_DUPLEX)
            gp.put_center(canvas, DIR_GLYPH[d], cx + 22, yc - 8, 0.55, DIR_COLORS[d], 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default="1234")
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--model", default="models/face_landmarker.task")
    args = ap.parse_args()
    pin = args.pin
    if len(pin) != PIN_LEN or any(ch not in SYMBOLS for ch in pin):
        raise SystemExit(f"PIN must be {PIN_LEN} digits 0-9, got: {pin}")
    print(f"[HeadPIN-4way] demo PIN = {pin} (console only, never shown on screen)")

    tracker = HeadTracker(args.model)
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    W, H = gp.CANVAS_W, gp.CANVAS_H
    app = "CALIB"
    calib_frames = 0
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
        if st.ok and st.bbox:
            cv2.rectangle(view, st.bbox[:2], st.bbox[2:], gp.OK_COLOR, 1)

        # debounced 4-way state
        raw = st.direction if st.ok else "CENTER"
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

        # ---- app state machine ----
        if app == "CALIB":
            if st.ok:
                calib_frames += 1
            if calib_frames >= 15 and tracker.calibrate():
                entry = PinEntry4()
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
                entry = PinEntry4()
                info, info_until = "Timed out - entry restarted", now + 3.0
                last_event = now
                g.beep(300, 150)

        elif app in ("SUCCESS", "FAIL"):
            if now >= until:
                if app == "SUCCESS":
                    attempts = 0
                entry = PinEntry4()
                armed = False
                pending_at = None
                last_event = now
                app = "ENTER"

        elif app == "LOCKOUT":
            if now >= until:
                attempts = 0
                entry = PinEntry4()
                armed = False
                pending_at = None
                last_event = now
                app = "ENTER"

        prev_state = state

        # ---- draw ----
        canvas = np.full((H, W, 3), gp.BG, np.uint8)
        gp.put(canvas, "HeadPIN 4-way Door Lock", (30, 45), 0.9, gp.TXT, 2, cv2.FONT_HERSHEY_DUPLEX)
        gp.put(canvas, f"attempts {attempts}/{MAX_ATTEMPTS}", (W - 180, 45), 0.55, gp.DIM)

        if app == "CALIB":
            cv2.circle(canvas, (W // 2, 300), 14, gp.ACCENT["LEFT"], -1)
            cv2.circle(canvas, (W // 2, 300), 22, gp.ACCENT["LEFT"], 2)
            gp.put_center(canvas, "Face the camera squarely to calibrate", W // 2, 380, 0.8, gp.TXT, 2)
            gp.put_center(canvas, f"{min(100, int(100 * calib_frames / 15))}%", W // 2, 420, 0.7, gp.DIM, 2)
            if not st.ok:
                gp.put_center(canvas, "NO FACE DETECTED", W // 2, 470, 0.7, gp.BAD_COLOR, 2)
        elif app == "SUCCESS":
            cv2.rectangle(canvas, (0, 90), (W, H), (35, 70, 35), -1)
            gp.put_center(canvas, "UNLOCKED", W // 2, 320, 2.6, gp.OK_COLOR, 6, cv2.FONT_HERSHEY_DUPLEX)
            gp.put_center(canvas, "Welcome!", W // 2, 400, 1.0, gp.TXT, 2)
        elif app == "LOCKOUT":
            cv2.rectangle(canvas, (0, 90), (W, H), (30, 30, 70), -1)
            gp.put_center(canvas, "LOCKED OUT", W // 2, 320, 2.2, gp.BAD_COLOR, 5, cv2.FONT_HERSHEY_DUPLEX)
            gp.put_center(canvas, f"try again in {int(until - now) + 1}s", W // 2, 400, 0.9, gp.TXT, 2)
        else:
            for i in range(PIN_LEN):
                cx = W // 2 - 90 + i * 60
                if i < entry.symbol_idx:
                    cv2.circle(canvas, (cx, 85), 12, gp.OK_COLOR, -1)
                else:
                    cv2.circle(canvas, (cx, 85), 12, gp.DIM, 2)
                    if i == entry.symbol_idx:
                        cv2.circle(canvas, (cx, 85), 15, gp.TXT, 1)
            for r in range(ROUNDS):
                cx = W // 2 - 15 + r * 30
                col = gp.OK_COLOR if r < entry.round_idx else gp.DIM
                cv2.circle(canvas, (cx, 113), 6, col, -1 if r < entry.round_idx else 1)

            live = state if state in DIRECTIONS else None
            fl = flash_side if now < flash_until else None
            draw_board(canvas, entry, live, fl, armed)

            if entry.symbol_idx >= PIN_LEN:
                gp.put_center(canvas, "Verifying...  (long blink cancels the last digit)",
                              W // 2, 490, 0.6, gp.TXT, 1)
            else:
                step = f"Digit {entry.symbol_idx + 1}/{PIN_LEN}  round {entry.round_idx + 1}/{ROUNDS}  -  "
                if armed:
                    gp.put_center(canvas, step + "turn your head toward your digit's arrow  |  long blink = redo digit",
                                  W // 2, 490, 0.58, gp.OK_COLOR, 1)
                else:
                    gp.put_center(canvas, step + "face forward to arm", W // 2, 490, 0.58, gp.TXT, 1)

        # 2D head joystick pad
        if st.ok and app in ("ENTER", "FAIL"):
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

        inset = cv2.resize(view, (240, 180))
        canvas[H - 200:H - 20, W - 260:W - 20] = inset
        cv2.rectangle(canvas, (W - 260, H - 200), (W - 20, H - 20), (90, 90, 90), 1)

        cv2.imshow("HeadPIN Door Lock", canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c'):
            app = "CALIB"
            calib_frames = 0
            tracker.reset_calibration()
        elif k == ord('x'):
            tracker.flip_yaw()
            print("[yaw sign flipped]")
        elif k == ord('r') and app == "ENTER":
            entry = PinEntry4()
            pending_at = None
            info, info_until = "Entry restarted", now + 2.0
            last_event = now

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()

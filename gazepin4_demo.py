# GazePIN 4-way demo — base-4 protocol driven by GAZE in four directions.
# Known risk (measured earlier): vertical gaze separation is weak on webcams
# (UP ~2.3 sigma, DOWN ~0.6 sigma at 45 deg). To give 4-way its best shot this
# demo uses a 5-point guided calibration (CENTER + L/R/U/D): per-user, per-axis
# signed thresholds at 55% of each measured excursion — signs and scales are
# learned, nothing is hand-tuned.
# Protocol/keypad UI are reused from headpin_demo (0-9 PIN, 2 rounds/digit).
# Hold a direction ~1.1s after selecting = cancel the current digit's rounds.
#
# Usage: python gazepin4_demo.py [--pin 1234] [--source 0]
# Keys : c = recalibrate, r = restart entry, q = quit
import argparse
import math
import time
import cv2
import numpy as np
from collections import deque
from uniface import RetinaFace
from onnx_inference import GazeEstimationONNX
import lr_gaze_fast as g    # beep + model path
import gazepin_demo as gp   # drawing helpers / colors / canvas
import headpin_demo as hp   # base-4 protocol + keypad board + direction colors

DETECT_EVERY = 10
SMOOTH_N = 7
DEBOUNCE_N = 3
ANTI_STICK = 20
ENTER_FRAC = 0.55    # threshold = 55% of calibrated excursion
EXIT_FRAC = 0.50     # drop back to CENTER below 50% of threshold
EMA = 0.02
CANCEL_HOLD = 32     # keep gazing the same direction ~1.1s -> cancel digit
CAL_STAGES = [("CENTER", 40), ("LEFT", 30), ("RIGHT", 30), ("UP", 30), ("DOWN", 30)]
CAL_DOTS = {"CENTER": (480, 300), "LEFT": (105, 300), "RIGHT": (855, 300),
            "UP": (480, 140), "DOWN": (480, 450)}
AXIS = {"LEFT": "yaw", "RIGHT": "yaw", "UP": "pitch", "DOWN": "pitch"}


class GazeFSM4:
    """4-direction hysteresis classifier over calibrated signed thresholds."""

    def __init__(self, thr):
        self.thr = thr  # dir -> signed threshold on its axis (rel units)
        self.state = "CENTER"
        self.cand, self.cand_n = None, 0
        self.weak_n = 0

    def _score(self, d, rel_yaw, rel_pitch):
        t = self.thr[d]
        if t == 0:
            return 0.0
        v = rel_yaw if AXIS[d] == "yaw" else rel_pitch
        s = v / t
        return s if s > 0 else 0.0

    def update(self, rel_yaw, rel_pitch):
        scores = {d: self._score(d, rel_yaw, rel_pitch) for d in hp.DIRECTIONS}
        if self.state == "CENTER":
            best = max(scores, key=scores.get)
            raw = best if scores[best] >= 1.0 else "CENTER"
        else:
            cur = scores[self.state]
            best = max(scores, key=scores.get)
            if scores[best] >= 1.0 and best != self.state:
                raw = best
            elif cur >= EXIT_FRAC:
                raw = self.state
            else:
                raw = "CENTER"
            self.weak_n = 0 if cur >= 1.0 else self.weak_n + 1
            if self.weak_n >= ANTI_STICK:
                raw = "CENTER"
                self.weak_n = 0
        if raw == self.state:
            self.cand, self.cand_n = None, 0
        elif raw == self.cand:
            self.cand_n += 1
            if self.cand_n >= DEBOUNCE_N:
                self.state, self.cand, self.cand_n = raw, None, 0
        else:
            self.cand, self.cand_n = raw, 1
        return self.state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default="1234")
    ap.add_argument("--source", type=int, default=0)
    args = ap.parse_args()
    pin = args.pin
    if len(pin) != hp.PIN_LEN or any(ch not in hp.SYMBOLS for ch in pin):
        raise SystemExit(f"PIN must be {hp.PIN_LEN} digits 0-9, got: {pin}")
    print(f"[GazePIN-4way] demo PIN = {pin} (console only, never shown on screen)")

    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    engine = GazeEstimationONNX(model_path=g.MODEL)
    detector = RetinaFace()

    W, H = gp.CANVAS_W, gp.CANVAS_H
    app = "CALIB"
    stage_idx, stage_samples = 0, []
    stage_gap_until = 0.0
    cal = {}            # stage name -> (yaw_med, pitch_med)
    center = None       # (yaw0, pitch0)
    fsm = None
    yaw_buf = deque(maxlen=SMOOTH_N)
    pitch_buf = deque(maxlen=SMOOTH_N)
    bbox = None
    frame_i = 0
    state = "CENTER"
    prev_state = "CENTER"
    armed = False
    entry = None
    attempts = 0
    until = 0.0
    flash_side, flash_until = None, 0.0
    last_event = time.time()
    info, info_until = "", 0.0
    pending_at = None
    hold_dir, hold_n = None, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()

        if frame_i % DETECT_EVERY == 0 or bbox is None:
            faces = detector.detect(frame)
            bbox = None
            for face in faces:
                b = face["bbox"] if isinstance(face, dict) else face.bbox
                bbox = list(map(int, b[:4]))
                break
        frame_i += 1

        yaw_s = pitch_s = None
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            x0, y0 = max(0, x0), max(0, y0)
            crop = frame[y0:y1, x0:x1]
            if crop.size > 0:
                yaw, pitch = engine.estimate(crop)
                yaw_buf.append(math.degrees(yaw))
                pitch_buf.append(math.degrees(pitch))
                yaw_s = float(np.median(yaw_buf))
                pitch_s = float(np.median(pitch_buf))
                cv2.rectangle(frame, (x0, y0), (x1, y1), gp.OK_COLOR, 1)

        rel_yaw = rel_pitch = None
        if center is not None and yaw_s is not None:
            rel_yaw = yaw_s - center[0]
            rel_pitch = pitch_s - center[1]
            state = fsm.update(rel_yaw, rel_pitch) if fsm else "CENTER"

        # ---- app state machine ----
        if app == "CALIB":
            name, need = CAL_STAGES[stage_idx]
            if yaw_s is not None and now >= stage_gap_until:
                stage_samples.append((yaw_s, pitch_s))
            if len(stage_samples) >= need:
                cal[name] = (float(np.median([s[0] for s in stage_samples])),
                             float(np.median([s[1] for s in stage_samples])))
                stage_samples = []
                stage_gap_until = now + 0.7
                g.beep(900, 60)
                stage_idx += 1
                if stage_idx >= len(CAL_STAGES):
                    center = cal["CENTER"]
                    thr = {}
                    for d in hp.DIRECTIONS:
                        ax = 0 if AXIS[d] == "yaw" else 1
                        exc = cal[d][ax] - center[ax]
                        thr[d] = ENTER_FRAC * exc
                        print(f"[cal] {d}: excursion {exc:+.1f} deg -> threshold {thr[d]:+.1f}")
                        if abs(exc) < 2.5:
                            print(f"[cal] WARNING: {d} excursion is tiny - expect misfires")
                    fsm = GazeFSM4(thr)
                    entry = hp.PinEntry4()
                    attempts = 0
                    armed = False
                    last_event = now
                    app = "ENTER"
                    g.beep(800, 150)
                    print("[calibrated] 5-point gaze calibration complete")

        elif app == "ENTER":
            if state == "CENTER" and pending_at is None:
                armed = True
                if rel_yaw is not None:
                    center = ((1 - EMA) * center[0] + EMA * yaw_s,
                              (1 - EMA) * center[1] + EMA * pitch_s)
            if armed and prev_state == "CENTER" and state in hp.DIRECTIONS and pending_at is None:
                armed = False
                last_event = now
                flash_side, flash_until = state, now + 0.35
                hold_dir, hold_n = state, 0
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
                    pending_at = now + hp.FINAL_CHECK_DELAY
            elif hold_dir is not None:
                if state == hold_dir:
                    hold_n += 1
                    if hold_n >= CANCEL_HOLD:
                        if pending_at is not None:
                            pending_at = None
                            entry.entered.pop()
                        entry.reset_symbol()
                        info, info_until = "Digit cancelled - restart this digit", now + 2.5
                        g.beep(500, 250)
                        print("[hold] current digit cancelled")
                        hold_dir = None
                        last_event = now
                else:
                    hold_dir = None
            if pending_at is not None and now >= pending_at:
                pending_at = None
                hold_dir = None
                if "".join(entry.entered) == pin:
                    app = "SUCCESS"
                    until = now + 4.0
                    g.beep(1400, 350)
                    print("[UNLOCKED]")
                else:
                    attempts += 1
                    print(f"[wrong PIN] attempt {attempts}/{hp.MAX_ATTEMPTS}")
                    if attempts >= hp.MAX_ATTEMPTS:
                        app = "LOCKOUT"
                        until = now + hp.LOCKOUT_S
                        g.beep(250, 700)
                    else:
                        app = "FAIL"
                        until = now + 2.5
                        g.beep(300, 400)
            if now - last_event > hp.INACTIVITY_S and (entry.entered or entry.bits):
                entry = hp.PinEntry4()
                info, info_until = "Timed out - entry restarted", now + 3.0
                last_event = now
                g.beep(300, 150)

        elif app in ("SUCCESS", "FAIL"):
            if now >= until:
                if app == "SUCCESS":
                    attempts = 0
                entry = hp.PinEntry4()
                armed = False
                pending_at = None
                last_event = now
                app = "ENTER"

        elif app == "LOCKOUT":
            if now >= until:
                attempts = 0
                entry = hp.PinEntry4()
                armed = False
                pending_at = None
                last_event = now
                app = "ENTER"

        prev_state = state

        # ---- draw ----
        canvas = np.full((H, W, 3), gp.BG, np.uint8)
        gp.put(canvas, "GazePIN 4-way Door Lock (gaze input)", (30, 45), 0.9, gp.TXT, 2,
               cv2.FONT_HERSHEY_DUPLEX)
        gp.put(canvas, f"attempts {attempts}/{hp.MAX_ATTEMPTS}", (W - 180, 45), 0.55, gp.DIM)

        if app == "CALIB":
            name, need = CAL_STAGES[stage_idx]
            dx, dy = CAL_DOTS[name]
            cv2.circle(canvas, (dx, dy), 14, gp.ACCENT["LEFT"], -1)
            cv2.circle(canvas, (dx, dy), 22, gp.ACCENT["LEFT"], 2)
            gp.put_center(canvas, f"Calibration {stage_idx + 1}/{len(CAL_STAGES)}: "
                          f"look at the dot ({name})", W // 2, 600, 0.75, gp.TXT, 2)
            gp.put_center(canvas, f"{min(100, int(100 * len(stage_samples) / need))}%",
                          W // 2, 640, 0.6, gp.DIM, 2)
            if yaw_s is None:
                gp.put_center(canvas, "NO FACE DETECTED", W // 2, 560, 0.7, gp.BAD_COLOR, 2)
        elif app == "SUCCESS":
            cv2.rectangle(canvas, (0, 90), (W, H), (35, 70, 35), -1)
            gp.put_center(canvas, "UNLOCKED", W // 2, 320, 2.6, gp.OK_COLOR, 6, cv2.FONT_HERSHEY_DUPLEX)
            gp.put_center(canvas, "Welcome!", W // 2, 400, 1.0, gp.TXT, 2)
        elif app == "LOCKOUT":
            cv2.rectangle(canvas, (0, 90), (W, H), (30, 30, 70), -1)
            gp.put_center(canvas, "LOCKED OUT", W // 2, 320, 2.2, gp.BAD_COLOR, 5, cv2.FONT_HERSHEY_DUPLEX)
            gp.put_center(canvas, f"try again in {int(until - now) + 1}s", W // 2, 400, 0.9, gp.TXT, 2)
        else:
            for i in range(hp.PIN_LEN):
                cx = W // 2 - 90 + i * 60
                if i < entry.symbol_idx:
                    cv2.circle(canvas, (cx, 85), 12, gp.OK_COLOR, -1)
                else:
                    cv2.circle(canvas, (cx, 85), 12, gp.DIM, 2)
                    if i == entry.symbol_idx:
                        cv2.circle(canvas, (cx, 85), 15, gp.TXT, 1)
            for r in range(hp.ROUNDS):
                cx = W // 2 - 15 + r * 30
                col = gp.OK_COLOR if r < entry.round_idx else gp.DIM
                cv2.circle(canvas, (cx, 113), 6, col, -1 if r < entry.round_idx else 1)

            live = state if state in hp.DIRECTIONS else None
            fl = flash_side if now < flash_until else None
            hp.draw_board(canvas, entry, live, fl, armed)

            if entry.symbol_idx >= hp.PIN_LEN:
                gp.put_center(canvas, "Verifying...  (hold any direction to cancel the last digit)",
                              W // 2, 490, 0.6, gp.TXT, 1)
            else:
                step = f"Digit {entry.symbol_idx + 1}/{hp.PIN_LEN}  round {entry.round_idx + 1}/{hp.ROUNDS}  -  "
                if armed:
                    gp.put_center(canvas, step + "glance toward your digit's arrow  |  hold = redo digit",
                                  W // 2, 490, 0.58, gp.OK_COLOR, 1)
                else:
                    gp.put_center(canvas, step + "look at the keypad center to arm", W // 2, 490, 0.58, gp.TXT, 1)

        # 2D gaze joystick pad
        if rel_yaw is not None and fsm is not None and app in ("ENTER", "FAIL"):
            px0, py0, sz = 80, 495, 150
            cv2.rectangle(canvas, (px0, py0), (px0 + sz, py0 + sz), (70, 70, 70), 1)
            cv2.line(canvas, (px0 + sz // 2, py0), (px0 + sz // 2, py0 + sz), (55, 55, 55), 1)
            cv2.line(canvas, (px0, py0 + sz // 2), (px0 + sz, py0 + sz // 2), (55, 55, 55), 1)
            nx = max(abs(fsm.thr["LEFT"]), abs(fsm.thr["RIGHT"])) / ENTER_FRAC
            ny = max(abs(fsm.thr["UP"]), abs(fsm.thr["DOWN"])) / ENTER_FRAC
            sgn_x = -1 if fsm.thr["LEFT"] > 0 else 1   # map LEFT excursion to screen-left
            sgn_y = -1 if fsm.thr["UP"] > 0 else 1     # map UP excursion to screen-up
            jx = px0 + sz // 2 + int(np.clip(sgn_x * rel_yaw / max(nx, 1e-6), -1, 1) * (sz // 2 - 6))
            jy = py0 + sz // 2 + int(np.clip(sgn_y * rel_pitch / max(ny, 1e-6), -1, 1) * (sz // 2 - 6))
            col = hp.DIR_COLORS.get(state, gp.OK_COLOR)
            cv2.circle(canvas, (jx, jy), 7, col, -1)
            gp.put(canvas, f"{state}", (px0, py0 + sz + 22), 0.55, col, 2)

        if now < info_until:
            gp.put_center(canvas, info, W // 2, 615, 0.6, (80, 200, 255), 2)
        gp.put(canvas, "c=recalibrate  r=restart  q=quit", (300, H - 20), 0.5, gp.DIM)

        inset = cv2.resize(frame, (240, 180))
        canvas[H - 200:H - 20, W - 260:W - 20] = inset
        cv2.rectangle(canvas, (W - 260, H - 200), (W - 20, H - 20), (90, 90, 90), 1)

        cv2.imshow("GazePIN 4-way Door Lock", canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c'):
            app = "CALIB"
            stage_idx, stage_samples = 0, []
            cal = {}
            center = None
            fsm = None
            state = "CENTER"
        elif k == ord('r') and app == "ENTER":
            entry = hp.PinEntry4()
            pending_at = None
            hold_dir = None
            info, info_until = "Entry restarted", now + 2.0
            last_event = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

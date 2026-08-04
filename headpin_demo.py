# HeadPIN door-unlock demo — the same GazePIN protocol, driven by HEAD TURNS.
# Sensing engine: HeadTracker (MediaPipe FaceLandmarker), copied from the
# head-gesture-doorlock project. Protocol + UI are imported from gazepin_demo,
# so the two variants stay directly comparable.
#
# Usage: python headpin_demo.py [--pin 1234] [--source 0]
# Keys : c = recalibrate neutral pose, r = restart entry, x = flip yaw sign, q = quit
import argparse
import time
import cv2
import numpy as np
from headtracker import HeadTracker
import lr_gaze_fast as g   # beep
import gazepin_demo as gp  # PIN protocol + shared UI

DEBOUNCE_N = 3  # frames a new head direction must persist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default="1234")
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--model", default="models/face_landmarker.task")
    args = ap.parse_args()
    pin = args.pin
    if len(pin) != gp.PIN_LEN or any(ch not in gp.SYMBOLS for ch in pin):
        raise SystemExit(f"PIN must be {gp.PIN_LEN} symbols from 1-8, got: {pin}")
    print(f"[HeadPIN] demo PIN = {pin} (console only, never shown on screen)")

    tracker = HeadTracker(args.model)
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    W, H = gp.CANVAS_W, gp.CANVAS_H
    app = "CALIB"
    calib_frames = 0
    state, cand, cand_n = "CENTER", None, 0
    prev_state = "CENTER"
    armed = False
    center_frames = 0
    entry = None
    attempts = 0
    until = 0.0
    flash_side, flash_until = None, 0.0
    last_event = time.time()
    info, info_until = "", 0.0
    undo_side, undo_frames = None, 0
    pending_at = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        now = time.time()
        st = tracker.update(frame)  # engine mirrors internally
        view = cv2.flip(frame, 1)
        if st.ok and st.bbox:
            cv2.rectangle(view, st.bbox[:2], st.bbox[2:], gp.OK_COLOR, 1)

        # head direction -> debounced L/R/CENTER state (UP/DOWN unused here)
        raw = st.direction if st.ok else "CENTER"
        if raw not in ("LEFT", "RIGHT"):
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

        # ---- app state machine (mirrors gazepin_demo) ----
        if app == "CALIB":
            if st.ok:
                calib_frames += 1
            if calib_frames >= 15 and tracker.calibrate():
                entry = gp.PinEntry()
                attempts = 0
                armed = False
                center_frames = 0
                state = "CENTER"
                last_event = now
                app = "ENTER"
                g.beep(800, 120)
                print("[calibrated] neutral head pose locked")

        elif app == "ENTER":
            if state == "CENTER":
                center_frames += 1
            else:
                center_frames = 0
            if not armed and center_frames >= gp.ARM_FRAMES and pending_at is None:
                armed = True
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
                else:
                    g.beep(700, 130)
                    pending_at = now + gp.FINAL_CHECK_DELAY
            elif undo_side is not None:
                if state == undo_side:
                    undo_frames += 1
                    if undo_frames >= gp.UNDO_HOLD_FRAMES:
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
                    print(f"[wrong PIN] attempt {attempts}/{gp.MAX_ATTEMPTS}")
                    if attempts >= gp.MAX_ATTEMPTS:
                        app = "LOCKOUT"
                        until = now + gp.LOCKOUT_S
                        g.beep(250, 700)
                    else:
                        app = "FAIL"
                        until = now + 2.5
                        g.beep(300, 400)
            if now - last_event > gp.INACTIVITY_S and (entry.entered or entry.bits):
                entry = gp.PinEntry()
                info, info_until = "Timed out - entry restarted", now + 3.0
                last_event = now
                g.beep(300, 150)

        elif app in ("SUCCESS", "FAIL"):
            if now >= until:
                if app == "SUCCESS":
                    attempts = 0
                entry = gp.PinEntry()
                armed = False
                undo_side, pending_at = None, None
                last_event = now
                app = "ENTER"

        elif app == "LOCKOUT":
            if now >= until:
                attempts = 0
                entry = gp.PinEntry()
                armed = False
                undo_side, pending_at = None, None
                last_event = now
                app = "ENTER"

        prev_state = state

        # ---- draw UI ----
        canvas = np.full((H, W, 3), gp.BG, np.uint8)
        gp.put(canvas, "HeadPIN Door Lock (head-turn input)", (30, 45), 0.9, gp.TXT, 2,
               cv2.FONT_HERSHEY_DUPLEX)
        gp.put(canvas, f"attempts {attempts}/{gp.MAX_ATTEMPTS}", (W - 180, 45), 0.55, gp.DIM)

        if app == "CALIB":
            cv2.circle(canvas, (W // 2, 300), 14, gp.ACCENT["LEFT"], -1)
            cv2.circle(canvas, (W // 2, 300), 22, gp.ACCENT["LEFT"], 2)
            gp.put_center(canvas, "Face the camera squarely to calibrate", W // 2, 380, 0.8, gp.TXT, 2)
            pct = min(100, int(100 * calib_frames / 15))
            gp.put_center(canvas, f"{pct}%", W // 2, 420, 0.7, gp.DIM, 2)
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
            for i in range(gp.PIN_LEN):
                cx = W // 2 - 90 + i * 60
                if i < entry.symbol_idx:
                    cv2.circle(canvas, (cx, 85), 12, gp.OK_COLOR, -1)
                else:
                    cv2.circle(canvas, (cx, 85), 12, gp.DIM, 2)
                    if i == entry.symbol_idx:
                        cv2.circle(canvas, (cx, 85), 15, gp.TXT, 1)
            for r in range(gp.ROUNDS):
                cx = W // 2 - 30 + r * 30
                col = gp.OK_COLOR if r < entry.round_idx else gp.DIM
                cv2.circle(canvas, (cx, 120), 6, col, -1 if r < entry.round_idx else 1)

            live = state if state in ("LEFT", "RIGHT") else None
            fl = flash_side if now < flash_until else None
            left_syms = [s for s in gp.SYMBOLS if s in entry.left]
            right_syms = [s for s in gp.SYMBOLS if s not in entry.left]
            gp.draw_entry_board(canvas, left_syms, right_syms, live, fl, armed, center_frames)

            if entry.symbol_idx >= gp.PIN_LEN:
                gp.put_center(canvas, "Verifying...  (keep your head turned to cancel last selection)",
                              W // 2, 470, 0.62, gp.TXT, 1)
            else:
                step = f"Symbol {entry.symbol_idx + 1}/{gp.PIN_LEN}  round {entry.round_idx + 1}/{gp.ROUNDS}  -  "
                if armed:
                    gp.put_center(canvas, step + "TURN YOUR HEAD toward the < or > on your digit's side",
                                  W // 2, 470, 0.62, gp.OK_COLOR, 1)
                else:
                    gp.put_center(canvas, step + "find your digit, then face forward",
                                  W // 2, 470, 0.62, gp.TXT, 1)
            if undo_side is not None and undo_frames > 8:
                frac = min(1.0, undo_frames / gp.UNDO_HOLD_FRAMES)
                cv2.rectangle(canvas, (W // 2 - 120, 486), (W // 2 + 120, 500), (70, 70, 70), 1)
                cv2.rectangle(canvas, (W // 2 - 120, 486),
                              (W // 2 - 120 + int(240 * frac), 500), (80, 200, 255), -1)
                gp.put_center(canvas, "hold to UNDO", W // 2, 520, 0.5, (80, 200, 255), 1)
            elif not armed and state != "CENTER":
                gp.put_center(canvas, "face forward...", W // 2, 500, 0.55, gp.DIM, 1)

        # head-yaw meter
        if st.ok and app in ("ENTER", "FAIL"):
            cx, my = 300, 560
            cv2.line(canvas, (cx - 150, my), (cx + 150, my), (90, 90, 90), 2)
            for t in (-tracker.thr_deg, tracker.thr_deg):
                tx = cx + int(np.clip(t, -30, 30) / 30 * 150)
                cv2.line(canvas, (tx, my - 8), (tx, my + 8), gp.DIM, 1)
            px = cx + int(np.clip(st.yaw, -30, 30) / 30 * 150)
            col = gp.ACCENT.get(state, gp.OK_COLOR)
            cv2.circle(canvas, (px, my), 8, col, -1)
            gp.put(canvas, f"{state}  yaw {st.yaw:+.0f}", (cx - 150, my + 35), 0.55, col, 2)

        if now < info_until:
            gp.put_center(canvas, info, W // 2, 615, 0.6, (80, 200, 255), 2)
        gp.put(canvas, "c=recalibrate  r=restart  x=flip-yaw  q=quit", (30, H - 20), 0.5, gp.DIM)

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
            entry = gp.PinEntry()
            undo_side, pending_at = None, None
            info, info_until = "Entry restarted", now + 2.0
            last_event = now

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()

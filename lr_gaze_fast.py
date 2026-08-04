# Raspberry Pi-oriented fast L/R gaze classifier
# Optimizations vs lr_gaze_test.py:
#  - MobileOne S0 (4.8MB) instead of ResNet-34 (81MB)
#  - Face detection only every DETECT_EVERY frames (bbox reused between)
#  - Camera buffer size 1 to avoid stale-frame lag
#  - Lighter smoothing/hold for faster response
#  - FPS / per-stage latency HUD
import cv2
import math
import time
import random
import threading
import numpy as np

try:
    import winsound
    def beep(freq, ms):
        threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()
except ImportError:  # e.g. Raspberry Pi: terminal bell fallback
    def beep(freq, ms):
        print("\a", end="", flush=True)
from collections import deque
from uniface import RetinaFace
from onnx_inference import GazeEstimationONNX

MODEL = "weights/mobileone_s0_gaze.onnx"
ENTER_LEFT_DEG = 16.0   # hysteresis: rel yaw to ENTER left state (left is noisier)
ENTER_RIGHT_DEG = 12.0  # hysteresis: rel yaw to ENTER right state
EXIT_DEG = 9.0          # hysteresis: |rel yaw| to RETURN to center
STUCK_FRAMES = 20       # in L/R state but below ENTER this long -> force CENTER (~0.7s)
BASELINE_EMA = 0.03     # while CENTER, baseline slowly tracks current yaw (drift fix)
SMOOTH_N = 7            # median filter window
DEBOUNCE_N = 3          # consecutive frames required to switch state
DETECT_EVERY = 10       # run face detector every N frames
TRIALS = 20
HOLD_FRAMES = 5
TRIAL_TIMEOUT_S = 4.0


class GazeStateMachine:
    """Hysteresis + debounce classifier: no flicker at the boundary."""

    def __init__(self):
        self.state = "CENTER"
        self.pending = None
        self.pending_n = 0
        self.weak_n = 0  # frames spent in L/R state without re-crossing ENTER

    def update(self, yaw_rel):
        if self.state == "CENTER":
            raw = "LEFT" if yaw_rel <= -ENTER_LEFT_DEG else "RIGHT" if yaw_rel >= ENTER_RIGHT_DEG else "CENTER"
        else:
            # currently LEFT or RIGHT: only fall back to center inside EXIT band,
            # or switch side if it crosses the opposite ENTER threshold
            if abs(yaw_rel) < EXIT_DEG:
                raw = "CENTER"
            elif yaw_rel <= -ENTER_LEFT_DEG:
                raw = "LEFT"
            elif yaw_rel >= ENTER_RIGHT_DEG:
                raw = "RIGHT"
            else:
                raw = self.state
            # anti-stick: still in L/R but signal no longer crosses ENTER -> count
            strong = (self.state == "LEFT" and yaw_rel <= -ENTER_LEFT_DEG) or \
                     (self.state == "RIGHT" and yaw_rel >= ENTER_RIGHT_DEG)
            self.weak_n = 0 if strong else self.weak_n + 1
            if self.weak_n >= STUCK_FRAMES:
                raw = "CENTER"
                self.weak_n = 0
        if raw == self.state:
            self.pending = None
            self.pending_n = 0
        elif raw == self.pending:
            self.pending_n += 1
            if self.pending_n >= DEBOUNCE_N:
                self.state = raw
                self.pending = None
                self.pending_n = 0
        else:
            self.pending = raw
            self.pending_n = 1
        return self.state

def main():
    cap = cv2.VideoCapture(0)  # DSHOW backend gives black frames on this webcam
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    engine = GazeEstimationONNX(model_path=MODEL)
    detector = RetinaFace()

    baseline = None
    yaw_buf = deque(maxlen=SMOOTH_N)
    fsm = GazeStateMachine()
    bbox = None
    frame_i = 0
    fps_buf = deque(maxlen=30)

    testing = False
    trial_idx = 0
    target = None
    hold = 0
    trial_t0 = 0.0
    results = []

    while cap.isOpened():
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # --- detection (every N frames only) ---
        t_det = 0.0
        if frame_i % DETECT_EVERY == 0 or bbox is None:
            td = time.perf_counter()
            faces = detector.detect(frame)
            t_det = (time.perf_counter() - td) * 1000
            bbox = None
            for face in faces:
                b = face["bbox"] if isinstance(face, dict) else face.bbox
                bbox = list(map(int, b[:4]))
                break
        frame_i += 1

        # --- gaze inference on cached bbox ---
        yaw_deg = None
        t_inf = 0.0
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            x0, y0 = max(0, x0), max(0, y0)
            crop = frame[y0:y1, x0:x1]
            if crop.size > 0:
                ti = time.perf_counter()
                yaw, pitch = engine.estimate(crop)
                t_inf = (time.perf_counter() - ti) * 1000
                yaw_deg = math.degrees(yaw)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)

        label = "NO FACE"
        yaw_rel = None
        if yaw_deg is not None:
            yaw_buf.append(yaw_deg)
            yaw_s = float(np.median(yaw_buf))
            if baseline is not None:
                yaw_rel = -(yaw_s - baseline)  # mirror flip sign fix
                label = fsm.update(yaw_rel)
                if label == "CENTER":
                    # slow drift compensation: baseline follows posture changes
                    baseline = (1 - BASELINE_EMA) * baseline + BASELINE_EMA * yaw_s
            else:
                label = "PRESS 'c' LOOKING AT CENTER"

        # --- HUD ---
        fps_buf.append(time.perf_counter() - t0)
        fps = 1.0 / max(np.mean(fps_buf), 1e-6)
        color = {"LEFT": (255, 200, 0), "RIGHT": (0, 200, 255), "CENTER": (0, 255, 0)}.get(label, (0, 0, 255))
        cv2.putText(frame, label, (w // 2 - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, f"FPS {fps:5.1f}  gaze {t_inf:5.1f}ms  det {t_det:5.1f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if yaw_rel is not None:
            cx = w // 2
            cv2.line(frame, (cx - 150, 100), (cx + 150, 100), (200, 200, 200), 2)
            px = int(np.clip(yaw_rel, -30, 30) / 30 * 150)
            cv2.circle(frame, (cx + px, 100), 8, color, -1)
        cv2.putText(frame, "c=calibrate  t=20 trials  q=quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- trials ---
        if testing and baseline is not None:
            cv2.putText(frame, f"TRIAL {trial_idx+1}/{TRIALS}: LOOK {target}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            arrow = "<<<" if target == "LEFT" else ">>>"
            cv2.putText(frame, arrow, (w // 2 - 60, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
            hold = hold + 1 if label == target else 0
            done = None
            if hold >= HOLD_FRAMES:
                done = True
            elif time.perf_counter() - trial_t0 > TRIAL_TIMEOUT_S:
                done = False
            if done is not None:
                beep(1200 if done else 300, 150)  # high beep = OK, low = miss
                rt = time.perf_counter() - trial_t0
                results.append((target, done, rt))
                print(f"trial {trial_idx+1}: {target} -> {'OK' if done else 'MISS'} ({rt:.2f}s)")
                trial_idx += 1
                hold = 0
                if trial_idx >= TRIALS:
                    testing = False
                    ok = sum(1 for _, r, _ in results if r)
                    rts = [t for _, r, t in results if r]
                    print("\n===== FAST L/R RESULT =====")
                    print(f"accuracy : {ok}/{TRIALS} ({100*ok/TRIALS:.0f}%)")
                    if rts:
                        print(f"reaction : mean {np.mean(rts):.2f}s  median {np.median(rts):.2f}s")
                else:
                    target = random.choice(["LEFT", "RIGHT"])
                    trial_t0 = time.perf_counter()

        cv2.imshow("Fast L/R Gaze", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c') and yaw_deg is not None:
            baseline = float(np.median(yaw_buf))
            fsm = GazeStateMachine()
            beep(800, 100)
            print(f"[calibrated] baseline yaw = {baseline:+.1f}")
        elif k == ord('t') and baseline is not None and not testing:
            testing = True
            trial_idx = 0
            results = []
            hold = 0
            target = random.choice(["LEFT", "RIGHT"])
            trial_t0 = time.perf_counter()
            print("[trials started]")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

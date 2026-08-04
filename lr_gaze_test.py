# Left/Right gaze discrimination test
# Step 1: press 'c' while looking at screen center -> baseline yaw captured.
# Step 2: live LEFT/CENTER/RIGHT classification shown on screen.
# Step 3: press 't' -> 20 random L/R trials, accuracy report.
import cv2
import math
import random
import numpy as np
from collections import deque
from uniface import RetinaFace
from onnx_inference import GazeEstimationONNX

MODEL = "weights/resnet34_gaze.onnx"
THRESH_DEG = 6.0        # yaw offset from baseline to count as left/right
SMOOTH_N = 5            # frames of smoothing
TRIALS = 20
HOLD_FRAMES = 8         # frames the correct direction must be held to score a trial
TRIAL_TIMEOUT = 120     # frames per trial (~4s)

def classify(yaw_rel):
    if yaw_rel <= -THRESH_DEG:
        return "LEFT"
    if yaw_rel >= THRESH_DEG:
        return "RIGHT"
    return "CENTER"

def main():
    cap = cv2.VideoCapture(0)
    engine = GazeEstimationONNX(model_path=MODEL)
    detector = RetinaFace()

    baseline = None
    yaw_buf = deque(maxlen=SMOOTH_N)

    # trial state
    testing = False
    trial_idx = 0
    target = None
    hold = 0
    frames_left = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        yaw_deg = None
        faces = detector.detect(frame)
        for face in faces:
            bbox = face["bbox"] if isinstance(face, dict) else face.bbox
            x0, y0, x1, y1 = map(int, bbox[:4])
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            yaw, pitch = engine.estimate(crop)
            yaw_deg = math.degrees(yaw)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)
            break

        label = "NO FACE"
        yaw_rel = None
        if yaw_deg is not None:
            yaw_buf.append(yaw_deg)
            yaw_s = float(np.mean(yaw_buf))
            if baseline is not None:
                # mirror-flipped frame: measured yaw sign is inverted vs user direction
                yaw_rel = -(yaw_s - baseline)
                label = classify(yaw_rel)
            else:
                label = "PRESS 'c' LOOKING AT CENTER"

        # ==== live HUD ====
        color = {"LEFT": (255, 200, 0), "RIGHT": (0, 200, 255), "CENTER": (0, 255, 0)}.get(label, (0, 0, 255))
        cv2.putText(frame, label, (w // 2 - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        if yaw_rel is not None:
            # bar meter
            cx = w // 2
            cv2.line(frame, (cx - 150, 100), (cx + 150, 100), (200, 200, 200), 2)
            px = int(np.clip(yaw_rel, -30, 30) / 30 * 150)
            cv2.circle(frame, (cx + px, 100), 8, color, -1)
            cv2.putText(frame, f"rel yaw {yaw_rel:+.1f}", (cx - 60, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "c=calibrate center  t=start 20 trials  q=quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ==== trial mode ====
        if testing and baseline is not None:
            cv2.putText(frame, f"TRIAL {trial_idx+1}/{TRIALS}: LOOK {target}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            arrow = "<<<" if target == "LEFT" else ">>>"
            cv2.putText(frame, arrow, (w // 2 - 60, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
            frames_left -= 1
            if label == target:
                hold += 1
            else:
                hold = 0
            done = None
            if hold >= HOLD_FRAMES:
                done = True
            elif frames_left <= 0:
                done = False
            if done is not None:
                results.append((target, done))
                print(f"trial {trial_idx+1}: target={target} -> {'OK' if done else 'MISS'}")
                trial_idx += 1
                hold = 0
                if trial_idx >= TRIALS:
                    testing = False
                    ok = sum(1 for _, r in results if r)
                    lt = [r for t, r in results if t == "LEFT"]
                    rt = [r for t, r in results if t == "RIGHT"]
                    print("\n===== L/R TEST RESULT =====")
                    print(f"total : {ok}/{TRIALS}  ({100*ok/TRIALS:.0f}%)")
                    print(f"LEFT  : {sum(lt)}/{len(lt)}")
                    print(f"RIGHT : {sum(rt)}/{len(rt)}")
                else:
                    target = random.choice(["LEFT", "RIGHT"])
                    frames_left = TRIAL_TIMEOUT

        cv2.imshow("L/R Gaze Test", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c') and yaw_deg is not None:
            baseline = float(np.mean(yaw_buf))
            print(f"[calibrated] baseline yaw = {baseline:+.1f}")
        elif k == ord('t') and baseline is not None and not testing:
            testing = True
            trial_idx = 0
            results = []
            hold = 0
            target = random.choice(["LEFT", "RIGHT"])
            frames_left = TRIAL_TIMEOUT
            print("[trials started]")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

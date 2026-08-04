# 45-degree gaze direction verification test
# Guides the user to look in 5 directions, records model yaw/pitch, reports stats.
import cv2
import math
import numpy as np
from uniface import RetinaFace
from onnx_inference import GazeEstimationONNX

MODEL = "weights/resnet34_gaze.onnx"

# (name, expected yaw deg, expected pitch deg)  yaw: + = right(screen), pitch: + = up
STAGES = [
    ("CENTER",      0,   0),
    ("LEFT 45",   -45,   0),
    ("RIGHT 45",   45,   0),
    ("UP 45",       0,  45),
    ("DOWN 45",     0, -45),
]
SAMPLES_PER_STAGE = 45  # ~1.5s at 30fps

def main():
    cap = cv2.VideoCapture(0)
    engine = GazeEstimationONNX(model_path=MODEL)
    detector = RetinaFace()

    stage_idx = 0
    recording = False
    samples = []
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        yaw_deg = pitch_deg = None
        faces = detector.detect(frame)
        for face in faces:
            bbox = face["bbox"] if isinstance(face, dict) else face.bbox
            x0, y0, x1, y1 = map(int, bbox[:4])
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            yaw, pitch = engine.estimate(crop)
            yaw_deg, pitch_deg = math.degrees(yaw), math.degrees(pitch)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            L = x1 - x0
            dx = int(-L * math.sin(yaw) * math.cos(pitch))
            dy = int(-L * math.sin(pitch))
            cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (0, 0, 255), 2, tipLength=0.25)
            break

        if stage_idx < len(STAGES):
            name, ey, ep = STAGES[stage_idx]
            cv2.putText(frame, f"[{stage_idx+1}/5] Look {name} deg  (SPACE=record, q=quit)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if yaw_deg is not None:
            cv2.putText(frame, f"yaw={yaw_deg:+.1f}  pitch={pitch_deg:+.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "no face", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if recording:
            if yaw_deg is not None:
                samples.append((yaw_deg, pitch_deg))
            cv2.putText(frame, f"REC {len(samples)}/{SAMPLES_PER_STAGE}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(samples) >= SAMPLES_PER_STAGE:
                arr = np.array(samples)
                name, ey, ep = STAGES[stage_idx]
                results.append((name, ey, ep, arr[:, 0].mean(), arr[:, 0].std(),
                                arr[:, 1].mean(), arr[:, 1].std(), len(arr)))
                print(f"[{name}] yaw mean {arr[:,0].mean():+.1f} (std {arr[:,0].std():.1f}), "
                      f"pitch mean {arr[:,1].mean():+.1f} (std {arr[:,1].std():.1f})")
                samples = []
                recording = False
                stage_idx += 1
                if stage_idx >= len(STAGES):
                    break

        cv2.imshow("45deg Gaze Test", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord(' ') and not recording and stage_idx < len(STAGES):
            recording = True
            samples = []

    cap.release()
    cv2.destroyAllWindows()

    if results:
        print("\n===== RESULT SUMMARY =====")
        print(f"{'stage':10s} {'exp(yaw,pitch)':>15s} {'meas yaw':>12s} {'meas pitch':>12s}")
        for name, ey, ep, my, sy, mp, sp, n in results:
            print(f"{name:10s} {f'({ey:+d},{ep:+d})':>15s} {my:+8.1f}±{sy:4.1f} {mp:+8.1f}±{sp:4.1f}")
        # Separability check vs center
        base = next((r for r in results if r[0] == "CENTER"), None)
        if base and len(results) == 5:
            print("\nSeparation from CENTER (>3 sigma = reliably distinguishable):")
            for name, ey, ep, my, sy, mp, sp, n in results[1:]:
                axis_val, axis_std, base_val = (my, sy, base[3]) if ey != 0 else (mp, sp, base[5])
                delta = abs(axis_val - base_val)
                sigma = max(axis_std, 1e-6)
                print(f"  {name:10s} delta={delta:5.1f} deg  ({delta/sigma:.1f} sigma)")

if __name__ == "__main__":
    main()

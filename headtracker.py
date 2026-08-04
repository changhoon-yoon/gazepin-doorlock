"""HeadTracker — 고개 방향(좌/우/상/하) + 깜빡임 추적 재사용 모듈.

engine_headpose.py(도어락 엔진)에서 추적 부분만 떼어낸 단일 파일 라이브러리.
다른 프로젝트에서는 이 파일과 models/face_landmarker.task 만 복사하면 된다.
의존성: opencv-python, numpy, mediapipe

사용 예:
    from headtracker import HeadTracker

    tracker = HeadTracker("models/face_landmarker.task")
    cap = cv2.VideoCapture(0)
    while True:
        ok, frame = cap.read()
        state = tracker.update(frame)          # 프레임 1장 처리
        if state.ok:
            print(state.direction, state.yaw, state.pitch)  # LEFT/RIGHT/UP/DOWN/CENTER
        if state.blink:
            print("blink!")
        # 사용자가 정면을 볼 때 한 번:
        # tracker.calibrate()   # 현재 자세(최근 0.5초 중앙값)를 중립 기준으로

데모 실행:  python headtracker.py [--cam 0]
"""
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

BLINK_ON, BLINK_OFF = 0.35, 0.25  # 깜빡임 이중 문턱값 (감김 판정 / 뜸 판정)


@dataclass
class HeadState:
    """update() 1회의 결과."""
    ok: bool = False               # 얼굴 감지 여부
    direction: str = "CENTER"      # LEFT / RIGHT / UP / DOWN / CENTER
    yaw: float = 0.0               # 도 단위, 보정(중립 기준) 적용 후
    pitch: float = 0.0
    blink: bool = False            # 이번 프레임에서 깜빡임 이벤트 발생
    blink_score: float = 0.0       # 0(뜸)~1(감김)
    face_px: int = 0               # 얼굴 폭 픽셀 (거리 추정용)
    bbox: tuple = field(default=None)  # (x0, y0, x1, y1) 또는 None


class HeadTracker:
    """MediaPipe FaceLandmarker 기반 고개 방향·깜빡임 추적기.

    - 방향: facial transformation matrix의 얼굴 전방 벡터에서 yaw/pitch 추출
    - 깜빡임: 양눈 blendshape의 min + 이중 문턱 (고개 돌림 오탐 차단)
    - 보정: calibrate() 호출 시점의 최근 0.5초 자세 중앙값을 중립(0°)으로
    """

    def __init__(self, model_path, thr_deg=12.0, up_thr_deg=8.0, down_thr_deg=10.0,
                 yaw_sign=-1.0, pitch_sign=1.0, mirror=True):
        """
        thr_deg / up_thr_deg / down_thr_deg: 방향 판정 임계각(도).
        yaw_sign / pitch_sign: 좌표 규약이 환경에 따라 다를 때 ±1로 부호 교정.
        mirror: True면 입력 프레임을 좌우 반전(거울 모드) 후 처리.
        """
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
            ))
        self.thr_deg = thr_deg
        self.up_thr_deg = up_thr_deg
        self.down_thr_deg = down_thr_deg
        self.yaw_sign = yaw_sign
        self.pitch_sign = pitch_sign
        self.mirror = mirror
        self._t0 = time.monotonic()
        self._prev_ts = -1
        self._was_closed = False
        self._yaw_off = 0.0
        self._pitch_off = 0.0
        self.calibrated = False
        self._raw_buf = deque(maxlen=8)  # 최근 원시 (yaw, pitch) — calibrate()용

    # ---------- 내부 ----------

    def _label(self, pitch, yaw):
        thr = math.sin(math.radians(self.thr_deg))
        up = math.sin(math.radians(self.up_thr_deg))
        down = math.sin(math.radians(self.down_thr_deg))
        dx = -math.sin(yaw) * math.cos(pitch)
        dy = -math.sin(pitch)
        if dx < -thr:
            return "LEFT"
        if dx > thr:
            return "RIGHT"
        if dy < -up:
            return "UP"
        if dy > down:
            return "DOWN"
        return "CENTER"

    @staticmethod
    def _pose_from_matrix(mat):
        r = np.asarray(mat)[:3, :3]
        f = r[:, 2]  # 얼굴 전방 벡터
        yaw = math.atan2(f[0], abs(f[2]) + 1e-9)
        pitch = math.atan2(f[1], math.hypot(f[0], f[2]) + 1e-9)
        return yaw, pitch

    # ---------- 공개 API ----------

    def update(self, frame_bgr):
        """BGR 프레임 1장을 처리하고 HeadState를 돌려준다."""
        st = HeadState()
        if frame_bgr is None:
            return st
        if self.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts = max(self._prev_ts + 1, int((time.monotonic() - self._t0) * 1000))
        self._prev_ts = ts
        res = self._landmarker.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)
        if not res.face_landmarks:
            return st
        st.ok = True

        # 깜빡임 (양눈 min + 이중 문턱)
        shapes = {c.category_name: c.score for c in res.face_blendshapes[0]}
        st.blink_score = min(shapes.get("eyeBlinkRight", 0.0), shapes.get("eyeBlinkLeft", 0.0))
        if not self._was_closed and st.blink_score > BLINK_ON:
            self._was_closed = True
            st.blink = True
        elif self._was_closed and st.blink_score < BLINK_OFF:
            self._was_closed = False

        # 고개 방향
        if res.facial_transformation_matrixes:
            y_raw, p_raw = self._pose_from_matrix(res.facial_transformation_matrixes[0])
            raw = (self.yaw_sign * y_raw, self.pitch_sign * p_raw)
            self._raw_buf.append(raw)
            yaw = raw[0] - self._yaw_off
            pitch = raw[1] - self._pitch_off
            st.yaw, st.pitch = math.degrees(yaw), math.degrees(pitch)
            st.direction = self._label(pitch, yaw)

        # 얼굴 박스
        h, w = rgb.shape[:2]
        xs = [lm.x for lm in res.face_landmarks[0]]
        ys = [lm.y for lm in res.face_landmarks[0]]
        st.bbox = (int(min(xs) * w), int(min(ys) * h), int(max(xs) * w), int(max(ys) * h))
        st.face_px = st.bbox[2] - st.bbox[0]
        return st

    def calibrate(self):
        """최근 자세(약 0.5초 중앙값)를 중립 기준(0°)으로 설정. 성공 여부 반환."""
        if len(self._raw_buf) < 3:
            return False
        self._yaw_off = float(np.median([a[0] for a in self._raw_buf]))
        self._pitch_off = float(np.median([a[1] for a in self._raw_buf]))
        self.calibrated = True
        return True

    def reset_calibration(self):
        self._yaw_off = self._pitch_off = 0.0
        self.calibrated = False

    def flip_yaw(self):
        """좌우가 반대로 잡힐 때 1회 호출로 교정."""
        self.yaw_sign = -self.yaw_sign

    def flip_pitch(self):
        self.pitch_sign = -self.pitch_sign

    def close(self):
        self._landmarker.close()


# ---------- 단독 실행 데모 ----------

def _demo():
    import argparse
    ap = argparse.ArgumentParser(description="HeadTracker 데모 — c=보정, r=해제, x/y=부호반전, ESC=종료")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "models", "face_landmarker.task"))
    args = ap.parse_args()

    tracker = HeadTracker(args.model)
    cap = cv2.VideoCapture(args.cam)
    blinks = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        st = tracker.update(frame)
        view = cv2.flip(frame, 1)
        if st.ok:
            x0, y0, x1, y1 = st.bbox
            cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
            if st.blink:
                blinks += 1
            cv2.putText(view, f"{st.direction}  yaw {st.yaw:+.0f} pitch {st.pitch:+.0f}  blink x{blinks}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(view, "calibrated" if tracker.calibrated else "press 'c' while facing camera",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow("HeadTracker demo", view)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("c"):
            tracker.calibrate()
        if key == ord("r"):
            tracker.reset_calibration()
        if key == ord("x"):
            tracker.flip_yaw()
        if key == ord("y"):
            tracker.flip_pitch()
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    _demo()

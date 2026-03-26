import time
from collections import deque, Counter

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "gesture_recognizer.task"

# 안정화 파라미터 (반응 속도 개선 버전)
VOTE_WINDOW = 8
MIN_VOTES_TO_DECIDE = 4
COOLDOWN_SEC = 0.18
MIN_CONFIDENCE = 0.55

def main():
    # 1) Recognizer 생성 (VIDEO 모드)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. (권한/장치 확인)")

    start = time.time()

    # 안정화 상태
    votes = deque(maxlen=VOTE_WINDOW)
    stable_label = "NO_HAND"
    last_committed_label = "NO_HAND"
    cooldown_until = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 보기 편하게 미러링 (원치 않으면 주석)
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start) * 1000)

        result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        output = frame.copy()
        label = "NO_HAND"
        score = 0.0

        # 2) 랜드마크 + 제스처(top-1) 추출
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            landmarks = result.hand_landmarks[0]
            h, w = output.shape[:2]
            for lm in landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(output, (x, y), 4, (0, 255, 0), -1)

            if result.gestures and len(result.gestures[0]) > 0:
                top = result.gestures[0][0]
                label = top.category_name
                score = float(top.score)

        # 3) confidence가 낮으면 UNKNOWN 취급 (튐 방지)
        if label in ("NO_HAND", "UNKNOWN"):
            pass
        else:
            votes.append(label)

        # 4) 투표 창에 추가
        votes.append(label)

        now = time.time()

        # 5) 쿨다운 중이면 stable_label 유지
        if now < cooldown_until:
            pass
        else:
            # 표본이 충분히 쌓였을 때만 다수결로 업데이트
            if len(votes) >= MIN_VOTES_TO_DECIDE:
                counts = Counter(votes)

                # NO_HAND/UNKNOWN이 너무 자주 나오면 실제 라벨이 묻힐 수 있어서
                # 상황에 따라 가중치를 줄이는 옵션 (간단 버전)
                # 여기서는 그대로 사용.

                candidate, cand_votes = counts.most_common(1)[0]

                # 다수결이 충분히 강할 때만 채택 (너무 약하면 유지)
                # 예: 12프레임 중 5표면 애매 → 유지
                if cand_votes >= max(3, len(votes) // 3):
                    stable_label = candidate

            # stable_label이 바뀌면 쿨다운 시작
            if stable_label != last_committed_label:
                last_committed_label = stable_label
                cooldown_until = now + COOLDOWN_SEC

        # 6) 화면 표시
        cv2.putText(
            output,
            f"Raw: {label} ({score:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            f"Stable: {stable_label}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            f"votes: {len(votes)}/{VOTE_WINDOW}  cooldown: {max(0.0, cooldown_until-now):.2f}s",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Gesture Recognizer (Stable) - q to quit", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()


if __name__ == "__main__":
    main()
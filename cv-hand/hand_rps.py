import time
from collections import deque, Counter

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "gesture_recognizer.task"

# --- RPS 매핑 (MediaPipe 기본 제스처 라벨) ---
MP_TO_RPS = {
    "Closed_Fist": "ROCK",
    "Open_Palm": "PAPER",
    "Victory": "SCISSORS",
}

# --- 안정화 파라미터 (빠르게 반응 + 덜 튐) ---
VOTE_WINDOW = 8
MIN_VOTES_TO_DECIDE = 4
COOLDOWN_SEC = 0.18
MIN_CONFIDENCE = 0.55

# 연속 프레임 같은 라벨이면 빠르게 확정
STREAK_TO_COMMIT = 3


def majority_vote(votes: deque) -> str:
    if len(votes) == 0:
        return "NO_HAND"
    c = Counter(votes)
    return c.most_common(1)[0][0]


def rps_winner(a: str, b: str) -> str:
    """a vs b 결과를 문자열로 반환"""
    if a == "NONE" or b == "NONE":
        return "SHOW BOTH HANDS"
    if a == b:
        return "DRAW"
    win = {
        ("ROCK", "SCISSORS"),
        ("SCISSORS", "PAPER"),
        ("PAPER", "ROCK"),
    }
    return "LEFT WINS" if (a, b) in win else "RIGHT WINS"


def main():
    # 1) Recognizer 생성 (2손, VIDEO 모드)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다. (권한/장치 확인)")

    start = time.time()

    # 각 '손 슬롯'(0,1)에 대한 안정화 상태
    votes = [deque(maxlen=VOTE_WINDOW), deque(maxlen=VOTE_WINDOW)]
    stable = ["NO_HAND", "NO_HAND"]
    last_committed = ["NO_HAND", "NO_HAND"]
    cooldown_until = [0.0, 0.0]
    streak_label = [None, None]
    streak_count = [0, 0]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # 거울모드
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start) * 1000)

        result = recognizer.recognize_for_video(mp_image, timestamp_ms)

        output = frame.copy()
        h, w = output.shape[:2]

        # 이번 프레임 raw 라벨(손 슬롯별)
        raw_label = ["NO_HAND", "NO_HAND"]
        raw_score = [0.0, 0.0]

        # 2) 감지된 손들을 슬롯 0..n에 채움 (result 리스트 순서 그대로)
        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks[:2]):
                # 랜드마크 점 표시
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(output, (x, y), 4, (0, 255, 0), -1)

                # 제스처(top-1)
                label = "UNKNOWN"
                score = 0.0
                if result.gestures and len(result.gestures) > i and len(result.gestures[i]) > 0:
                    top = result.gestures[i][0]
                    label = top.category_name
                    score = float(top.score)

                # confidence 낮으면 UNKNOWN
                if score < MIN_CONFIDENCE:
                    label = "UNKNOWN"

                raw_label[i] = label
                raw_score[i] = score

        now = time.time()

        # 3) 손 슬롯별 안정화 업데이트
        for i in range(2):
            lbl = raw_label[i]

            votes[i].append(lbl)

            # streak 계산
            if lbl == streak_label[i]:
                streak_count[i] += 1
            else:
                streak_label[i] = lbl
                streak_count[i] = 1

            if now < cooldown_until[i]:
                continue

            # 빠른 확정 (연속 n프레임)
            if streak_label[i] not in ("UNKNOWN", "NO_HAND") and streak_count[i] >= STREAK_TO_COMMIT:
                stable[i] = streak_label[i]
            else:
                # 다수결 확정
                if len(votes[i]) >= MIN_VOTES_TO_DECIDE:
                    cand = majority_vote(votes[i])
                    # 너무 약한 다수결이면 유지 (튜는 것 방지)
                    cand_votes = Counter(votes[i])[cand]
                    if cand_votes >= max(3, len(votes[i]) // 3):
                        stable[i] = cand

            # 라벨이 바뀌면 쿨다운 시작
            if stable[i] != last_committed[i]:
                last_committed[i] = stable[i]
                cooldown_until[i] = now + COOLDOWN_SEC

        # 4) RPS로 변환 (stable 기준)
        left_rps = MP_TO_RPS.get(stable[0], "NONE")
        right_rps = MP_TO_RPS.get(stable[1], "NONE")

        # 결과
        outcome = rps_winner(left_rps, right_rps)

        # 5) 화면 텍스트 표시
        cv2.putText(output, f"Left raw: {raw_label[0]} ({raw_score[0]:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(output, f"Right raw: {raw_label[1]} ({raw_score[1]:.2f})",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.putText(output, f"Left: {left_rps}   Right: {right_rps}",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)

        cv2.putText(output, f"Result: {outcome}",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 255, 0), 3)

        cv2.imshow("RPS (Two Hands) - q to quit", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()


if __name__ == "__main__":
    main()
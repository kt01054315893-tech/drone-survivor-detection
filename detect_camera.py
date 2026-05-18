from ultralytics import YOLO
import cv2

# ✅ 훈련된 모델 경로 (학습 완료 후 생성된 best.pt 사용)
# 아직 학습 중이라면 기본 모델로 테스트 가능
MODEL_PATH = r"C:\Users\yesyo\runs\detect\train\weights\best.pt"
# 학습 전 테스트용: MODEL_PATH = "yolov8s.pt"

# 모델 로드
model = YOLO(MODEL_PATH)

# 노트북 기본 카메라 (0번)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("✅ 카메라 실행 중... 'q' 키를 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    # YOLO 탐지 실행
    results = model(frame, conf=0.5)  # conf: 신뢰도 임계값 (0.5 = 50% 이상만 표시)

    # 탐지 결과를 프레임에 그리기
    annotated_frame = results[0].plot()

    # 탐지된 객체 수 화면에 표시
    count = len(results[0].boxes)
    cv2.putText(annotated_frame, f"탐지된 조난자: {count}명",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 화면 출력
    cv2.imshow("조난자 탐지 시스템", annotated_frame)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ 종료되었습니다.")

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("agh_src11_hrc0.avi")

line_y = 300
car_positions = {}
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 2:  # car
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])
                center_y = int((y1 + y2) / 2)

                if track_id not in car_positions:
                    car_positions[track_id] = center_y
                else:
                    previous_y = car_positions[track_id]

                    if previous_y < line_y and center_y >= line_y:
                        counter += 1

                    elif previous_y > line_y and center_y <= line_y:
                        counter -= 1

                    car_positions[track_id] = center_y

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.line(frame, (0,line_y), (frame.shape[1],line_y), (0,0,255), 2)
    cv2.putText(frame, f"Cars: {counter}", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Parking Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
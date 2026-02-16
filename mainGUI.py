import cv2
import threading
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from ultralytics import YOLO

# -----------------------------
# Simple Centroid Tracker
# -----------------------------
class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []
        new_center_points = {}

        for rect in objects_rect:
            x, y, w, h = rect
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = np.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    new_center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                new_center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        self.center_points = new_center_points.copy()
        return objects_bbs_ids


# -----------------------------
# Parking GUI
# -----------------------------
class ParkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parking Counter")

        self.car_count = 0
        self.max_capacity = 10

        self.label = tk.Label(root, text="0", font=("Arial", 60), fg="blue")
        self.label.pack(pady=20)

        self.capacity_label = tk.Label(
            root, text=f"Max Capacity: {self.max_capacity}",
            font=("Arial", 14)
        )
        self.capacity_label.pack()

        self.settings_button = tk.Button(
            root, text="⚙ Settings", command=self.change_capacity
        )
        self.settings_button.pack(pady=10)

        self.status_label = tk.Label(root, text="", font=("Arial", 16))
        self.status_label.pack()

        self.update_display()

    def update_display(self):
        self.label.config(text=str(self.max_capacity - self.car_count))
        self.capacity_label.config(
            text=f"Max Capacity: {self.max_capacity}"
        )

        if self.car_count >= self.max_capacity:
            self.status_label.config(text="PARKING FULL", fg="red")
        else:
            self.status_label.config(text="AVAILABLE", fg="green")

    def car_entered(self):
        if self.car_count < self.max_capacity:
            self.car_count += 1
        self.update_display()

    def car_left(self):
        if self.car_count > 0:
            self.car_count -= 1
        self.update_display()

    def change_capacity(self):
        new_capacity = simpledialog.askinteger(
            "Settings",
            "Enter maximum parking capacity:",
            minvalue=1
        )
        if new_capacity:
            self.max_capacity = new_capacity
            self.update_display()

# -----------------------------
# MAIN
# -----------------------------
root = tk.Tk()
app = ParkingApp(root)

cap = cv2.VideoCapture("agh_src4_hrc0.avi")
model = YOLO("yolov8n.pt")
tracker = Tracker()
line_position = 300
car_positions = {}

def process_frame():
    global cap

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    results = model(frame)[0]
    detections = []

    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r

        if int(class_id) == 2 and score > 0.5:
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)
            detections.append([x, y, w, h])

    boxes_ids = tracker.update(detections)

    cv2.line(frame, (0, line_position),
             (frame.shape[1], line_position),
             (0, 255, 255), 2)

    for box in boxes_ids:
        x, y, w, h, id = box
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        cv2.rectangle(frame, (x, y),
                      (x + w, y + h),
                      (255, 0, 0), 2)

        if id not in car_positions:
            car_positions[id] = cy

        previous_y = car_positions[id]

        if previous_y < line_position and cy >= line_position:
            app.car_entered()

        if previous_y > line_position and cy <= line_position:
            app.car_left()

        car_positions[id] = cy

    cv2.imshow("Parking Detection", frame)

    if cv2.waitKey(1) != 27:
        root.after(10, process_frame)
    else:
        cap.release()
        cv2.destroyAllWindows()
        root.quit()

process_frame()
root.mainloop()
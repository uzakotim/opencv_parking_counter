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
        self.root.title("Smart Parking System")
        self.root.attributes("-fullscreen", True)
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        self.root.geometry(f"{width}x{height}+0+0")
        self.root.configure(bg="#0f172a")  # dark navy
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

        self.car_count = 0
        self.max_capacity = 10

        # ===== TITLE =====
        self.title_label = tk.Label(
            root,
            text="SMART PARKING",
            font=("Helvetica", 18, "bold"),
            fg="white",
            bg="#0f172a"
        )
        self.title_label.pack(pady=(30, 10))

        # ===== CARD FRAME =====
        self.card = tk.Frame(root, bg="#1e293b", bd=0)
        self.card.pack(padx=30, pady=20, fill="both", expand=True)

        # ===== AVAILABLE TEXT =====
        self.available_text = tk.Label(
            self.card,
            text="Available Spots",
            font=("Helvetica", 14),
            fg="#94a3b8",
            bg="#1e293b"
        )
        self.available_text.pack(pady=(40, 10))

        # ===== BIG NUMBER =====
        self.counter_label = tk.Label(
            self.card,
            text="0",
            font=("Helvetica", 360, "bold"),
            fg="#22c55e",
            bg="#1e293b"
        )
        self.counter_label.pack()

        # ===== STATUS BADGE =====
        self.status_label = tk.Label(
            self.card,
            text="AVAILABLE",
            font=("Helvetica", 120, "bold"),
            fg="white",
            bg="#22c55e",
            padx=15,
            pady=5
        )
        self.status_label.pack(pady=20)

        # ===== CAPACITY INFO =====
        self.capacity_label = tk.Label(
            root,
            text="Capacity: 10",
            font=("Helvetica", 11),
            fg="#94a3b8",
            bg="#0f172a"
        )
        self.capacity_label.pack()

        # ===== SETTINGS BUTTON =====
        self.settings_button = tk.Button(
            root,
            text="⚙ Settings",
            font=("Helvetica", 11, "bold"),
            fg="#100625",
            bg="#3b82f6",
            activebackground="#2563eb",
            activeforeground="white",
            relief="flat",
            padx=20,
            pady=8,
            command=self.change_capacity
        )
        self.settings_button.pack(pady=20)

        self.update_display()

    # =============================
    def update_display(self):
        available = self.max_capacity - self.car_count

        self.counter_label.config(text=str(available))
        self.capacity_label.config(text=f"Capacity: {self.max_capacity}")

        if self.car_count >= self.max_capacity:
            self.status_label.config(
                text="FULL",
                bg="#ef4444"
            )
            self.counter_label.config(fg="#ef4444")
        else:
            self.status_label.config(
                text="AVAILABLE",
                bg="#22c55e"
            )
            self.counter_label.config(fg="#22c55e")

    # =============================
    def car_entered(self):
        if self.car_count < self.max_capacity:
            self.car_count += 1
        self.update_display()

    # =============================
    def car_left(self):
        if self.car_count > 0:
            self.car_count -= 1
        self.update_display()

    # =============================
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
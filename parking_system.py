import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# 1. Load Model YOLOv8 (nano version agar cepat)
model = YOLO('yolov8n.pt') 

# 2. Definisikan Titik Slot Parkir (Gunakan hasil dari Langkah 2)
# Area parkir 1
parkir1_1 = [(38, 162), (66, 205), (129, 207), (87, 160)]
parkir1_2 = [(129, 207), (87, 160), (139, 158), (183, 207)]
parkir1_3 = [(139, 158), (183, 207), (233, 205), (192, 161)]
parkir1_4 = [(233, 205), (192, 161), (247, 158), (287, 207)]
parkir1_5 = [(247, 158), (287, 207), (342, 201), (299, 160)]
parkir1_6 = [(342, 201), (299, 160), (352, 161), (395, 206)]
parkir1_7 = [(352, 161), (395, 206), (446, 206), (402, 158)]
parkir1_8 = [(446, 206), (402, 158), (455, 159), (504, 208)]
parkir1_9 = [(455, 159), (504, 208), (557, 202), (516, 160)]
parkir1_10 = [(557, 202), (516, 160), (566, 159), (610, 203)]
parkir1_11 = [(566, 159), (610, 203), (665, 206), (620, 157)]
parkir1_12 = [(665, 206), (620, 157), (674, 161), (715, 203)]
parkir1_13 = [(674, 161), (715, 203), (767, 203), (723, 160)]
parkir1_14 = [(767, 203), (723, 160), (779, 161), (816, 202)]
parkir1_15 = [(779, 161), (816, 202), (872, 206), (826, 161)]
parkir1_16 = [(872, 206), (826, 161), (879, 162), (924, 206)]
parkir1_17 = [(879, 162), (924, 206), (972, 206), (926, 156)]
parkir1_18 = [(972, 206), (926, 156), (985, 159), (1014, 194)]
# Area parkir 2
parkir2_1 = [(18, 310), (82, 377), (126, 374), (65, 312)]
parkir2_2 = [(126, 374), (65, 312), (108, 312), (172, 368)]
parkir2_3 = [(108, 312), (172, 368), (222, 368), (158, 313)]
parkir2_4 = [(222, 368), (158, 313), (206, 313), (275, 374)]
parkir2_5 = [(206, 313), (275, 374), (318, 366), (258, 314)]
parkir2_6 = [(318, 366), (258, 314), (307, 317), (368, 373)]
parkir2_7 = [(307, 317), (368, 373), (425, 376), (355, 318)]
parkir2_8 = [(425, 376), (355, 318), (406, 313), (475, 373)]
parkir2_9 = [(406, 313), (475, 373), (518, 372), (460, 316)]
parkir2_10 = [(518, 372), (460, 316), (507, 315), (572, 376)]
parkir2_11 = [(507, 315), (572, 376), (631, 373), (560, 313)]
parkir2_12 = [(631, 373), (560, 313), (611, 313), (674, 374)]
parkir2_13 = [(611, 313), (674, 374), (735, 372), (663, 316)]
parkir2_14 = [(735, 372), (663, 316), (715, 314), (778, 372)]
parkir2_15 = [(715, 314), (778, 372), (829, 372), (763, 318)]
parkir2_16 = [(829, 372), (763, 318), (814, 315), (879, 375)]
parkir2_17 = [(814, 315), (879, 375), (906, 355), (873, 316)]

# Masukkan semua area ke dalam list
parking_slots = [
    #parkir area 1
    {"id": 1, "points": np.array(parkir1_1, np.int32), "status": "Free"},
    {"id": 2, "points": np.array(parkir1_2, np.int32), "status": "Free"},
    {"id": 3, "points": np.array(parkir1_3, np.int32), "status": "Free"},
    {"id": 4, "points": np.array(parkir1_4, np.int32), "status": "Free"},
    {"id": 5, "points": np.array(parkir1_5, np.int32), "status": "Free"},
    {"id": 6, "points": np.array(parkir1_6, np.int32), "status": "Free"},
    {"id": 7, "points": np.array(parkir1_7, np.int32), "status": "Free"},
    {"id": 8, "points": np.array(parkir1_8, np.int32), "status": "Free"},
    {"id": 9, "points": np.array(parkir1_9, np.int32), "status": "Free"},
    {"id": 10, "points": np.array(parkir1_10, np.int32), "status": "Free"},
    {"id": 11, "points": np.array(parkir1_11, np.int32), "status": "Free"},
    {"id": 12, "points": np.array(parkir1_12, np.int32), "status": "Free"},
    {"id": 13, "points": np.array(parkir1_13, np.int32), "status": "Free"},
    {"id": 14, "points": np.array(parkir1_14, np.int32), "status": "Free"},
    {"id": 15, "points": np.array(parkir1_15, np.int32), "status": "Free"},
    {"id": 16, "points": np.array(parkir1_16, np.int32), "status": "Free"},
    {"id": 17, "points": np.array(parkir1_17, np.int32), "status": "Free"},
    {"id": 18, "points": np.array(parkir1_18, np.int32), "status": "Free"},
    #parkir area 2
    {"id": 19, "points": np.array(parkir2_1, np.int32), "status": "Free"},
    {"id": 20, "points": np.array(parkir2_2, np.int32), "status": "Free"},
    {"id": 21, "points": np.array(parkir2_3, np.int32), "status": "Free"},
    {"id": 22, "points": np.array(parkir2_4, np.int32), "status": "Free"},
    {"id": 23, "points": np.array(parkir2_5, np.int32), "status": "Free"},
    {"id": 24, "points": np.array(parkir2_6, np.int32), "status": "Free"},
    {"id": 25, "points": np.array(parkir2_7, np.int32), "status": "Free"},
    {"id": 26, "points": np.array(parkir2_8, np.int32), "status": "Free"},
    {"id": 27, "points": np.array(parkir2_9, np.int32), "status": "Free"},
    {"id": 28, "points": np.array(parkir2_10, np.int32), "status": "Free"},
    {"id": 29, "points": np.array(parkir2_11, np.int32), "status": "Free"},
    {"id": 30, "points": np.array(parkir2_12, np.int32), "status": "Free"},
    {"id": 31, "points": np.array(parkir2_13, np.int32), "status": "Free"},
    {"id": 32, "points": np.array(parkir2_14, np.int32), "status": "Free"},
    {"id": 33, "points": np.array(parkir2_15, np.int32), "status": "Free"},
    {"id": 34, "points": np.array(parkir2_16, np.int32), "status": "Free"},
    {"id": 35, "points": np.array(parkir2_17, np.int32), "status": "Free"},
]

# Kelas yang ingin dideteksi (COCO dataset: 2=car, 3=motorcycle, 5=bus, 7=truck)
target_classes = [2, 3, 5, 7]

cap = cv2.VideoCapture('Parking.mp4') # Ganti video Anda

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Resize agar sesuai dengan koordinat yang Anda ambil di Langkah 2
    frame = cv2.resize(frame, (1020, 500))

    # Deteksi Object dengan YOLO
    results = model.predict(frame, conf=0.25, verbose=False) # conf=threshold confidence
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    # Reset status setiap frame
    occupied_count = 0
    for slot in parking_slots:
        slot["status"] = "Free"

    # Loop setiap objek yang terdeteksi
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5]) # class ID
        
        # Cari titik tengah mobil (center point)
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2

        box_height = y2 - y1
        quarter_offset = box_height // 4

        # Kumpulkan titik-titik yang akan diuji:
        test_points = [
            (cx, cy), 
            (cx, y1 + quarter_offset), 
            (cx, y2 - quarter_offset) 
        ]

        if d in target_classes:
            for slot in parking_slots:
                is_occupied = False
                for point in test_points:
                    result = cv2.pointPolygonTest(slot["points"], (cx, cy), False)
                    if result >= 0:
                        is_occupied = True
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        break

                if is_occupied:
                    slot["status"] = "Occupied"
                    break

    # --- Visualisasi ---
    for slot in parking_slots:
        color = (0, 255, 0) if slot["status"] == "Free" else (0, 0, 255) 
        label = "Empty" if slot["status"] == "Free" else "Occupied"
        
        # Gambar kotak slot parkir
        cv2.polylines(frame, [slot["points"]], True, color, 2)
        # Tulis teks status
        cv2.putText(frame, str(slot["id"]), (slot["points"][0][0], slot["points"][0][1]-5), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
        
        if slot["status"] == "Occupied":
            occupied_count += 1

    # Tampilkan Dashboard Sederhana
    free_space = len(parking_slots) - occupied_count
    cv2.putText(frame, f'Free: {free_space}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(frame, f'Occupied: {occupied_count}', (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("Parking Management", frame)
    if cv2.waitKey(1) & 0xFF == 27: # Tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
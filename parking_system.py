import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# --- FUNGSI BARU UNTUK MENGHITUNG TITIK TENGAH (CENTROID) ---
def get_centroid(points):
    """Menghitung titik tengah (x, y) dari sebuah poligon (list 4 titik)."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    centroid_x = int(sum(x_coords) / len(x_coords))
    centroid_y = int(sum(y_coords) / len(y_coords))
    return (centroid_x, centroid_y)
# -----------------------------------------------------------


# 1. Load Model YOLOv8 (nano version agar cepat)
model = YOLO('yolov8l.pt') 

# 2. Definisikan Titik Slot Parkir (Menggunakan koordinat 4 titik Anda)
parkir1_1 = [(773, 228), (833, 304), (904, 302), (843, 227)]
parkir1_2 = [(904, 302), (843, 227), (913, 231), (974, 301)]
parkir1_3 = [(913, 231), (974, 301), (1039, 301), (980, 230)]
parkir1_4 = [(1039, 301), (980, 230), (1046, 225), (1103, 297)]
parkir1_5 = [(1046, 225), (1103, 297), (1169, 304), (1109, 225)]
parkir1_6 = [(1169, 304), (1109, 225), (1172, 223), (1234, 304)]

parkir2_1 = [(510, 456), (591, 528), (662, 535), (571, 450)]
parkir2_2 = [(662, 535), (571, 450), (640, 451), (719, 536)]
parkir2_3 = [(640, 451), (719, 536), (790, 540), (707, 451)]
parkir2_4 = [(790, 540), (707, 451), (772, 445), (848, 537)]
parkir2_5 = [(772, 445), (848, 537), (930, 535), (833, 449)]


# Masukkan semua area ke dalam list (MENGGUNAKAN CENTROID/TITIK TUNGGAL)
parking_slots = [
    # parkir area 1
    {"id": 1, "point": get_centroid(parkir1_1), "status": "Free"},
    {"id": 2, "point": get_centroid(parkir1_2), "status": "Free"},
    {"id": 3, "point": get_centroid(parkir1_3), "status": "Free"},
    {"id": 4, "point": get_centroid(parkir1_4), "status": "Free"},
    {"id": 5, "point": get_centroid(parkir1_5), "status": "Free"},
    {"id": 6, "point": get_centroid(parkir1_6), "status": "Free"},

    {"id": 19, "point": get_centroid(parkir2_1), "status": "Free"},
    {"id": 20, "point": get_centroid(parkir2_2), "status": "Free"},
    {"id": 21, "point": get_centroid(parkir2_3), "status": "Free"},
    {"id": 22, "point": get_centroid(parkir2_4), "status": "Free"},
    {"id": 23, "point": get_centroid(parkir2_5), "status": "Free"},
]

# Kelas yang ingin dideteksi (COCO dataset: 2=car, 3=motorcycle, 5=bus, 7=truck)
target_classes = [2, 3, 5, 7]

cap = cv2.VideoCapture('Parking.mp4') # Ganti video Anda

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Resize agar sesuai dengan koordinat yang Anda ambil di Langkah 2
    frame = cv2.resize(frame, (1280, 720))

    # Deteksi Object dengan YOLO
    results = model.predict(frame, conf=0.25, imgsz=1280, verbose=False) 
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    # Reset status setiap frame
    occupied_count = 0
    # Tambahkan kunci "is_occupied_in_frame" untuk visualisasi
    for slot in parking_slots:
        slot["is_occupied_in_frame"] = False 

    # Loop setiap objek yang terdeteksi
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5]) # class ID
        
        # Gambar Bounding Box di mobil
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1) # Kotak Merah tipis

        if d in target_classes:
            # Cek apakah titik parkir berada di dalam Bounding Box mobil
            for slot in parking_slots:
                px_slot = slot["point"][0] # X koordinat titik parkir
                py_slot = slot["point"][1] # Y koordinat titik parkir
                
                # Syarat: x1 < px_slot < x2  DAN y1 < py_slot < y2
                if x1 < px_slot < x2 and y1 < py_slot < y2:
                    slot["is_occupied_in_frame"] = True
                    break # Mobil ini sudah ketemu slotnya
    
    # --- Visualisasi Titik dan Dashboard ---
    for slot in parking_slots:
        color = (0, 255, 0) # Default Hijau (Free)
        if slot["is_occupied_in_frame"]:
            color = (0, 0, 255) # Merah (Occupied)
            occupied_count += 1
        
        # Tampilkan titik parkir 
        cv2.circle(frame, slot["point"], 5, color, -1) 
        
        # Tulis ID di sebelah titik
        text_position = (slot["point"][0] + 10, slot["point"][1] + 5)
        cv2.putText(frame, str(slot["id"]), text_position, cv2.FONT_HERSHEY_PLAIN, 1, color, 1)


    # Tampilkan Dashboard Sederhana
    total_slots = len(parking_slots)
    free_space = total_slots - occupied_count
    cv2.putText(frame, f'Free: {free_space}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(frame, f'Occupied: {occupied_count}', (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("Parking Management", frame)
    if cv2.waitKey(1) & 0xFF == 27: # Tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
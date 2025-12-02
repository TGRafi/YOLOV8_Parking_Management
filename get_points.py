import cv2
import numpy as np

# Ganti dengan nama video Anda
video_path = 'Parking.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"🚨 ERROR: Tidak dapat membuka video dari path: {video_path}")
    print("Pastikan path file video sudah benar dan file video tidak rusak.")
    exit()

points = []

def draw_polygon(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[{x}, {y}]") 
        points.append([x, y])

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_polygon)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame jika terlalu besar (opsional)
    frame = cv2.resize(frame, (1280, 720))

    for point in points:
        cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)

    cv2.imshow('Frame', frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
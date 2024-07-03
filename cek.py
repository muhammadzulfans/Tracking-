import cv2
import csv

# Buka file video
cap = cv2.VideoCapture("istockphoto-1184900033-640_adpp_is.mp4")

# Inisialisasi background subtractor
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Buka file CSV untuk menyimpan fitur
with open('car_features.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Area', 'Aspect_Ratio', 'Label'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Terapkan background subtraction
        mask = object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

        # Temukan kontur objek yang bergerak
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Analisis setiap kontur
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h

                # Asumsikan kendaraan berdasarkan rasio aspek (sesuaikan threshold sesuai kebutuhan)
                label = 1 if aspect_ratio > 3.5 else 0  # 1 untuk mobil, 0 untuk bukan mobil

                # Simpan fitur dan label ke file CSV
                writer.writerow([area, aspect_ratio, label])

cap.release()
print("Ekstraksi fitur selesai dan disimpan ke car_features.csv")

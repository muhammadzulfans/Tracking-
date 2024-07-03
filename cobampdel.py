import cv2
import pickle
import math
import time  # Tambahkan ini untuk mengimpor modul time

# Muat model dari file .pkl
with open('car_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Buka file video
cap = cv2.VideoCapture("istockphoto-1184900033-640_adpp_is.mp4")

# Inisialisasi background subtractor
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Threshold untuk jarak minimum antara centroid objek untuk menganggapnya sebagai objek yang sama
MIN_DIST_THRESHOLD = 50

# Pengaturan delay dan frekuensi tampilan frame
frame_delay = 0.1  # Delay antara frame dalam detik (100 ms)
display_every_n_frames = 2  # Tampilkan setiap frame ke-2

frame_counter = 0
car_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % display_every_n_frames != 0:
        continue

    # Terapkan background subtraction
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # Temukan kontur objek yang bergerak
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # List untuk menyimpan centroid mobil yang terdeteksi
    car_centroids = []

    # Reset jumlah mobil untuk frame ini
    frame_car_count = 0

    # Analisis setiap kontur
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h

            # Prediksi apakah objek adalah mobil
            features = [[area, aspect_ratio]]
            is_car = model.predict(features)[0]

            if is_car:
                # Hitung centroid dari bounding box
                cx = x + w // 2
                cy = y + h // 2

                # Cek apakah centroid ini dekat dengan centroid mobil yang sudah ada
                duplicate_found = False
                for centroid in car_centroids:
                    dist = math.sqrt((cx - centroid[0]) ** 2 + (cy - centroid[1]) ** 2)
                    if dist < MIN_DIST_THRESHOLD:
                        duplicate_found = True
                        break

                # Jika tidak ditemukan duplikat, anggap sebagai mobil baru
                if not duplicate_found:
                    frame_car_count += 1
                    car_count += 1
                    car_centroids.append((cx, cy))

                    # Gambar bounding box untuk visualisasi
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Tampilkan jumlah mobil pada frame
    cv2.putText(frame, f'Mobil: {car_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Delay untuk memperlambat pemrosesan
    time.sleep(frame_delay)

    # Keluar jika tombol ESC ditekan
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Total Mobil:", car_count)

from flask import Flask, request, render_template, redirect, url_for, Response
import os
import cv2
import pickle
import math
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model dari file .pkl
with open('car_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Inisialisasi background subtractor
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Threshold untuk jarak minimum antara centroid objek untuk menganggapnya sebagai objek yang sama
MIN_DIST_THRESHOLD = 50

# Pengaturan delay dan frekuensi tampilan frame
frame_delay = 0.1  # Delay antara frame dalam detik (100 ms)
display_every_n_frames = 2  # Tampilkan setiap frame ke-2

car_count = 0

def detect_cars(video_path):
    cap = cv2.VideoCapture(video_path)

    global car_count
    frame_counter = 0

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

        # Konversi frame ke format yang bisa ditampilkan di browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Delay untuk memperlambat pemrosesan
        time.sleep(frame_delay)

    cap.release()

def get_files():
    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        return []
    files = os.listdir(upload_folder)
    return files

@app.route('/', methods=['GET', 'POST'])
def index():
    car_count = len(get_files())
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
 
            return redirect(url_for('video_feed', filename=file.filename))
    return render_template('index.html', car_count=car_count, files=get_files())

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(detect_cars(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)

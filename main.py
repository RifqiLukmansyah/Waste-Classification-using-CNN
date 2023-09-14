import cv2
import tensorflow as tf
import numpy as np

# Muat model CNN yang telah dilatih sebelumnya
model = tf.keras.models.load_model('model.h5')

# Fungsi untuk mendeteksi objek dalam citra
def detect_object(frame, model):
    # Ubah ukuran frame sesuai dengan ukuran yang diperlukan oleh model
    input_size = (224, 224)  # Sesuaikan dengan ukuran input model Anda
    frame = cv2.resize(frame, input_size)

    # Normalisasi frame
    frame = frame / 255.0

    # Lakukan prediksi menggunakan model
    predictions = model.predict(np.expand_dims(frame, axis=0))

    # Ambil indeks kelas dengan probabilitas tertinggi
    class_index = np.argmax(predictions)

    # Probabilitas kelas dengan probabilitas tertinggi
    confidence = predictions[0][class_index]

    return class_index, confidence

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Deteksi objek dalam frame
    class_index, confidence = detect_object(frame, model)

    # Tentukan warna teks
    if class_index == 0:
        text_color = (0, 0, 255)  # Merah jika "Anorganik"
    elif class_index == 1:
        text_color = (0, 255, 0)  # Hijau jika "Organik"
    else:
        text_color = (255, 0, 0)  # Biru jika tidak keduanya

    # Ubah indeks menjadi nama kelas
    class_name = "Anorganik" if class_index == 0 else "Organik" if class_index == 1 else "Tidak Diketahui"

    # Tampilkan tipe objek dan tingkat kepercayaan pada frame
    text = f'Object Type: {class_name}, Confidence: {confidence:.2f}'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Tampilkan frame
    cv2.imshow('Object Detection', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()

from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Fungsi untuk mengambil frame video dan mengubahnya menjadi MJPEG


def generate():
    cascPath = "src\\face.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    video_capture = cv2.VideoCapture('http://192.168.100.220:8080/?action=stream')
    # video_capture = cv2.VideoCapture('0')

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a green rectangle around the faces and add the label "wajah"
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add label "wajah" above the rectangle
            label = "wajah"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mengubah frame menjadi MJPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_capture.release()
    cv2.destroyAllWindows()

# Route untuk halaman utama


@app.route('/')
def index():
    return render_template('index.html')

# Route untuk streaming video


@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='192.168.100.220', debug=True)

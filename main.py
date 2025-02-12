# main.py 改进代码
import cv2
from ultralytics import YOLO
from flask import Flask, Response


class VideoProcessingApp:
    def __init__(self, host='0.0.0.0', port=5000):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.setup_routes()
        self.recording = False
        self.model = YOLO('./best.pt')
        self.cap = None  # 延迟初始化摄像头

    def setup_routes(self):
        self.app.add_url_rule('/predictCamera', 'predictCamera', self.predictCamera)
        self.app.add_url_rule('/stopCamera', 'stopCamera', self.stopCamera, methods=['GET'])

    def run(self):
        self.app.run(host=self.host, port=self.port)

    def init_camera(self):
        """初始化摄像头"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def release_camera(self):
        """释放摄像头资源"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def predictCamera(self):
        """改进的视频流处理方法"""
        self.recording = True
        self.init_camera()

        def generate():
            while self.recording and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                results = self.model.predict(source=frame, imgsz=640)
                processed_frame = results[0].plot()
                _, jpeg = cv2.imencode('.jpg', processed_frame)
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            self.release_camera()

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def stopCamera(self):
        """改进的停止方法"""
        self.recording = False
        self.release_camera()
        return "摄像头已停止", 200


if __name__ == '__main__':
    video_app = VideoProcessingApp()
    video_app.run()

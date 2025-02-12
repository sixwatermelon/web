import cv2
from ultralytics import YOLO
from flask import Flask, Response


# Flask 应用设置
class VideoProcessingApp:
    def __init__(self, host='0.0.0.0', port=5000):
        """初始化 Flask 应用并设置路由"""
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.setup_routes()
        self.recording = False  # 标志位，判断是否正在录制视频
        self.model = YOLO('./best.pt')
        self.cap = cv2.VideoCapture(0)

    def setup_routes(self):
        """设置所有路由"""
        self.app.add_url_rule('/predictCamera', 'predictCamera', self.predictCamera)
        self.app.add_url_rule('/stopCamera', 'stopCamera', self.stopCamera, methods=['GET'])

    def run(self):
        """启动 Flask 应用"""
        self.app.run(host=self.host, port=self.port, debug=True)  # 使用 Flask 的 run 方法

    def predictCamera(self):
        """摄像头视频流处理接口"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.recording = True

        def generate():
            try:
                while self.recording:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    results = self.model.predict(source=frame, imgsz=640, show=False)
                    processed_frame = results[0].plot()
                    _, jpeg = cv2.imencode('.jpg', processed_frame)
                    yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            finally:
                if self.cap.isOpened():
                    self.cap.release()
                cv2.destroyAllWindows()

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def stopCamera(self):
        """停止摄像头预测"""
        self.recording = False
        return "摄像头已停止", 200


# 启动应用
if __name__ == '__main__':
    video_app = VideoProcessingApp()
    video_app.run()  # 启动 Flask 应用
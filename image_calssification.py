from keras.models import load_model  # TensorFlow 필요, Keras가 동작하려면 tensorflow version=2.12.0
from PIL import Image, ImageOps  # PIL 대신 pillow를 설치 필요
import cv2  # opencv-python 설치 필요
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
import sys

class ImageProcessor(QMainWindow):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        loadUi("./resources/ui.ui", self)  # UI 파일 로드
 
        # 버튼 연결
        self.load_bt.clicked.connect(self.load_image)
        self.cam_bt.clicked.connect(self.capture_image)
        self.play_bt.clicked.connect(self.out_image)

        self.selected_image = None
        self.class_name = None
        self.confidence_score = None

        # 모델과 클래스 이름 로드
        self.model = load_model("./resources/keras_model.h5", compile=False)
        self.class_names = open("./resources/labels.txt", "r").readlines()

    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "이미지 파일 열기", "", "이미지 파일 (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            self.selected_image = filename
            pixmap = QPixmap(filename)
            self.image.setPixmap(pixmap)
            self.predict_image()

    def capture_image(self):
        pixmap = QPixmap("./resources/three.jpg")
        self.image.setPixmap(pixmap)
        cv2.waitKey(1000)
        pixmap = QPixmap("./resources/two.jpg")
        self.image.setPixmap(pixmap)
        cv2.waitKey(1000)
        pixmap = QPixmap("./resources/one.jpg")
        self.image.setPixmap(pixmap)
        cv2.waitKey(300)
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("captured_image.jpg", frame)
            self.selected_image = "captured_image.jpg"
            pixmap = QPixmap("captured_image.jpg")
            self.image.setPixmap(pixmap)
            self.predict_image()
        cap.release()

    def out_image(self):
        if self.selected_image:
            QMessageBox.information(self, "결과", f"클래스: {self.class_name[2:]}정확도: {self.confidence_score}%")

        else:
            QMessageBox.warning(self, "이미지 없음", "선택된 이미지가 없습니다.")

    def predict_image(self):
        # 이미지 전처리 및 예측
        image = Image.open(self.selected_image).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        prediction = self.model.predict(data)
        index = np.argmax(prediction)
        self.class_name = self.class_names[index]
        confidence_score_1 = prediction[0][index]
        self.confidence_score = str(np.round(confidence_score_1 * 100))[:-2]

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    while 1:
        window = ImageProcessor()
        window.show()
        sys.exit(app.exec_())



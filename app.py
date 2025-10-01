from flask import Flask, render_template, request, send_file
import os
import cv2
from utils.detect_sticks import detect_sticks

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            # ตรวจจับแท่งไม้
            result_img, count = detect_sticks(image_path)

            # บันทึกผลลัพธ์
            result_path = os.path.join(RESULT_FOLDER, "result.jpg")
            cv2.imwrite(result_path, result_img)

            return render_template("index.html", count=count, result="result.jpg")

    return render_template("index.html", count=None, result=None)

@app.route("/results/<filename>")
def results(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename))

if __name__ == "__main__":
    app.run(debug=True)

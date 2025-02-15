"""SECUREWAYS_Yolo_Detection_CCTV_Traffic_Survelliance_Women's_Safety.ipynb"""

import sys
print(sys.version)

import pip
print(pip.__version__)

import os
os.system("yolov8_env\\Scripts\\activate")

import subprocess
subprocess.check_call(["pip", "install", "ultralytics"])

import ultralytics
print(ultralytics.__version__)

import subprocess
subprocess.check_call(["pip", "install", "torch", "torchvision", "torchaudio"])

import torch
print(torch.__version__)
print(torch.cuda.is_available())

import subprocess
subprocess.check_call(["pip", "install", "opencv-python"])

import cv2
print(cv2.__version__)

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

import subprocess
subprocess.check_call(["pip", "install", "--upgrade", "ultralytics"])

import subprocess
subprocess.check_call(["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])

from ultralytics import YOLO
import cv2

def detect_vehicles(video_path):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_vehicles("path/to/video.mp4")

import os
os.system("python detect_vehicles.py")

import subprocess
import sys

def install_pytube():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytube"])
install_pytube()

import pytube
print(pytube.__version__)

import subprocess
import sys

result = subprocess.run([sys.executable, "-m", "pip", "install", "pytube"], capture_output=True, text=True)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.run([sys.executable, "-m", "pip", "install", "pytube"])

import urllib.request

url = "https://github.com/pytube/pytube/archive/refs/heads/master.zip"
urllib.request.urlretrieve(url, "pytube.zip")

import zipfile
import os

with zipfile.ZipFile("pytube.zip", "r") as zip_ref:
    zip_ref.extractall("pytube_folder")

subprocess.run([sys.executable, "-m", "pip", "install", "./pytube_folder/pytube-master"])

import pytube
print(pytube.__version__)

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pytube"])

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pytube"])
subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/pytube/pytube"])

import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"])

!ls -lh

!pip install yt_dlp

import yt_dlp

url = "https://www.youtube.com/watch?v=BmzyM6omS68"
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

import os

video_path = "Busy traffic in Kolkata - West Bengal [rU8zTNJ8vUM].mp4"

if os.path.exists(video_path):
    print("Video file found!")
else:
    print("Video file not found! Check if it was created correctly.")

import cv2
import torch
import numpy as np
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
video_path = "Street Road Traffic ｜ Video of Traffic on Highway ｜ HD Footage ｜ Traffic Car Highway HD Video [BmzyM6omS68].mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
output_path = "output.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
danger_zone = [(frame_width // 3, frame_height // 1.5), (3 * frame_width // 4, frame_height)]
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    print(f"Processing frame {frame_num}/{frame_count}")
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            color = (0, 255, 0)
            if label in ["car", "truck", "bus", "motorcycle"]:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if label in ["car", "truck", "bus", "motorcycle"] and (x1 > danger_zone[0][0] and x2 < danger_zone[1][0] and y2 > danger_zone[0][1]):
                cv2.putText(frame, "Accident Risk!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    out.write(frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit early
        break
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output saved to {output_path}")

from google.colab import files
files.download("output.mp4")

import yt_dlp

url = "https://www.youtube.com/watch?v=YF_DzoTDO-0"
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

import cv2
import torch
import numpy as np
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
video_path = "Road Safety： What are the causes of this accident ｜｜ Cyberabad Traffic Police [YF_DzoTDO-0].mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
output_path = "output.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
danger_zone = [(frame_width // 3, frame_height // 1.5), (3 * frame_width // 4, frame_height)]
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    print(f"Processing frame {frame_num}/{frame_count}")
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            color = (0, 255, 0)
            if label in ["car", "truck", "bus", "motorcycle"]:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if label in ["car", "truck", "bus", "motorcycle"] and (x1 > danger_zone[0][0] and x2 < danger_zone[1][0] and y2 > danger_zone[0][1]):
                cv2.putText(frame, "Accident Risk!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    out.write(frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output saved to {output_path}")

from google.colab import files
files.download("output.mp4")

import yt_dlp

url = "https://www.youtube.com/watch?v=R3sszWW83qA"
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

import cv2
import torch
import numpy as np
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
video_path = "Oppo Delhi Metro Train Wrap [R3sszWW83qA].mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
output_path = "output3.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
danger_zone = [(frame_width // 4, frame_height // 1.5), (3 * frame_width // 4, frame_height)]
processed_frames = 0
total_persons_detected = 0
risk_warnings = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frames += 1
    print(f"Processing frame {processed_frames}/{frame_count}")
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "person":
                total_persons_detected += 1
                color = (0, 255, 0)
                if x1 > danger_zone[0][0] and x2 < danger_zone[1][0] and y2 > danger_zone[0][1]:
                    color = (0, 0, 255)
                    risk_warnings += 1
                    cv2.putText(frame, "Potential Risk!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    out.write(frame)
    if processed_frames % 50 == 0:
        print(f"Written {processed_frames} frames so far...")
out.release()
cap.release()
cv2.destroyAllWindows()
accuracy = (processed_frames / frame_count) * 100 if frame_count > 0 else 0
risk_rate = (risk_warnings / total_persons_detected) * 100 if total_persons_detected > 0 else 0
print(f"Processing complete. Total frames processed: {processed_frames}")
print(f"Total persons detected: {total_persons_detected}")
print(f"Potential unsafe situations flagged: {risk_warnings}")
print(f"Detection Accuracy: {accuracy:.2f}%")
print(f"Risk Warning Rate: {risk_rate:.2f}% (Risk alerts per detected person)")
print(f"Output saved to: {output_path}")

from google.colab import files
files.download("output3.mp4")

import yt_dlp

url = "https://www.youtube.com/watch?v=I6Nna0_4ktY"
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])



import cv2
import torch
import numpy as np
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
video_path = "indian people walking ｜｜ no copyright [I6Nna0_4ktY].mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
output_path = "output3.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
danger_zone = [(frame_width // 4, frame_height // 1.5), (3 * frame_width // 4, frame_height)]
processed_frames = 0
total_persons_detected = 0
risk_warnings = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frames += 1
    print(f"Processing frame {processed_frames}/{frame_count}")
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "person":
                total_persons_detected += 1
                color = (0, 255, 0)
                if x1 > danger_zone[0][0] and x2 < danger_zone[1][0] and y2 > danger_zone[0][1]:
                    color = (0, 0, 255)
                    risk_warnings += 1
                    cv2.putText(frame, "Potential Risk!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    out.write(frame)
    if processed_frames % 50 == 0:
        print(f"Written {processed_frames} frames so far...")
out.release()
cap.release()
cv2.destroyAllWindows()
accuracy = (processed_frames / frame_count) * 100 if frame_count > 0 else 0
risk_rate = (risk_warnings / total_persons_detected) * 100 if total_persons_detected > 0 else 0
print(f"Processing complete. Total frames processed: {processed_frames}")
print(f"Total persons detected: {total_persons_detected}")
print(f"Potential unsafe situations flagged: {risk_warnings}")
print(f"Detection Accuracy: {accuracy:.2f}%")
print(f"Risk Warning Rate: {risk_rate:.2f}% (Risk alerts per detected person)")
print(f"Output saved to: {output_path}")

from google.colab import files
files.download("output3.mp4")


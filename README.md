# yolov8
yolov8 - 웹캠에 yolov8 모델 적용방법 코드
참고 주소 : https://datalab.medium.com/yolov8-detection-from-webcam-step-by-step-cpu-d590a0700e36
# source from https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993

from ultralytics import YOLO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    #coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# ros2 버젼 yolov8 #####################################################################################
from typing import List, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from ultralytics import YOLO
import cv2
import math

class yolov8Publisher(Node):
    def __init__(self):
        super().__init__('yolov8_publisher')
        qos_profile = QoSProfile(depth=10)
        self.publisher = self.create_publisher(String, 'yolov8_detection',qos_profile)
        self.bridge = CvBridge()
    
    def publish_detection_msg(self, class_name):
        msg = String()
        msg.data = class_name
        self.publisher.publish(msg)
        self.get_logger().info(f"Published detection result: {class_name}")
        
def main(args=None):
    rclpy.init(args = args)
    yolov8_publisher = yolov8Publisher()
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    model = YOLO("yolov8n.pt")
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
    
        for r in results:
            boxes = r.boxes
        
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  #탐지된 객체의 좌표
                #정수형으로 변환
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            
                #탐지된 객체를 사각형으로 표시
                cv2.rectangle(img, (x1,y1),(x2,y2), (255, 0, 255), 2)
            
                #탐지된 객체의 확률
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence = ", confidence)
            
                #탐지된 객체의 클래스 인덱스
                cls = int(box.cls[0])
                #클래스 이름 출력
                print("Class name = ", classNames[cls])
                #클래스 이름 화면에 표시
                cv2.putText(img, classNames[cls], [x1,y1], cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),2)
    
                yolov8_publisher.publish_detection_msg(classNames[cls]) #클래스 이름을 다른 노드로 퍼블리쉬 
                
        cv2.imshow('Wdbcam', img)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    rclpy.spin(yolov8_publisher)
    yolov8_publisher.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()



#Segmentation#########################################################################################

from ultralytics import YOLO
import cv2
import math
import numpy as np

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolov8n-seg.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)

    for r in results:
        masks = r.masks  # 세그멘테이션 마스크

        for i, mask in enumerate(masks):
            # 마스크를 바이너리 형태로 변환
            mask_data = mask.data.cpu().numpy().astype(np.uint8).squeeze()

            # 마스크를 이미지에 오버레이
            color = (0, 255, 0)  # 세그멘테이션 마스크의 색상
            mask_color = np.zeros_like(img)
            mask_color[:, :, 1] = mask_data * 255  # Green channel
            
            img = cv2.addWeighted(img, 1, mask_color, 0.5, 0)

            # confidence
            confidence = math.ceil((r.boxes.conf[i] * 100)) / 100
            print("Confidence --->", confidence)

            # class name
            cls = int(r.boxes.cls[i])
            print("Class name -->", classNames[cls])

            # object details
            bbox = r.boxes.xyxy[i]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            text_color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, text_color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#Segentation && bounding box #################################################################################

from ultralytics import YOLO
import cv2
import math
import numpy as np

#image 
img_path = cv2.imread("C:\pothole\pothole7.png")
img = cv2.resize(img_path, (640,640))

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model_seg = YOLO("yolov8n-seg.pt")
model_det = YOLO("pothole.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

classNames2 = ["pothole"]

while True:
    #success, img = cap.read()
    #if not success:
    #    break

    results_seg = model_seg(img, stream=True)
    results_det = model_det(img, stream=True)

    for r in results_seg:
        masks = r.masks  # 세그멘테이션 마스크

        for i, mask in enumerate(masks):
            # 마스크를 바이너리 형태로 변환
            mask_data = mask.data.cpu().numpy().astype(np.uint8).squeeze()

            # 마스크를 이미지에 오버레이
            mask_color = np.zeros_like(img)
            mask_color[:, :, 1] = mask_data * 255  # Green channel
            
            img = cv2.addWeighted(img, 1, mask_color, 0.5, 0)

            # confidence
            #confidence = math.ceil((r.boxes.conf[i] * 100)) / 100
            #print("Confidence (segmentation) --->", confidence)

            # class name
            #cls = int(r.boxes.cls[i])
            
            #print("Class name (segmentation) -->", classNames[cls])

    for r in results_det:
        boxes = r.boxes

        for i, box in enumerate(boxes):
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            #confidence = math.ceil((box.conf[i] * 100)) / 100
            #print("Confidence (detection) --->", confidence)

            # class name
            cls = int(box.cls[0])
            #print("Class name (detection) -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames2[cls], org, font, fontScale, color, thickness)

    break


cv2.imshow('phothole', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

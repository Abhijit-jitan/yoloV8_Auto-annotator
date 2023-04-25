import cv2,datetime,os
from ultralytics import YOLO
import pybboxes as pbx


def detect_box(results,txt_file_name,imW,imH):
    boxes = results[0].boxes
    bboxes=boxes.xyxy
    classes=boxes.cls

    with open(txt_file_name,'w') as f:

        for index in range(len(boxes)):
            if int(classes[index]) == 2: 
                xmin=int(bboxes[index][0])
                ymin=int(bboxes[index][1])
                xmax=int(bboxes[index][2])
                ymax=int(bboxes[index][3])
                class_id=0 #int(classes[index])
            
            if int(classes[index]) == 7: 
                xmin=int(bboxes[index][0])
                ymin=int(bboxes[index][1])
                xmax=int(bboxes[index][2])
                ymax=int(bboxes[index][3])
                class_id=1 #int(classes[index])
            

            yolo_bbox=pbx.convert_bbox((xmin,ymin,xmax,ymax),from_type="voc", to_type="yolo", image_size=(imW,imH))
            f.write(f"{class_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]} \n")



image_directory=r"E:\Tech\yolov8\data"
model = YOLO(r"E:\Tech\yolov8\pretrained model\yolov8x.pt")
imW,imH=1364,768


for img in os.listdir(image_directory):
    if img.endswith(".jpg"):
        image_path=os.path.join(image_directory,img)
        base_name=os.path.basename(image_path).split(".jpg")[0]+".txt"

        results=model(image_path,conf=.75,iou=.6)    
        frame=results[0].orig_img
        detect_box(results,os.path.join(image_directory,base_name),imW,imH)

print("Task Complete !!!!")


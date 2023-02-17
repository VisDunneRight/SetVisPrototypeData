import os
from transformers import pipeline

from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from transformers import AutoImageProcessor, ConditionalDetrForObjectDetection

import torch
from PIL import Image
import requests






# models = ["facebook/detr-resnet-50", "facebookresearch/detr", "facebookresearch/maskrcnn-benchmark", "fvcore/mask-rcnn-fpn", "fvcore/detr", "fvcore/retinanet-fpn"]

def run_Facebook_OneHundredOne(folder_path, folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)

        image = Image.open(image_path)

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        with open(os.path.join(folderName, image_file.split(".")[0] + ".txt"), 'w') as f:
            
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = box.tolist()
                    box = [round(box[0] / image.size[0], 4), round(box[1] / image.size[1], 4),
                        round(box[2] / image.size[0], 4), round(box[3] / image.size[1], 4)]
                    f.write(f"{model.config.id2label[label.item()]} {box[0]} {box[1]} {box[2]} {box[3]} {round(score.item(), 3)}\n")



def run_Facebook_Fifty(folder_path, folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    
        with open(os.path.join(folderName, image_file.split(".")[0] + ".txt"), 'w') as f:
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = box.tolist()
                    box = [round(box[0] / image.size[0], 4), round(box[1] / image.size[1], 4),
                        round(box[2] / image.size[0], 4), round(box[3] / image.size[1], 4)]
                    f.write(f"{model.config.id2label[label.item()]} {box[0]} {box[1]} {box[2]} {box[3]} {round(score.item(), 3)}\n")
   
def run_Google_Owl(folder_path, folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)

        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    
        with open(os.path.join(folderName, image_file.split(".")[0] + ".txt"), 'w') as f:
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = box.tolist()
                    box = [round(box[0] / image.size[0], 4), round(box[1] / image.size[1], 4),
                        round(box[2] / image.size[0], 4), round(box[3] / image.size[1], 4)]
                    f.write(f"{model.config.id2label[label.item()]} {box[0]} {box[1]} {box[2]} {box[3]} {round(score.item(), 3)}\n")

def run_YOLO_tiny(folder_path, folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)

        feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
        model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        logits = outputs.logits
        bboxes = outputs.pred_boxes

        # print(logits)
        # print(bboxes)
        target_sizes = torch.tensor([image.size[::-1]])
        results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    
        with open(os.path.join(folderName, image_file.split(".")[0] + ".txt"), 'w') as f:
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = box.tolist()
                    box = [round(box[0] / image.size[0], 4), round(box[1] / image.size[1], 4),
                        round(box[2] / image.size[0], 4), round(box[3] / image.size[1], 4)]
                    f.write(f"{model.config.id2label[label.item()]} {box[0]} {box[1]} {box[2]} {box[3]} {round(score.item(), 3)}\n")

def run_YOLO_base(folder_path, folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)

        feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-base')
        model = YolosForObjectDetection.from_pretrained('hustvl/yolos-base')

        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        logits = outputs.logits
        bboxes = outputs.pred_boxes

        # print(logits)
        # print(bboxes)
        target_sizes = torch.tensor([image.size[::-1]])
        results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    
        with open(os.path.join(folderName, image_file.split(".")[0] + ".txt"), 'w') as f:
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = box.tolist()
                    box = [round(box[0] / image.size[0], 4), round(box[1] / image.size[1], 4),
                        round(box[2] / image.size[0], 4), round(box[3] / image.size[1], 4)]
                    f.write(f"{model.config.id2label[label.item()]} {box[0]} {box[1]} {box[2]} {box[3]} {round(score.item(), 3)}\n")

def run_Microsoft_detr(folder_path, folderName):
    if not os.path.exists(folderName):
        os.makedirs(folderName)
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)

        processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
        model = ConditionalDetrForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    
        with open(os.path.join(folderName, image_file.split(".")[0] + ".txt"), 'w') as f:
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = box.tolist()
                    box = [round(box[0] / image.size[0], 4), round(box[1] / image.size[1], 4),
                        round(box[2] / image.size[0], 4), round(box[3] / image.size[1], 4)]
                    f.write(f"{model.config.id2label[label.item()]} {box[0]} {box[1]} {box[2]} {box[3]} {round(score.item(), 3)}\n")

# these work
# run_Facebook_OneHundredOne("./oneHundredImages/images", "./oneHundredImages/detr-resnet-101")
# run_Facebook_Fifty("./oneHundredImages/images", "./oneHundredImages/detr-resnet-50")
# run_YOLO_tiny("./oneHundredImages/images", "./oneHundredImages/yolos-tiny")
# run_YOLO_base("./oneHundredImages/images", "./oneHundredImages/yolos-base")
# run_Microsoft_detr("./oneHundredImages/images", "./oneHundredImages/conditional-detr-resnet-50")


run_Facebook_OneHundredOne("./twentyFive/images", "./twentyFive/detr-resnet-101")
run_Facebook_Fifty("./twentyFive/images", "./twentyFive/detr-resnet-50")
run_YOLO_tiny("./twentyFive/images", "./twentyFive/yolos-tiny")
run_YOLO_base("./twentyFive/images", "./twentyFive/yolos-base")
run_Microsoft_detr("./twentyFive/images", "./twentyFive/conditional-detr-resnet-50")

# run_Google_Owl("./twentyFiveImages/images", "./twentyFiveImages/owlvit-base-patch32")

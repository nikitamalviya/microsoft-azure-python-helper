import os, sys, click, io, numpy as np, cv2
from cognitive_service_vision_model_customization_python_samples import PredictionClient
from cognitive_service_vision_model_customization_python_samples import ResourceType

def calculate_iou(box1, box2):
    # Calculate the intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    # Calculate the areas of the bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area
    # Calculate the IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def apply_nms(boxes, scores, iou_threshold):
    indices = list(range(len(boxes)))
    selected_indices = []
    while indices:
        current = indices[0]
        selected_indices.append(current)
        remaining_indices = []
        for i in indices[1:]:
            iou = calculate_iou(boxes[current], boxes[i])
            if iou <= iou_threshold:
                remaining_indices.append(i)
            else:
                click.secho(f"Eliminated box {i} due to IoU {iou:.2f} with box {current}", fg="red")
        indices = remaining_indices
    return selected_indices

def filter_boxes_with_nms(boxes, scores, threshold, iou_threshold):
    filtered_indices = [i for i in range(len(scores)) if scores[i] >= threshold]
    filtered_boxes = [boxes[i] for i in filtered_indices]
    filtered_scores = [scores[i] for i in filtered_indices]

    selected_indices = apply_nms(filtered_boxes, filtered_scores, iou_threshold)
    selected_boxes = [filtered_boxes[i] for i in selected_indices]
    selected_filtered_indices = [filtered_indices[i] for i in selected_indices]
    click.secho(f"Selected {len(selected_boxes)} boxes after NMS", fg="green")
    return selected_filtered_indices


def predict_image(image_content, model_name, resource_name, resource_key):
    ''' Prediction main '''
    try:
        """Perform prediction on an image using Azure Cognitive Services with exponential backoff."""
        # blob_stream.seek(0) # image_content == blob_stream
        # image_bytes = io.BytesIO(blob_stream.read())
        image_bytes = io.BytesIO(image_content)
        resource_type = ResourceType.SINGLE_SERVICE_RESOURCE
        prediction_client = PredictionClient(resource_type, resource_name, None, resource_key)
        prediction = prediction_client.predict(model_name, image_bytes, content_type='image/jpeg')
        return prediction
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"\nUnable to make predictions on the image : {len(blob_stream)}...!", fg="red")
        click.secho(f"\nError Type : {exc_type} ----->> File Name : {fname} ----->> Line Number : {exc_tb.tb_lineno}", fg="red")

def parse_detection_results(prediction, model_name, threshold_human, threshold_head, nms_threshold_human, nms_threshold_head):
    ''' Parse output JSON response '''
    try:
        detections = {}
        annotations = []
        json_predictions = []
        thresh_detections = 0
        nms_detections = 0
        
        results={
            "status": False,
            "modelName": model_name,
            "modelVersion": "",
            "imgWidth": 0,
            "imgHeight": 0,
            "numOfObjectsDetectedAzure" : 0, 
            "numOfObjectsDetectedThresh" : 0,
            "numOfObjectsDetectedNMS" : 0,
            "detections": [],
            "annotations" : [],
            "json_predictions" : []
        }
        # print(f"\n\nprediction : {prediction}") # azure output
        if prediction["customModelResult"] != {}:
            detections = prediction["customModelResult"]["objectsResult"]["values"]

            #########################################################################
            if len(detections) > 0:
                ### human
                human_boxes = []
                human_scores = []
                human_annotations = []
                ### head 
                head_boxes = []
                head_scores = []
                head_annotations = []
                ### iterate over the detections
                for detection in detections:
                    confidence = round(detection["tags"][0]["confidence"], 2)
                    class_label = detection["tags"][0]["name"]
                    if class_label == "human": 
                        threshold = threshold_human
                    else: 
                        threshold = threshold_head
                    
                    if confidence >= threshold:
                        xmin = int(detection["boundingBox"]["x"])
                        ymin = int(detection["boundingBox"]["y"])
                        xmax = int(detection["boundingBox"]["x"]) + int(detection["boundingBox"]["w"])
                        ymax = int(detection["boundingBox"]["y"]) + int(detection["boundingBox"]["h"])
                        annotation = {
                            "id": detection["id"],
                            "bbox": [xmin, ymin, xmax, ymax],
                            "category": class_label,
                            "score": confidence,
                            "azurebbox": detection["boundingBox"],
                            "score_original" : detection["tags"][0]["confidence"]
                        }
                        if class_label == "human":
                            human_boxes.append([xmin, ymin, xmax, ymax])
                            human_scores.append(confidence)
                            human_annotations.append(annotation)
                        else:
                            head_boxes.append([xmin, ymin, xmax, ymax])
                            head_scores.append(confidence)
                            head_annotations.append(annotation)

                click.secho(f"## Filter AZURE detections : {len(detections)} --> {len(human_annotations) + len(head_annotations)}", fg="blue")
                thresh_detections = len(human_annotations) + len(head_annotations)
                
                print(f"\nhuman_annotations 1: {human_annotations}")
                
                ##### Apply NMS for humans
                if len(human_boxes) > 0:
                    human_indices = filter_boxes_with_nms(human_boxes, human_scores, threshold_human, nms_threshold_human)
                    human_nms_annotations = [human_annotations[i] for i in human_indices]
                    click.secho(f"## Filter human bboxes using NMS : {len(human_annotations)} --> {len(human_nms_annotations)}", fg="blue")
                else:
                    human_nms_annotations = []

                # print(f"\nhuman_annotations 2: {human_nms_annotations}")

                ##### Apply NMS for heads
                if len(head_boxes) > 0:
                    head_indices = filter_boxes_with_nms(head_boxes, head_scores, threshold_head, nms_threshold_head)
                    head_nms_annotations = [head_annotations[i] for i in head_indices]
                    click.secho(f"## Filter head bboxes using NMS : {len(head_annotations)} --> {len(head_nms_annotations)}", fg="blue")
                else:
                    head_nms_annotations = []
                                
                # Combine the final annotations
                annotations = human_nms_annotations + head_nms_annotations
                print("\n\nFinal annotations after NMS : ", annotations)

            ##### Add in Pipeline format results JSON key
            # Convert annotations to json_predictions format
            for ann in annotations:
                pred_item = {
                    "category": ann["category"],
                    "score": ann["score_original"],
                    "bounding_box": {
                        "xmin": ann["azurebbox"]["x"],
                        "ymin": ann["azurebbox"]["y"],
                        "xmax": ann["azurebbox"]["x"] + ann["azurebbox"]["w"],
                        "ymax": ann["azurebbox"]["y"] + ann["azurebbox"]["h"]
                    }
                }
                json_predictions.append(pred_item)

        results["status"]=True
        results["modelVersion"]=prediction["modelVersion"]
        results["imgWidth"] = prediction["metadata"]["width"]
        results["imgHeight"] = prediction["metadata"]["height"]
        results["json_predictions"] = json_predictions
        results["numOfObjectsDetectedAzure"]=len(detections)
        results["numOfObjectsDetectedThresh"]=thresh_detections
        results["numOfObjectsDetectedNMS"]=len(annotations)
        results["detections"] = detections
        results["annotations"] = annotations
 
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"\nUnable to parse the prediction output....!", fg="red")
        click.secho(f"\nError Type : {exc_type} ----->> File Name : {fname} ----->> Line Number : {exc_tb.tb_lineno}", fg="red")
    return detections, results


##########################################################################
'''

##### Apply NMS for humans
if len(human_boxes) > 0:
    human_boxes = np.array(human_boxes)
    human_scores = np.array(human_scores)
    human_indices = cv2.dnn.NMSBoxes(human_boxes.tolist(), human_scores.tolist(), threshold_human, nms_threshold_human)
    human_indices = human_indices.flatten() if len(human_indices) > 0 else []

    selected_annotations = []
    dropped_annotations = []
    for i in range(len(human_boxes)):
        if i in human_indices:
            selected_annotations.append(human_annotations[i])
        else:
            dropped_annotations.append(human_annotations[i])
    # Print details of selected bounding boxes
    print("Selected Bounding Boxes (After NMS):")
    for idx in human_indices:
        print(f"Box: {human_boxes[idx]}, Score: {human_scores[idx]}")

    # Print details of dropped bounding boxes
    print("\nDropped Bounding Boxes (After NMS):")
    for annotation in dropped_annotations:
        print(f"Box: {annotation['bbox']}, Score: {annotation['score']}")
        for idx in human_indices:
            iou_score = calculate_iou(annotation['bbox'], human_boxes[idx])
            print(f" -- Overlap with selected Box {human_boxes[idx]}: IoU={iou_score}, Selected Box Score: {human_scores[idx]}")
    click.secho(f"## Filter human bboxes using NMS : {len(human_annotations)} --> {len(selected_annotations)}", fg="blue")
else:
    selected_annotations = []




if len(detections) > 0:
    boxes = []
    scores = []
    annotations = []
    for detection in detections:
        confidence = round(detection["tags"][0]["confidence"], 2)
        class_label = detection["tags"][0]["name"]
        if class_label == "human": threshold = threshold_human
        else: threshold = threshold_head
        if confidence >= threshold:
            xmin = int(detection["boundingBox"]["x"])
            ymin = int(detection["boundingBox"]["y"])
            xmax = int(detection["boundingBox"]["x"]) + int(detection["boundingBox"]["w"])
            ymax = int(detection["boundingBox"]["y"]) + int(detection["boundingBox"]["h"])
            boxes.append([xmin, ymin, xmax, ymax])
            scores.append(confidence)
            # create the annotation dictionary(old format) for the bounding box detections
            annotations.append({"id": detection["id"], # unique id
                                "bbox": [xmin, ymin, xmax, ymax],
                                "category": class_label,
                                "score": confidence,
                                "azurebbox": detection["boundingBox"],})
    click.secho(f"## Filter AZURE detections : {len(detections)} --> {len(annotations)}", fg="blue")
    
    ##### Apply NMS threshold
    if len(boxes) > 0:
        # Convert boxes and scores to the appropriate format for NMS
        boxes = np.array(boxes)
        scores = np.array(scores)
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), threshold, nms_threshold)
        indices = indices.flatten() if len(indices) > 0 else []
        nms_annotations = [annotations[i] for i in indices]
        click.secho(f"## Filter bboxes using NMS : {len(annotations)} --> {len(nms_annotations)}", fg="blue")
        print(annotations, " -----> ", nms_annotations)
        annotations = nms_annotations
    '''

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np, os, sys, click, cv2

def calculate_iou(bbox1, bbox2):
    x1_max = max(bbox1[0], bbox2[0])
    y1_max = max(bbox1[1], bbox2[1])
    x2_min = min(bbox1[2], bbox2[2])
    y2_min = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def extract_prediction_values_with_NMS(results, threshold, h, w, ch, nms_threshold=0.5):
    # Process the predictions to match the annotation format
    try:
        boxes = []
        scores = []
        annotations = []
        for prediction in results.predictions:
            if prediction.probability >= threshold:
                bbox = prediction.bounding_box
                xmin = int(bbox.left * w)
                ymin = int(bbox.top * h)
                xmax = int((bbox.left + bbox.width) * w)
                ymax = int((bbox.top + bbox.height) * h)
                boxes.append([xmin, ymin, xmax, ymax])
                scores.append(prediction.probability)
                annotations.append({
                    "bbox": [xmin, ymin, xmax, ymax],
                    "category": prediction.tag_name,
                    "score": prediction.probability
                })
        if len(boxes) > 0:
            # Convert boxes and scores to the appropriate format for NMS
            boxes = np.array(boxes)
            scores = np.array(scores)
            
            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), threshold, nms_threshold)
            indices = indices.flatten() if len(indices) > 0 else []
            nms_annotations = [annotations[i] for i in indices]
            return nms_annotations
        else:
            return annotations
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception: {e}\nError Type: {exc_type}\nFile Name: {fname}\nLine Number: {exc_tb.tb_lineno}", fg="red")
        return []

def extract_prediction_values_without_NMS(results, threshold, h, w, ch):
    # Process the predictions to match the annotation format
    try:
        annotations = []
        for prediction in results.predictions:
            if prediction.probability >= threshold:
                bbox = prediction.bounding_box
                xmin = int(bbox.left * w)
                ymin = int(bbox.top * h)
                xmax = int((bbox.left + bbox.width) * w)
                ymax = int((bbox.top + bbox.height) * h)
                annotations.append({
                    "bbox": [xmin, ymin, xmax, ymax],
                    "category": prediction.tag_name,
                    "score": prediction.probability
                })
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception: {e}\nError Type: {exc_type}\nFile Name: {fname}\nLine Number: {exc_tb.tb_lineno}", fg="red")
    return annotations


def predict_main(image_file, output_path, threshold, nms_threshold, prediction_client, project_id, model_name, drawBboxFlag):
    try:
        annotations=[]
        # Load image and get height, width and channels
        # print('Detecting objects in', image_file)
        image = Image.open(image_file)
        h, w, ch = np.array(image).shape

        # Detect objects in the test image
        with open(image_file, mode="rb") as image_data:
            results = prediction_client.detect_image(project_id, model_name, image_data)

        # annotations = extract_prediction_values(results=results,
        #                                         threshold=threshold,
        #                                         h=h,
        #                                         w=w,
        #                                         ch=ch)

        annotations = extract_prediction_values_with_NMS(results=results,
                                                         threshold=threshold,
                                                         h=h,
                                                         w=w,
                                                         ch=ch,
                                                         nms_threshold=nms_threshold)

        if drawBboxFlag:
            draw_bounding_boxes_with_legend(image_path=image_file,
                                            annotations=annotations,
                                            output_path=output_path,
                                            bbox_color='red',
                                            bbox_thickness=2,
                                            text_color='white',
                                            font_size=2)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception: {e}\nError Type: {exc_type}\nFile Name: {fname}\nLine Number: {exc_tb.tb_lineno}", fg="red")
    return annotations



##########################################################################

import numpy as np, os, sys, click, cv2
from matplotlib import pyplot as plt

def draw_bounding_boxes_with_legend(image_path, annotations, output_path, bbox_color, bbox_thickness, text_color, font_size):
    try:
        """Draw bounding boxes on the given image based on annotations and create a legend showing the colors used."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        # Define color map for bounding box and text colors
        color_map = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0)
        }
        # Convert colors from strings to BGR tuples
        bbox_color = color_map.get(bbox_color.lower(), (0, 0, 255))
        text_color = color_map.get(text_color.lower(), (255, 255, 255))
        # Initialize a dictionary to map categories to colors
        category_colors = {}
        used_colors = set()
        color_index = 0
        # List of colors for categories
        colors_list = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255)  # Blue
        ]
        # Assign colors to categories and draw bounding boxes
        for annotation in annotations:
            xmin, ymin, xmax, ymax = annotation["bbox"]
            category = annotation["category"]
            # Assign a color to the category if not already assigned
            if category not in category_colors:
                # Use a color from the list and move to the next one
                if color_index >= len(colors_list):
                    color_index = 0  # Loop back if we've used all colors
                category_colors[category] = colors_list[color_index]
                color_index += 1
            # Get the color for the current category
            color = category_colors[category]
            # Draw the bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, bbox_thickness)
        # Draw the legend
        x_legend = image.shape[1] - 300  # Adjust the x position of the legend
        y_legend = 30  # Initial y position for the legend
        line_height = 30  # Adjust the height of each line in the legend
        for category, color in category_colors.items():
            # Draw a square with the category color
            cv2.rectangle(image, (x_legend, y_legend), (x_legend + 15, y_legend + 15), color, -1)
            # Find the maximum confidence score for the category
            max_score = max(annotation["score"] for annotation in annotations if annotation["category"] == category)
            # Draw the label text next to the square with confidence score
            cv2.putText(image, f"{category} ({max_score:.2f})", (x_legend + 20, y_legend + 12), cv2.FONT_HERSHEY_SIMPLEX, font_size * 0.5, text_color, 1)
            # Move down to the next line in the legend
            y_legend += line_height
        # Save the image with bounding boxes and legend
        cv2.imwrite(output_path, image)
        # print(f"Saved output image: {output_path}")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception: {e}\nError Type: {exc_type}\nFile Name: {fname}\nLine Number: {exc_tb.tb_lineno}", fg="red")
        return False
    return True


#######################################################################

import numpy as np, os, sys, click, cv2
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

if __name__ == "__main__":

    #### execution settings
    num_of_images_to_process = 1
    threshold = 0.8
    nms_threshold=0.6
    drawBboxFlag = True

    #### path setup

    PATH = f"C:/data/"
    image_dir = f"{PATH}test/images/"

    current_time = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_dir = f"./results/custom-vision-model-v1-{current_time}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #### Get Configuration Settings
    load_dotenv()
    prediction_endpoint = os.getenv('PredictionEndpoint')
    prediction_key = os.getenv('PredictionKey')
    project_id = os.getenv('ProjectID')
    model_name = os.getenv('ModelName')

    #### Authenticate a client for the training API
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

    #### iterate over the input directory
    counter=0
    image_files = os.listdir(image_dir)
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    for image_file in tqdm(image_files):
        output_path = f"{output_dir}{image_file.split('/')[-1]}"

        predict_main(image_file=f"{image_dir}{image_file}",
                     output_path=output_path,
                     threshold=threshold,
                     nms_threshold=nms_threshold,
                     prediction_client=prediction_client,
                     project_id=project_id,
                     model_name=model_name,
                     drawBboxFlag=drawBboxFlag)
        
        counter+=1
        if counter==num_of_images_to_process:
            break

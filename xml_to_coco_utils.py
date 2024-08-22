import os, sys, click
from io import BytesIO
import xml.etree.ElementTree as ET

def read_xml_from_blob(xml_path, container_client):
    try:
        # Load the XML file
        xml_blob_client = container_client.get_blob_client(xml_path)
        xml_blob = xml_blob_client.download_blob().content_as_bytes()
        # Parse the XML content
        xml_content = BytesIO(xml_blob)
        tree = ET.parse(xml_content)
        root = tree.getroot()
        return root
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"\nError Type : {exc_type} ----->> File Name : {fname} ----->> Line Number : {exc_tb.tb_lineno}", fg="red")

def set_key_images_for_coco_file(root, image_id, image_blob_path):
    try:
        image_info = {}
        ''' It sets up the 'images' key for the coco file using the XML content. '''
        # Extract image information
        file_name = root.find('filename').text
        file_name = file_name.split("/")[-1]
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        # Replace local image path with blob path
        blob_path = f"{image_blob_path}"
        # Create image info dictionary
        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name,
            "absolute_url": blob_path
        }
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"\nError Type : {exc_type} ----->> File Name : {fname} ----->> Line Number : {exc_tb.tb_lineno}", fg="red")
    return image_info

def set_key_annotations_for_coco_file(root, category_dict, annotation_id, image_id, coco_data):
    try:
        annotation_info = {}
        # Process annotations for each object in the XML
        # print(f"")
        for obj in root.findall('object'):
            category_name = obj.find('name').text
            # picking data for only required(defined) classes
            if category_name in category_dict:
                category_id = category_dict[category_name]
                annotation_id+=1
                # Calculate bounding box coordinates
                xmin = int(float(obj.find('bndbox/xmin').text))
                ymin = int(float(obj.find('bndbox/ymin').text))
                xmax = int(float(obj.find('bndbox/xmax').text))
                ymax = int(float(obj.find('bndbox/ymax').text))
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                area = bbox_width * bbox_height
                # Create annotation dictionary
                annotation_info = {
                    "id": annotation_id,
                    "category_id": category_id,
                    "image_id": image_id,
                    "area": area,
                    "bbox": [xmin, ymin, bbox_width, bbox_height],
                }
                # Append annotations info to coco_data
                coco_data["annotations"].append(annotation_info)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"\nError Type : {exc_type} ----->> File Name : {fname} ----->> Line Number : {exc_tb.tb_lineno}", fg="red")
    return coco_data, annotation_id


import cv2, io, pandas as pd, click, numpy as np
import xml.etree.ElementTree as ET
from databricks.sdk.runtime import *
from databricks.sdk.runtime import dbutils

# Ensure the directory exists
def ensure_directory_exists(directory_path):
    try:
        dbutils.fs.ls(directory_path)
        print(f"Directory {directory_path} already exists.")
    except:
        dbutils.fs.mkdirs(directory_path)
        print(f"Directory {directory_path} created.")

# Check if a file exists
def check_file_exists(file_path):
    printFlag = False
    try:
        dbutils.fs.ls(file_path)
        if printFlag: print(f"File {file_path} exists.")
        return True
    except:
        if printFlag: print(f"File {file_path} does not exist.")
        return False

def read_file_from_dbfs(file_path):
    # Read the file from DBFS
    with open(f'/dbfs{file_path}', 'rb') as file:
        content = file.read()
    return content

def read_xml_from_dbfs(file_path):
    # Read the XML file
    xml_content = read_file_from_dbfs(file_path)
    # Parse the XML content
    tree = ET.ElementTree(ET.fromstring(xml_content.decode('utf-8')))
    root = tree.getroot()
    # print(f"Root tag: {root.tag}")
    return tree


def read_image_from_dbfs(file_path):
    try:
        printFlag = False
        # Read the image file
        image_content = read_file_from_dbfs(file_path)
        # Convert the image content to a numpy array and decode
        image_array = np.asarray(bytearray(image_content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        # Check if the image was loaded successfully
        if image is None or image.size == 0:
            raise ValueError("Failed to load the image. The image is empty or not a valid image.")
        if printFlag : print(f"Image loaded successfully: {image.shape}")
        return image
    except Exception as e:
        click.secho(f"Error loading image from {file_path} ---> read_image_from_dbfs", fg="red")
        click.secho(f"Error: {e}", fg="red")
        return None
    

def save_file_to_dbfs(content, save_path):
    """
    Save content (e.g., XML or JSON) to the specified path in DBFS.
    Parameters:
        content (str): The content to save (XML, JSON, etc.).
        save_path (str): The DBFS path where the file will be saved.
    """
    try:
        printFlag = False
        # Write the content to the specified path in DBFS
        with open(f'/dbfs{save_path}', 'w') as file:
            file.write(content)
        if printFlag: print(f"File saved to {save_path}")
        return True
    except Exception as e:
        click.secho(f"Unable to save the file in blob ---> save_file_to_dbfs", fg="red")
        return False

def save_image_to_dbfs(image, save_path):
    """
    Save an OpenCV image to the specified path in DBFS.
    Parameters:
        image (numpy.ndarray): The image to save.
        save_path (str): The DBFS path where the image will be saved.
    """
    try:
        printFlag = False
        if image is None or image.size == 0:
            raise ValueError("The provided image is empty or not valid.")
        # Encode the image as JPEG in memory
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Could not encode image as JPEG")
        image_bytes = buffer.tobytes()
        # Write the image bytes to the specified path in DBFS
        with open(f'/dbfs{save_path}', 'wb') as file:
            file.write(image_bytes)
        if printFlag: click.secho(f"Image saved to {save_path}", fg="blue")
        return True
    except Exception as e:
        click.secho(f"Unable to save the image in blob ---> save_image_to_dbfs", fg="red")
        click.secho(f"Error: {e}", fg="red")
        return False


def save_excel_to_dbfs(df, save_path):
    with pd.ExcelWriter(f'/dbfs{save_path}', engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    print(f"Excel file saved to {save_path}")

# Function to save CSV file
def save_csv_to_dbfs(df, save_path):
    printFlag = True
    df.to_csv(f'/dbfs{save_path}', index=False)
    if printFlag: click.secho(f"\nCSV file saved to {save_path}", fg="green")


if __name__ == "__main__":
    
    # Example usage
    blob_name = "AddDatas20240604"
    input_blob = f'{blob_name}/Annotations/'
    annotation_blob = f'/mnt/kubota-test-dataset/{input_blob}'
    xml_path = '16_Camera2_198_00588.xml'
    xml_input_blob_file_path = f"{annotation_blob}{xml_path}"

    # Read XML file
    xml_root = read_xml_from_dbfs(xml_input_blob_file_path)

    # Read image file
    image_blob = f'{blob_name}/JPEGImages/'
    image_blob_path = f'/mnt/kubota-test-dataset/{image_blob}'
    image_path = '16_Camera2_198_00588.jpg'
    image_input_blob_file_path = f"{image_blob_path}{image_path}"

    image = read_image_from_dbfs(image_input_blob_file_path)

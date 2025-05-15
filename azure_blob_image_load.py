!pip install azure-storage-blob opencv-python

AZURE_STORAGE_CONNECTION_STRING = ""
BLOB_CONTAINER_NAME = ""
BLOB_FOLDER_PATH = "path_of_folder/path_of_sub_folder/"  # folder inside the container


import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient
from io import BytesIO

def load_image_from_blob(blob_client):
    """
    Downloads and decodes a single image blob.
    Returns: (image array or None)
    """
    try:
        stream = blob_client.download_blob()
        blob_bytes = stream.readall()

        # Convert bytes to NumPy array
        np_array = np.frombuffer(blob_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is not None:
            return image
        else:
            print("Warning: Image could not be decoded.")
            return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def load_images_from_blob_folder(connection_string, container_name, folder_path):
    """
    Loads all images from a blob folder path and returns a list of (filename, image array) tuples.
    """
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    blob_list = container_client.list_blobs(name_starts_with=folder_path)
    images = []

    for blob in blob_list:
        if blob.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            blob_client = container_client.get_blob_client(blob.name)
            image = load_image_from_blob(blob_client)

            if image is not None:
                images.append((blob.name, image))
                print(f"{blob.name} loaded with shape: {image.shape}")
            else:
                print(f"Warning: Failed to load image from {blob.name}")
    return images


if __name__ == "__main__":

    images = load_images_from_blob_folder(
        AZURE_STORAGE_CONNECTION_STRING,
        BLOB_CONTAINER_NAME,
        BLOB_FOLDER_PATH
    )

    # You can now process the list `images` as needed

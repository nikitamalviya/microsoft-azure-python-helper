import os, requests
from azure.storage.blob import BlobServiceClient


## move_blurred_images.py
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=abcd;AccountKey=O/qoQg==;EndpointSuffix=core.windows.net"
COMMON_CONTAINER_NAME = "blur-detection"
DATASET_FOLDER_NAME = "blur-dataset-350/motion_blurred/"
SOURCE_BLOB_PREFIX = f"datasets/{DATASET_FOLDER_NAME}"
DEST_BLOB_PREFIX = f"outputs/fastapi_outputs/{DATASET_FOLDER_NAME}"


# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
# Track blobs that weren't found
missing_blobs = []


def get_source_blob_name(image_path: str):
    return image_path  # Already like: 'datasets/.../image.jpg

def get_destination_blob_name(image_path: str):
    return image_path.replace("datasets/", "outputs/fastapi_outputs/")

def blob_exists(container, blob_name):
    try:
        blob = blob_service_client.get_blob_client(container=container, blob=blob_name)
        blob.get_blob_properties()
        return True
    except Exception as e:
        return False

def copy_blob(image_path: str):
    try:
        source_blob_name = get_source_blob_name(image_path)
        dest_blob_name = get_destination_blob_name(image_path)

        source_blob = blob_service_client.get_blob_client(container=COMMON_CONTAINER_NAME, blob=source_blob_name)

        # Check existence before attempting copy
        if not blob_exists(COMMON_CONTAINER_NAME, source_blob_name):
            raise FileNotFoundError("Source blob not found")

        dest_blob = blob_service_client.get_blob_client(container=COMMON_CONTAINER_NAME, blob=dest_blob_name)
        copy_source_url = source_blob.url
        copy_result = dest_blob.start_copy_from_url(copy_source_url)

        print(f" ********** Copied: {source_blob_name} → {dest_blob_name} | Status: {copy_result['copy_status']}")
        return True
    except Exception as e:
        print(f"######## Copy Error: {e}\nImage Path: {image_path} ########")
        missing_blobs.append(image_path)
        return False

def move_blob(image_path: str):
    if copy_blob(image_path):
        try:
            source_blob_name = get_source_blob_name(image_path)
            source_blob = blob_service_client.get_blob_client(container=COMMON_CONTAINER_NAME, blob=source_blob_name)
            source_blob.delete_blob()
            print(f"###### Deleted: {source_blob_name}")
        except Exception as e:
            print(f"###### Error deleting source blob: {e}")

# === MAIN ===
def main_fetch_and_process_blobs(move=False):
    try:
        response = requests.get(FASTAPI_URL)
        response.raise_for_status()
        data = response.json()

        print(f"\n ######## Total Records Fetched: {len(data)}")

        for record in data:
            image_path = record.get("image")
            if not image_path:
                continue

            print(f"\n***** Processing image: {image_path} *****")
            if move:
                move_blob(image_path)
            else:
                copy_blob(image_path)

            break  # Remove this once testing is done

        # Summary
        if missing_blobs:
            print("\n ********** Summary of Missing Images:")
            print(f"Total missing: {len(missing_blobs)}")
            for img in missing_blobs:
                print(f" - {img}")
        else:
            print("\n########## All images processed successfully.")

    except Exception as e:
        print(f"######### API/Processing Error: {e}")

# === RUN ===
if __name__ == "__main__":
    # Test one image
    # main_fetch_and_process_blobs(move=False)

    # To test move:
    main_fetch_and_process_blobs(move=True)















'''
# def extract_blob_path(image_path):
#     """
#     Extracts blob path relative to container from full image path.
#     E.g., 'datasets/blur-dataset-350/motion_blurred/image.jpg' → 'blur-dataset-350/motion_blurred/image.jpg'
#     """
#     return "/".join(image_path.split("/")[1:])

# ------------------ FETCH FROM API ------------------
def ai_call_fetch_blurred_images():
    try:
        response = requests.get(FASTAPI_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch images: {e}")
        return []

# ------------------ MOVE BLOB FUNCTION ------------------
def move_blob(image_name):
    try:
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        source_blob = blob_service.get_blob_client(container=SOURCE_CONTAINER, blob=image_name)

        destination_blob_path = get_destination_path(image_name)
        destination_blob = blob_service.get_blob_client(container=DESTINATION_CONTAINER, blob=destination_blob_path)

        # Construct source URL
        source_url = source_blob.url

        # Start copy operation
        destination_blob.start_copy_from_url(source_url)

        # Optionally, delete the original blob (simulate "move")
        source_blob.delete_blob()

        return True, destination_blob_path
    except Exception as e:
        return False, str(e)

# ------------------ MAIN FUNCTION ------------------
def main():
    fetched_data = ai_call_fetch_blurred_images()

    print(fetched_data)
    exit()

    if not images:
        print("No images to process.")
        return

    for img in images:
        image_name = img["image"]
        status, message = move_blob(image_name)
        if status:
            print(f"[SUCCESS] Moved: {image_name} → {message}")
        else:
            print(f"[FAILED] {image_name}: {message}")

'''

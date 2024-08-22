%pip install azure-storage-blob importlib
# dbutils.library.restartPython()
from azure.storage.blob import BlobServiceClient

# Azure Storage Account credentials
storage_account_name = ''
storage_account_key = ''

# Containers
container_name = "container_name"
images_path = f'ContainerName/set1/imagespath/'
labels_path = f'ContainerName/set1/annotationspath/'
coco_filename = "coco_file.json"
coco_path = f"ContainerName/set1/{coco_filename}"

# Initialize the BlobServiceClient
connection_string = f'DefaultEndpointsProtocol=https;AccountName={storage_account_name};AccountKey={storage_account_key};EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Access the container
container_client = blob_service_client.get_container_client(container_name)
print(container_client)


## Check path exists or not
def path_exists(container_client, path):
    try:
        # Try listing blobs under the given path
        blob_list = list(container_client.list_blobs(name_starts_with=path))
        if blob_list:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error checking path {path}: {e}")
        return False

def check_sample_file(container_client, path):
    try:
        blob_list = list(container_client.list_blobs(name_starts_with=path))
        if not blob_list:
            print(f"No files found in path: {path}")
            return False
        # Try to access a sample file
        sample_blob = blob_list[0]
        # print(f"\n\n-- Sample blob : {sample_blob}\n")
        blob_client = container_client.get_blob_client(sample_blob.name)
        blob_client.download_blob().content_as_bytes()
        return True
    except Exception as e:
        print(f"Sample file not found in path: {path} : {e}")
        return False
    except Exception as e:
        print(f"Error accessing sample file in path {path}: {e}")
        return False
      
''' ################################################################################################## '''

import importlib, json, os, sys
from io import BytesIO
import xml_to_coco_utils
importlib.reload(xml_to_coco_utils)
from xml_to_coco_utils import (read_xml_from_blob,
                               set_key_images_for_coco_file,
                               set_key_annotations_for_coco_file)


if __name__=="__main__":
    
    # Test flag, enable execution for the single file
    LIMIT_FLAG = True
    iter_max = 1400 #10000000000
    iter_count = 0

    ## file setups
    # unique image ID for coco_data["images"]
    image_id = 0
    
    # annotation id
    annotation_id = 0
    category_dict = {"human": 1,
                    "head": 2}
    # Initialize COCO data structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1,"name": "human"},{"id": 2,"name": "head"}]
    }
    
    for training_set in ["set1", "set2", "set3"]:
        # set dataset paths  
        images_path = f'ImageAnalysisDataset/{training_set}/JPEGImages/'
        labels_path = f'ImageAnalysisDataset/{training_set}/Annotations_Human_Conversion/'
        # image blob absolute path
        images_blob_path = f"https://{storage_account_name}.blob.core.windows.net/{container_name}/ImageAnalysisDataset/{training_set}/JPEGImages/"

        ## Load images and xmls to prepare a coco file
        # List and load image paths
        image_blobs = container_client.list_blobs(name_starts_with=images_path)
        image_paths = [blob.name for blob in image_blobs]
        print(f"\nNumber of loaded images from set {training_set} : {len(image_paths)}")

        # Loop over image paths to load the corresponding XML file
        for image_path in image_paths:
            iter_count+=1

            # Derive the XML path from the image path
            image_filename = image_path.split('/')[-1]
            xml_filename = image_filename.replace('.jpg', '.xml')  # Assuming images are in .jpg format
            xml_path = f'{labels_path}/{xml_filename}'
            
            # Read the XML file content
            root = read_xml_from_blob(xml_path, container_client)
            print("XML root : ", root)

            # prepare content for key : images for this file
            image_id+=1
            
            image_blob_path = images_blob_path + image_filename   
            image_info = set_key_images_for_coco_file(root, image_id, image_blob_path)
            # Append image info to coco_data
            coco_data["images"].append(image_info)
            print("image_info :\n", image_info)

            # prepare content for key : annotations for this file
            # extract XML file content and prepare key : annotations
            coco_data, annotation_id = set_key_annotations_for_coco_file(root, category_dict, annotation_id, image_id, coco_data)
            # print("xml_info :\n", xml_info)

            if LIMIT_FLAG and iter_max <= iter_count:
                break
            
        ## Create a sub-coco file
        if not LIMIT_FLAG:
            try:
                # set output coco file path
                coco_filename = f"training_{training_set}_{iter_count}.json"
                coco_path = f"ImageAnalysisDataset/training_coco/{coco_filename}"
                # dump JSON data
                json_data = json.dumps(coco_data)
                # Upload the JSON content to the specified path
                blob_client = container_client.get_blob_client(coco_path)
                blob_client.upload_blob(json_data, overwrite=True)
                print(f"\n\nJSON file created and uploaded to {coco_path} successfully.")
                # save the coco file in coco directory workspace
                with open(f"./coco/{coco_filename}", 'w') as f:
                    json.dump(coco_data, f, indent=2)
                    print(f"\nSaved COCO to the local path ./coco/{coco_filename}")
            except Exception as e:
                print(f"\n\nFailed to upload JSON file to {coco_path}. Error: {e}")

        if LIMIT_FLAG and iter_max <= iter_count:
            break

    ## Create a general COCO file
    try:
        # set output coco file path
        coco_filename = f"training_{iter_count}.json"
        coco_path = f"ImageAnalysisDataset/training_coco/{coco_filename}"
        # dump JSON data
        json_data = json.dumps(coco_data)
        # Upload the JSON content to the specified path
        blob_client = container_client.get_blob_client(coco_path)
        blob_client.upload_blob(json_data, overwrite=True)
        print(f"\n\nJSON file created and uploaded to {coco_path} successfully.")
        # save the coco file in coco directory workspace
        with open(f"./coco/{coco_filename}", 'w') as f:
            json.dump(coco_data, f, indent=2)
            print(f"\nSaved COCO to the local path ./coco/{coco_filename}")
    except Exception as e:
       print(f"\n\nFailed to upload JSON file to {coco_path}. Error: {e}")

'''
# Check if images path exists and can be accessed
if path_exists(container_client, images_path):
    print("Images path exists.")
    if check_sample_file(container_client, images_path):
        print("Successfully accessed a sample image file.")
    else:
        print("Failed to access a sample image file.")
else:
    print("Images path does not exist.")

# Check if labels path exists and can be accessed
if path_exists(container_client, labels_path):
    print("Labels path exists.")
    if check_sample_file(container_client, labels_path):
        print("Successfully accessed a sample label file.")
    else:
        print("Failed to access a sample label file.")
else:
    print("Labels path does not exist.")
'''

## Check if the file exits or not

for dict_ in coco_data["images"]:
    print(dict_["absolute_url"])
    blob_url = dict_["absolute_url"]
    blob_path = "ImageAnalysisDataset/set1/JPEGImages/15_Camera0_74_mini_20230803172723_01437.jpg"
    # blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{blob_path}"

    blob_client = container_client.get_blob_client(blob_path)
    if blob_client.exists():
        print(f"The blob at {blob_url} exists.")
    else:
        print(f"The blob at {blob_url} does not exist.")

    break


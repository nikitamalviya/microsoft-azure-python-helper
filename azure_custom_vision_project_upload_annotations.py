from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import time, json, os, sys, click
from tqdm import tqdm

# os.environ['NO_PROXY'] = 'customvision.ai'

def main(image_folder, output_json_file):
    from dotenv import load_dotenv
    global training_client
    global custom_vision_project
    try:
        # Get Configuration Settings
        load_dotenv()
        training_endpoint = os.getenv('TrainingEndpoint')
        training_key = os.getenv('TrainingKey')
        project_id = os.getenv('ProjectID')

        # Authenticate a client for the training API
        credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
        training_client = CustomVisionTrainingClient(training_endpoint, credentials)
        
        click.secho(f"Successful authentication.", fg="green")
        
        # Get the Custom Vision project
        print("Getting the Custom Vision project using project id....")
        custom_vision_project = training_client.get_project(project_id)

        click.secho(f"Successful got the custom vision project.", fg="green")

        # Upload and tag images
        Upload_Images(image_folder, output_json_file)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception : {e}\nError Type : {exc_type}\nFile Name : {fname}\nLine Number : {exc_tb.tb_lineno}", fg="red")


def Upload_Images(folder, output_json_file):
    try:
        # Get the tags defined in the project
        tags = training_client.get_tags(custom_vision_project.id)
        click.secho(f"\nRetrieved the tags from the project : {tags}", fg="green")
        # Create a list of images with tagged regions
        tagged_images_with_regions = []

        # Get the images and tagged regions from the JSON file
        with open(output_json_file, 'r') as json_file:
            tagged_images = json.load(json_file)

            for image in tqdm(tagged_images['files']):
                # Get the filename
                file = image['filename']
                # Get the tagged regions
                regions = []
                for tag in tqdm(image['tags']):
                    tag_name = tag['tag']
                    # print("\ntag_name : ", tag_name, file)
                    # Look up the tag ID for this tag name
                    tag_id = None
                    for t in tags:
                        if t.name == tag_name:
                            tag_id = t.id
                            break
                    # Check if tag_id is not None
                    if tag_id is None:
                        print(f"No tag found for tag name: {tag_name} in file {file}..")
                        continue
                    # Add a region for this tag using the coordinates and dimensions in the JSON
                    regions.append(Region(tag_id=tag_id, left=tag['left'], top=tag['top'], width=tag['width'], height=tag['height']))
                
                # Add the image and its regions to the list
                # print(f"\n---------------- {os.path.join(folder, file)}")
                with open(os.path.join(folder, file), mode="rb") as image_data:
                    tagged_images_with_regions.append(
                        ImageFileCreateEntry(name=file, contents=image_data.read(), regions=regions)
                    )
        click.secho(f"Successfully prepared tagged images with regions...", fg="green")

        # Upload the list of images as a batch
        print("Uploading images...")
        upload_result = training_client.create_images_from_files(custom_vision_project.id,
                                                                 ImageFileCreateBatch(images=tagged_images_with_regions))
        # Check for failure
        if not upload_result.is_batch_successful:
            print("Image batch upload failed.")
            for image in tqdm(upload_result.images):
                print("Image status: ", image.status)
        else:
            print("Images uploaded.")
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception: {e}\nError Type: {exc_type}\nFile Name: {fname}\nLine Number: {exc_tb.tb_lineno}", fg="red")

if __name__ == "__main__":
    
    PATH = "C:/data/"
    image_folder = f"{PATH}train/images/"    
    json_file_path = f"{PATH}JSONs/"

    if os.path.exists(image_folder):
        # load all the JSONs and upload the annotations on Custom Vision project        
        for output_json_file in os.listdir(json_file_path):
            # output_json_file = f"{json_file_path}kubota-train_chunk1.json"
            output_json_file = json_file_path + output_json_file
            # upload the data
            main(image_folder, output_json_file)
            click.secho(f"Data uploaded from {output_json_file}", fg="green")
    else:
        click.secho(f"Image folder not found : {image_folder}", fg="red")

import os, sys, click, json
import xml.etree.ElementTree as ET
from tqdm import tqdm

def save_json_in_chunks(json_data, output_dir, output_json_file_prefix, max_sub_dicts):
    """
    Save the given JSON data into chunks based on the specified maximum number of sub-dictionaries per chunk.

    :param json_data: Dictionary containing the original JSON data
    :param output_json_file_prefix: Prefix for the output JSON file names
    :param max_sub_dicts: Maximum number of sub-dictionaries to include in each JSON file
    """
    # List of sub-dictionaries in the JSON data
    files_list = json_data.get("files", [])
    # Number of chunks required
    num_chunks = (len(files_list) + max_sub_dicts - 1) // max_sub_dicts  # Ceiling division

    for i in range(num_chunks):
        # Calculate start and end indices for the current chunk
        start_index = i * max_sub_dicts
        end_index = min((i + 1) * max_sub_dicts, len(files_list))
        # Create a new dictionary for the current chunk
        chunk_data = {
            "files": files_list[start_index:end_index]
        }
        # Construct the output file name
        chunk_filename = f"{output_json_file_prefix}_chunk{i + 1}.json"
        
        # Save the current chunk data as a JSON file
        with open(output_dir + chunk_filename, 'w') as json_file:
            json.dump(chunk_data, json_file, indent=4)
        # Print a message indicating the chunk has been saved
        print(f"Chunk {i + 1} saved as {chunk_filename}")
    click.secho(f"\nDivided the JSON into {i} sub-JSON files.", fg="green")


def parse_xml_file_with_absolute_pixel_values(xml_path, classes, xmls_with_required_classes_list, xmls_without_required_classes_list):
    """Parse an XML file and return bounding box annotations."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tags = []
    # flag to check of the xml file got the required classes or not (head, standing, crouching)
    required_classes_found_flag = False

    for obj in root.findall('object'):
        category = obj.find('name').text
        if category in classes:
            required_classes_found_flag = True

            bndbox = obj.find('bndbox')
            # Get the absolute coordinates
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            # Calculate width and height
            width = xmax - xmin
            height = ymax - ymin

            # Add bounding box info to tags list
            tag = {
                "tag": category,
                "left": xmin,
                "top": ymin,
                "width": width,
                "height": height
            }
            tags.append(tag)
            
    if required_classes_found_flag:
        xmls_with_required_classes_list.append(xml_path)
    else:
        xmls_without_required_classes_list.append(xml_path)
    return tags, xmls_with_required_classes_list, xmls_without_required_classes_list


def parse_xml_file(xml_path, classes, xmls_with_required_classes_list, xmls_without_required_classes_list):
    """Parse an XML file and return bounding box annotations."""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tags = []
    # Flag to check if the XML file contains the required classes
    required_classes_found_flag = False
    
    # Get the image dimensions (width and height) from the XML
    size = root.find('size')
    if size is None:
        raise ValueError(f"Image size not found in XML file: {xml_path}")
    
    # Parse the image dimensions from the XML
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    
    for obj in root.findall('object'):
        category = obj.find('name').text
        if category in classes:
            required_classes_found_flag = True
            
            bndbox = obj.find('bndbox')
            # Get the absolute coordinates
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Calculate width and height
            width = xmax - xmin
            height = ymax - ymin
            
            # Convert absolute coordinates to proportions
            left_proportion = xmin / image_width
            top_proportion = ymin / image_height
            width_proportion = width / image_width
            height_proportion = height / image_height
            
            # Add bounding box information to tags list
            tag = {
                "tag": category,
                "left": left_proportion,
                "top": top_proportion,
                "width": width_proportion,
                "height": height_proportion
            }
            tags.append(tag)
    
    if required_classes_found_flag:
        xmls_with_required_classes_list.append(xml_path)
    else:
        xmls_without_required_classes_list.append(xml_path)
    
    return tags, xmls_with_required_classes_list, xmls_without_required_classes_list


def convert_xml_to_json(image_dir, annotation_dir, json_file_path, output_json_filename, max_sub_dicts, classes):
    try:
        """Convert multiple XML files to one JSON file."""
        # dict to store the data in JSON format
        json_data = {"files": []}
        # path of images not having XML present for it
        xml_not_found = []
        # xmls having the required classes
        xmls_with_required_classes_list = []
        # xmls not having the required classes
        xmls_without_required_classes_list = []

        # Iterate over XML files in the annotation directory
        for xml_file in tqdm(os.listdir(annotation_dir)):
            if xml_file.endswith('.xml'):
                xml_path = os.path.join(annotation_dir, xml_file)
                
                # Parse XML file
                tags, xmls_with_required_classes_list, xmls_without_required_classes_list = parse_xml_file(xml_path,
                                                                                                        classes,
                                                                                                        xmls_with_required_classes_list,
                                                                                                        xmls_without_required_classes_list)
                
                if len(tags) > 0:
                    # Get the corresponding image filename
                    image_filename = os.path.splitext(xml_file)[0] + '.jpg'
                    
                    # Check if the image file exists in the image directory
                    image_path = os.path.join(image_dir, image_filename)
                    if not os.path.exists(image_path):
                        # print(f"Image file '{image_filename}' not found for XML '{xml_file}'. Skipping...")
                        xml_not_found.append(image_filename)
                        continue

                    # adding relative image path
                    # Add the image data and tags to the JSON structure
                    json_data["files"].append({
                        "filename": image_path,
                        "tags": tags
                    })
        
        # Save the JSON data to the specified output file
        # with open(output_json_file, 'w') as json_file:
        #     json.dump(json_data, json_file, indent=4)
        
        # Call the function to save the JSON data in chunks
        save_json_in_chunks(json_data,
                            json_file_path,
                            output_json_filename,
                            max_sub_dicts)

        print(f"Converted XML annotations to JSON and saved to '{json_file_path}'.")
        print(f"\n\nXMLs not found for images : \n{xml_not_found}\nXMLs not found count : {len(xml_not_found)}")
        print(f"\nxmls_with_required_classes_list : {len(xmls_with_required_classes_list)}\n\nxmls_without_required_classes_list :\n{xmls_without_required_classes_list}, {len(xmls_without_required_classes_list)}")

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception : {e}\nError Type : {exc_type}\nFile Name : {fname}\nLine Number : {exc_tb.tb_lineno}", fg="red")

if __name__ == "__main__":
    
    try:
        PATH = "C:/main/"
        
        # classes to consider
        classes = ["class1", "class2", "class3"]

        # Specify the paths to the image and annotation directories and the output JSON file        
        image_dir = f"{PATH}train/images/"
        annotation_dir = f'{PATH}train/labels/'
        json_file_path = f"{PATH}JSONs/"
        print("\nValidate paths : ", os.path.exists(annotation_dir), os.path.exists(image_dir),"\n")

        # to save JSON
        if not os.path.exists(json_file_path):
            os.makedirs(json_file_path)
        output_json_filename = f"output-file"
        # Divide the JSON file data and store only defined JSON chunks in a single JSON file
        max_sub_dicts = 20 # this creates a JSON having 20 images only, to upload

        # Convert XML annotations to JSON
        convert_xml_to_json(image_dir,
                            annotation_dir,
                            json_file_path,
                            output_json_filename,
                            max_sub_dicts,
                            classes)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        click.secho(f"Exception : {e}\nError Type : {exc_type}\nFile Name : {fname}\nLine Number : {exc_tb.tb_lineno}", fg="red")


# -- test
# image_dir = 'C:/data/images/'
# annotation_dir = 'C:/data/annotations/'

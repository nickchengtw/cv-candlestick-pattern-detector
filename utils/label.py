import xml.etree.ElementTree as ET
from pathlib import Path

def load_labels(train_root, xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    result = []
    
    # Iterate through images
    for image_tag in root.find('images'):  # Find the <images> tag
        filename = image_tag.attrib['file']  # Get the file attribute
        image_file = Path(train_root) / filename  # Construct full path

        # Extract bounding boxes
        bounding_boxes = []
        for box in image_tag.findall('box'):  # Find all <box> tags within the image
            top = int(box.attrib['top'])
            left = int(box.attrib['left'])
            width = int(box.attrib['width'])
            height = int(box.attrib['height'])
            bounding_boxes.append((top, left, width, height))

        # Append the image file and its bounding boxes to the result
        result.append((image_file, bounding_boxes))
    
    return result

def load_label_batches(base_dir, batches, xml_filename):
    result = []
    for batch in batches:
        batch_path = f"{base_dir}/{batch}"
        xml_path = f"{batch_path}/{xml_filename}"
        result += load_labels(batch_path, xml_path)
    return result

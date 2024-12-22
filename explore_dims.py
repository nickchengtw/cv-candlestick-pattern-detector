import xml.etree.ElementTree as ET
import numpy as np

# Path to the DLIB XML file
xml_file = 'training/batch3/10_imglab.xml'

# Parse the XML file
tree = ET.parse(xml_file)
root = tree.getroot()

widths = []
heights = []
# Extract and calculate size and aspect ratio
for image in root.find('images'):  # Find the <images> tag
    image_file = image.attrib['file']  # Get the file attribute
    print(f"Image: {image_file}")
    
    for box in image.findall('box'):  # Find all <box> tags within the image
        width = int(box.attrib['width'])
        height = int(box.attrib['height'])
        
        # Calculate size and aspect ratio
        size = width * height
        aspect_ratio = width / height
        
        # Extract box coordinates for reference
        # top = box.attrib['top']
        # left = box.attrib['left']

        widths.append(width)
        heights.append(height)
        
        # print(f"  Box - Top: {top}, Left: {left}, Width: {width}, Height: {height}", end='')
        print(f"    Size: {size}, Aspect Ratio: {aspect_ratio:.2f}")

# compute the average of both the width and height lists
(avgWidth, avgHeight) = (np.mean(widths), np.mean(heights))
print("[INFO] avg. width: {:.2f}".format(avgWidth))
print("[INFO] avg. height: {:.2f}".format(avgHeight))
print("[INFO] aspect ratio: {:.2f}".format(avgWidth / avgHeight))
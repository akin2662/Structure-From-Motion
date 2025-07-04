import cv2
import os
import re

"""Function to get keypoints and descriptors using SIFT of an input image."""
def get_keypoints_and_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)  # Detect keypoints and descriptors using SIFT
    return keypoints, descriptors

"""Function to get the RGB values at a given keypoint location in an image."""
def get_rgb(image, x, y):
    if y >= image.shape[0] or x >= image.shape[1]:  # Check if the keypoint is within the image bounds
        return (0, 0, 0)
    return image[y, x, :]

"""Function to save the keypoints and descriptors matches to a file."""
def save_keypoints_descriptors_matches(filename, images_data, all_matches):
    global length
    with open(filename, 'w') as f:
        # Iterate through all images and their keypoints/descriptors
        for img_name, (img, keypoints, descriptors) in images_data.items():
            for i, (kp, des) in enumerate(zip(keypoints, descriptors)):
                # Check if the keypoint has matches and filter out those with only one match
                if img_name in all_matches and i in all_matches[img_name]:
                    matches = all_matches[img_name][i]
                    if len(matches) < 1:
                        continue  # Skip keypoints with less than two matches

                    # Get the RGB values at the keypoint location
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    rgb = get_rgb(img, x, y)
                    rgb_str = ' '.join(map(str, rgb))

                    # Prepare the match information string
                    match_info_str = ''
                    num_matches = len(matches) + 1  # Count itself and the matches

                    for match in matches:
                        match_info_str += f'{match[0]} {match[1][0]} {match[1][1]} '

                    # Write the keypoint and match information to the file
                    f.write(f'{num_matches} {rgb_str} {kp.pt[0]} {kp.pt[1]} {match_info_str}\n')
                    length += 1


"""Function to prepend a line to the top of a file."""
def prepend_line_to_file(file_path, line):
    # Read the existing contents of the file
    with open(file_path, 'r') as file:
        contents = file.readlines()
    
    # Prepend the new line
    contents.insert(0, line + '\n')
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.writelines(contents)

"""Function to process images in a directory, compute keypoints/descriptors, and find matches."""
def process_images(image_dir):
    global length
    images_data = {}
    all_matches = {}
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # Create a BFMatcher object

    # Read images from the directory and compute keypoints and descriptors
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
            keypoints, descriptors = get_keypoints_and_descriptors(gray)  # Get keypoints and descriptors
            images_data[filename] = (img, keypoints, descriptors)
    # Find matches between different images using Lowe's ratio test
    for img_name, (img, keypoints, descriptors) in images_data.items():
        all_matches[img_name] = {}
        for other_img_name, (other_img, other_keypoints, other_descriptors) in images_data.items():
            img_num = int(re.findall(r'\d+', img_name)[0])
            other_img_num = int(re.findall(r'\d+', other_img_name)[0])

            if int(img_name[:1]) >= int(other_img_name[:1]):
                continue
            
            # Apply KNN matching
            knn_matches = bf.knnMatch(descriptors, other_descriptors, k=2)

            # Apply Lowe's ratio test
            for m_n in knn_matches:
                if len(m_n) == 2 and m_n[0].distance < 0.75 * m_n[1].distance:
                    best_match = m_n[0]
                    if best_match.queryIdx not in all_matches[img_name]:
                        all_matches[img_name][best_match.queryIdx] = []
                    all_matches[img_name][best_match.queryIdx].append((other_img_name[:1], (other_keypoints[best_match.trainIdx].pt[0], other_keypoints[best_match.trainIdx].pt[1])))

    # Save matches to files and prepend header
    for img_name in images_data.keys():
        matches_file = os.path.join(image_dir, f'matching{img_name[:-4]}.txt')
        if(len(images_data[img_name][1]) == 0):
            continue
        save_keypoints_descriptors_matches(matches_file, {img_name: images_data[img_name]}, all_matches)
        prepend_line_to_file(matches_file, f'nFeatures:{length}')
        length = 0

# Specify the directory containing the images
length = 0
image_dir = r'D:/SIFT_BFMatcher/'
process_images(image_dir)

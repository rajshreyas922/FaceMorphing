import dlib
import numpy as np
import cv2

# Calculate the margins to crop images to match their sizes
def calculate_margin(img0, img1):
    size0 = img0.shape
    size1 = img1.shape
    width0, height1 = size0[:2] # Only need the first two elements of the tuple
    width1, height2 = size1[:2]

    # Calculate the differences and averages between the sizes of the two images
    diff0 = abs(width0 - width1) // 2
    diff1 = abs(height1 - height2) // 2
    avg0 = (width0 + width1) // 2
    avg1 = (height1 + height2) // 2

    # Return a list of the calculated values
    return [size0, size1, diff0, diff1, avg0, avg1]


# Crop the larger image to match the size of the smaller image
# If the sizes are the same, return the original images
def crop_image(img0, img1):
    # Calculate the margins between the two images
    [size0, size1, diff0, diff1, avg0, avg1] = calculate_margin(img0, img1)
    width0, height1 = size0[:2] # Only need the first two elements of the tuple
    width1, height2 = size1[:2]

    # If the sizes are the same, return the original images
    if width0 == width1 and height1 == height2:
        return [img0, img1]

    # If img0 is smaller than img1, resize img1 and crop it to match the size of img0
    elif width0 <= width1 and height1 <= height2:
        scale0 = width0 / width1
        scale1 = height1 / height2
        # Choose the larger scale factor to ensure that the entire image is visible after resizing
        if scale0 > scale1:
            res = cv2.resize(img1, None, fx=scale0, fy=scale0, interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img1, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_AREA)
        return crop_image_help(img0, res)

    # If img0 is larger than img1, resize img0 and crop it to match the size of img1
    elif width0 >= width1 and height1 >= height2:
        scale0 = width1 / width0
        scale1 = height2 / height1
        # Choose the larger scale factor to ensure that the entire image is visible after resizing
        if scale0 > scale1:
            res = cv2.resize(img0, None, fx=scale0, fy=scale0, interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img0, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_AREA)
        return crop_image_help(res, img1)

    # If img0 is wider than img1 but shorter, crop both images horizontally
    elif width0 >= width1 and height1 <= height2:
        return [img0[diff0:avg0, :], img1[:, -diff1:avg1]]

    # If img0 is taller than img1 but narrower, crop both images vertically
    else:
        return [img0[:, diff1:avg1], img1[-diff0:avg0, :]]


# Crop the larger image to match the size of the smaller image
# If the sizes are the same, return the original images
def crop_image_help(img0, img1):
    # Calculate the margins between the two images
    [size0, size1, diff0, diff1, avg0, avg1] = calculate_margin(img0, img1)
    width0, height1 = size0[:2] # Only need the first two elements of the tuple
    width1, height2 = size1[:2]
    # If the sizes are the same, return the original images
    if width0 == width1 and height1 == height2:
        return [img0, img1]

    # If img0 is smaller than img1, crop img1 horizontally and vertically
    elif width0 <= width1 and height1 <= height2:
        return [img0, img1[-diff0:avg0, -diff1:avg1]]

    # If img0 is larger than img1, crop img0 horizontally and vertically
    elif width0 >= width1 and height1 >= height2:
        return [img0[diff0:avg0, diff1:avg1], img1]

    # If img0 is wider than img1 but shorter, crop both images horizontally
    elif width0 >= width1 and height1 <= height2:
        return [img0[diff0:avg0, :], img1[:, -diff1:avg1]]

    # If img0 is taller than img1 but narrower, crop both images vertically
    else:
        return [img0[:, diff1:avg1], img1[diff0:avg0, :]]



# Generate corresponding facial features points for two images
def generate_face_correspondences(img0, img1):
    # Initialize the face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Initialize an array to store the facial landmark points
    corrs = np.zeros((68, 2))

    # Crop the two images to match their sizes
    imgList = crop_image(img0, img1)

    # Initialize two lists to store the facial landmark points for each image
    list0 = []
    list1 = []

    # Flag to alternate between the two lists
    flag = True

    for img in imgList:
        # Get the size of the image
        size = (img.shape[0], img.shape[1])

        # Determine which list to use
        if flag:
            tempList = list0
        else:
            tempList = list1

        # Detect the faces in the image using the detector
        dets = detector(img, 1)

        # Update the flag to switch to the other list
        flag = False

        # For each detected face, get the facial landmark points and add them to the current list
        for rect in dets:
            # Get the landmarks/parts for the face in rect.
            shape = predictor(img, rect)

            for i in range(0, 68):
                x = shape.part(i).x
                y = shape.part(i).y
                tempList.append((x, y))
                corrs[i][0] += x
                corrs[i][1] += y

            # Add extra points to each list
            tempList.append((1, 1))
            tempList.append((size[1] - 1, 1))
            tempList.append(((size[1] - 1) // 2, 1))
            tempList.append((1, size[0] - 1))
            tempList.append((1, (size[0] - 1) // 2))
            tempList.append(((size[1] - 1) // 2, size[0] - 1))
            tempList.append((size[1] - 1, size[0] - 1))
            tempList.append(((size[1] - 1), (size[0] - 1) // 2))

    # Add extra points to the midpoint array
    mid_image = corrs / 2
    mid_image = np.append(mid_image, [[1, 1]], axis=0)
    mid_image = np.append(mid_image, [[size[1] - 1, 1]], axis=0)
    mid_image = np.append(mid_image, [[(size[1] - 1) // 2, 1]], axis=0)
    mid_image = np.append(mid_image, [[1, size[0] - 1]], axis=0)
    mid_image = np.append(mid_image, [[1, (size[0] - 1) // 2]], axis=0)
    mid_image = np.append(mid_image, [[(size[1] - 1) // 2, size[0] - 1]], axis=0)
    mid_image = np.append(mid_image, [[size[1] - 1, size[0] - 1]], axis=0)
    mid_image = np.append(mid_image, [[(size[1] - 1), (size[0] - 1) // 2]], axis=0)

    # Return the size of the images, the two cropped images, the two lists of facial landmark points, and the midpoint array
    return [size, imgList[0], imgList[1], list0, list1, mid_image]

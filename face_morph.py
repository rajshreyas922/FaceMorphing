import numpy as np
import cv2

#Does not work
def getAffineTransform(srcTri, dstTri):
    # Create matrix A with srcTri coordinates
    A = np.array([
        [srcTri[0, 0], srcTri[0, 1], 1, 0, 0, 0],
        [0, 0, 0, srcTri[0, 0], srcTri[0, 1], 1],
        [srcTri[1, 0], srcTri[1, 1], 1, 0, 0, 0],
        [0, 0, 0, srcTri[1, 0], srcTri[1, 1], 1],
        [srcTri[2, 0], srcTri[2, 1], 1, 0, 0, 0],
        [0, 0, 0, srcTri[2, 0], srcTri[2, 1], 1]
    ])

    # Create vector B with dstTri coordinates
    B = np.array([
        dstTri[0, 0],
        dstTri[0, 1],
        dstTri[1, 0],
        dstTri[1, 1],
        dstTri[2, 0],
        dstTri[2, 1]
    ])

    # Solve the linear system Ax = B to find the transformation matrix coefficients
    coeffs = np.linalg.solve(A, B)

    # Reshape the coefficients to form the 2x3 transformation matrix
    warpMat = np.array([
        [coeffs[0], coeffs[1], coeffs[2]],
        [coeffs[3], coeffs[4], coeffs[5]]
    ])

    return warpMat

# Define a function to apply affine transformation
def apply_affine_transform(src, srcTri, dstTri, size):
    # Convert the input triangles to numpy arrays
    srcTri = np.array(srcTri, dtype=np.float32)
    dstTri = np.array(dstTri, dtype=np.float32)

    # Calculate the transformation matrix
    warpMat = cv2.getAffineTransform(srcTri, dstTri)

    # Apply the transformation matrix to the input image
    dst = cv2.warpAffine(src, warpMat, size)

    # Return the transformed image
    return dst

def morph_triangle(img0, img1, img, t0, t1, t, alpha):

    # get the bounding rectangles for the triangles
    r0 = cv2.boundingRect(np.array([t0], dtype=np.float32))
    r1 = cv2.boundingRect(np.array([t1], dtype=np.float32))
    r = cv2.boundingRect(np.array([t], dtype=np.float32))

    t0Rect = []
    t1Rect = []
    tRect = []

    # get the vertices for the triangles within the bounding rectangles
    for i in range(3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t0Rect.append(((t0[i][0] - r0[0]), (t0[i][1] - r0[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))

    # create a mask for the triangular region
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # get the rectangular regions of interest in the input images and warp them to match the triangle in the output image
    img0Rect = img0[r0[1]:r0[1]+r0[3], r0[0]:r0[0]+r0[2]]
    img1Rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    size = (r[2], r[3])
    warpImage0 = apply_affine_transform(img0Rect, t0Rect, tRect, size)
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)

    # blend the two warped images according to the alpha parameter
    imgRect = (1.0 - alpha) * warpImage0 + alpha * warpImage1

    # blend the rectangular region in the output image with the warped triangular region using the mask
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + imgRect * mask


# Define a function to generate a morphed image
def morph(img0, img1, points0, points1, tri_list, alpha=0.5):
    # Convert the input images to float data
    img0 = np.float32(img0)
    img1 = np.float32(img1)

    # Create a list of weighted average point coordinates
    points = []
    for i in range(len(points0)):
        x = (1 - alpha) * points0[i][0] + alpha * points1[i][0]
        y = (1 - alpha) * points0[i][1] + alpha * points1[i][1]
        points.append((x, y))

    # Create an output image
    morphed_img = np.zeros(img0.shape, dtype=img0.dtype)
    # Loop over each triangle in the triangle list and morph it
    for i in range(len(tri_list)):
        x, y, z = int(tri_list[i][0]), int(tri_list[i][1]), int(tri_list[i][2])
        t0, t1 = [points0[x], points0[y], points0[z]], [points1[x], points1[y], points1[z]]
        t = [points[x], points[y], points[z]]
        morph_triangle(img0, img1, morphed_img, t0, t1, t, alpha)

    # Convert the morphed image back to unsigned 8-bit integers
    morphed_img = np.uint8(morphed_img)

    # Return the morphed image
    return morphed_img

import numpy as np
import cv2
import math

""" auxilary functions """
# unwarp corodinates
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])
""" end of auxilary functions """


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


def save_results(img, score_text, score_link, text_threshold, link_threshold, low_text, ratio_w, ratio_h):
    # Post-processing: Get Detected Bouding boxes
    # use thresholding to filter predictions based on text_threshold, low text bound and link threshold
    boxes, polys, mapper = getDetBoxes_core(score_text, score_link, text_threshold, link_threshold, low_text)

    # Adjust Coordinates
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    # original image
    img = np.array(img)

    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        
        poly = poly.reshape(-1, 2)
        cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

    # Save result image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("ouput_image.jpg", img)


def process_results(img, score_text, score_link, text_threshold, link_threshold, low_text, ratio_w, ratio_h):
    
    boxes, polys, mapper = getDetBoxes_core(score_text, score_link, text_threshold, link_threshold, low_text)

    # Adjust Coordinates
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    # Convert original image to numpy array if it's not already
    img = np.array(img)

    # List to store cropped regions
    cropped_regions = []

    for box in boxes:
        # Convert box coordinates to integer
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        
        # Get bounding box coordinates
        x_min = max(0, np.min(poly[:, 0]))
        y_min = max(0, np.min(poly[:, 1]))
        x_max = min(img.shape[1], np.max(poly[:, 0]))
        y_max = min(img.shape[0], np.max(poly[:, 1]))
        
        # Crop the region
        cropped = img[y_min:y_max, x_min:x_max]
        cropped_regions.append(cropped)

    return cropped_regions   

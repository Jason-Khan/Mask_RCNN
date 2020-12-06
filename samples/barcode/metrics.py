import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import json
import glob
import math
from tqdm.notebook import tqdm
import pandas as pd
from sklearn import metrics

#obtain confidence scores
def get_confidence_score(im_mask, bb):
    cnt = 0
    pixel_acc = 0
    im_w = im_mask.shape[0]
    im_h = im_mask.shape[1]
    poly = Polygon(bb)
    minx, miny, maxx, maxy = list(np.int_(poly.bounds))
    for row in range(minx, min(minx + 400, maxx)):
        for col in range(miny, min(miny + 400, maxy)):
            p_temp = Point(row, col)
            if p_temp.within(poly):
                cnt += 1
                pixel_acc = pixel_acc + im_mask[col, row]
    if cnt == 0:
        return 0
    avg_pix_value = pixel_acc/cnt
    return avg_pix_value

# Finds all bounding boxes using contours in a mask
def findBoundingboxs(image):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxs = []
    scores = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        poly = cv2.boxPoints(rect)
        box = np.int0(poly)
        boundingBoxs.append(box)
        scores.append(get_confidence_score(image, box))
        
    return boundingBoxs, scores


# Calculate Intersection over Union of two bounding boxes
def iou(bb1, bb2):
    a = Polygon(bb1)
    b = Polygon(bb2)
    intersection = a.intersection(b).area
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return intersection / union

# Calculates the mean Intersection over Union of two masks
# Only considers the number of BBs in label masks
# algo may needs update
def mean_iou(pred_bbs, label_bbs):
    if len(pred_bbs) == 0:
        return 0
    iou_scores = np.zeros((len(label_bbs), len(pred_bbs)))
    for i in range(len(label_bbs)):
        for j in range(len(pred_bbs)):
            iou_scores[i, j] = iou(label_bbs[i], pred_bbs[j])

    scores_taken = np.amax(iou_scores, axis=1)
    assert(len(scores_taken) == len(label_bbs))
    return sum(scores_taken) / len(label_bbs)

def normalize_to_gray(im):
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im / im.max()

def evaluate(predictions, ground_truths, findBoundingBoxs=findBoundingboxs, threshold=0.5, iou_threshold=0.5):
    """
    Given predictions and ground truths, calculate AP, Precision and Recall.
    :param predictions: An array of prediction images.
    :param ground_truths: An array of ground truth images.
    :param findBoundingBoxes: A method that takes in an image, and return 
                              its bounding boxes and their corresponding confidence scores.
    :param threshold: Convert image to a binary image using this threshold.
    :param iou_threshold: IOU threshold for classifying a True Positive.
    :return AP, precision, recall, and calculation table of dataset.
    """
    total_num_GT = 0
    all_data = {"confidenceScore": [], "iou": []}
    for prediction, ground_truth in tqdm(zip(predictions, ground_truths), total=len(predictions)):
        if prediction is None or ground_truth is None:
            continue
        # normalize prediction and ground truth
        prediction = (normalize_to_gray(prediction) > threshold).astype(np.uint8)
        ground_truth = (normalize_to_gray(ground_truth) > threshold).astype(np.uint8)
        
        # find bounding boxes
        bbs_pred, confidences = findBoundingBoxs(prediction)
        bbs_truth, _ = findBoundingBoxs(ground_truth)
        
        # find truth/prediction bounding box correspondence
        for bb_truth in bbs_truth:
            max_iou = 0
            max_cs = 0
            for i in range(len(bbs_pred)):
                curr_iou = iou(bbs_pred[i], bb_truth)
                if curr_iou > max_iou:
                    max_iou = curr_iou
                    max_cs = confidences[i]
            all_data["iou"].append(max_iou)
            all_data["confidenceScore"].append(max_cs)
        
        # increment total ground truths count
        total_num_GT += len(bbs_truth)
    data = pd.DataFrame(all_data)
    data["TP"] = (data["iou"] >= iou_threshold) * 1
    data["FP"] = (data["iou"] < iou_threshold) * 1
    data = data.sort_values(by=["confidenceScore"], ascending=[False])
    data["Acc TP"] = np.cumsum(data["TP"])
    data["Acc FP"] = np.cumsum(data["FP"])
    data["Precision"] = data["Acc TP"] / (data["Acc TP"] + data["Acc FP"])
    data["Recall"] = data["Acc TP"] / total_num_GT
    plt.plot(list(data["Recall"]), list(data["Precision"]))
    interpolation = list(data["Precision"])
    for i in reversed(range(len(interpolation) - 1)):
        if interpolation[i+1]>interpolation[i]:
            interpolation[i]=interpolation[i+1]
    plt.plot(list(data["Recall"]), interpolation)
    AP = metrics.auc(list(data["Recall"]), interpolation)
    precision = max(list(data["Precision"]))
    recall = max(list(data["Recall"]))
    print("At IOU", iou_threshold * 100)
    print("AP:", AP)
    print("Precision:", precision)
    print("Recall:", recall)
    return AP, precision, recall, data
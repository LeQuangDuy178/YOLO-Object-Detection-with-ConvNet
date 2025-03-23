#------------------------------------------- Car Detection YOLO ---------------------------------
print("Car Detection!!!")

##-----------------------------------------
# Non-max Suppression test
import tensorflow as tf
import numpy as np

# Example 1: boxes with shape (2, 4) and scores with shape (2,)
# Box 1 is kept first since it has larger scores
# Box 2 then is compared with box 1 based on IOU
# If IOU is higher than the threshold, meaning that it may detect same obj with box 1
# NMS will discard this box, in case IOS is lower than threshold, box 2 can be kept

# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: iou

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """

    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    ### START CODE HERE
    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ##(≈ 7 lines)
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = xi2 - xi1
    inter_height = yi2 - yi1
    inter_area = max(inter_width, 0) * max(inter_height, 0) # Return 0 if edges, vertices gives 0 or negative detected
    
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ## (≈ 3 lines)
    box1_area = np.multiply(np.subtract(box1_x2, box1_x1), np.subtract(box1_y2, box1_y1))
    box2_area = np.multiply(np.subtract(box2_x2, box2_x1), np.subtract(box2_y2, box2_y1))
    union_area = np.subtract(np.add(box1_area, box2_area), inter_area)
    
    # compute the IoU
    iou = np.divide(inter_area, union_area)
    ### END CODE HERE
    
    return iou

# Inputs
boxes_1 = tf.constant([[0, 0, 1, 1], [0.1, 0.1, 1.1, 1.1]], dtype=tf.float32)
scores_1 = tf.constant([0.9, 0.8], dtype=tf.float32)

iou = iou(boxes_1[0], boxes_1[1])

selected_indices_1 = tf.image.non_max_suppression(
    boxes=boxes_1, scores=scores_1, max_output_size=2, iou_threshold=0.5
)

result_1 = tf.gather(boxes_1, selected_indices_1)

print("Example 1:")
print(f"IOU: {iou}")
print("Boxes:", boxes_1.numpy())
print("Scores:", scores_1.numpy())
print("Selected Indices:", selected_indices_1.numpy())
print("Selected Boxes:", result_1.numpy())

# Example 2: boxes with shape (1, 4) and scores with shape (1,)
boxes_2 = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)
scores_2 = tf.constant([0.9], dtype=tf.float32)

selected_indices_2 = tf.image.non_max_suppression(
    boxes=boxes_2, scores=scores_2, max_output_size=1, iou_threshold=0.5
)

result_2 = tf.gather(boxes_2, selected_indices_2)

print("\nExample 2:")
print("Boxes:", boxes_2.numpy())
print("Scores:", scores_2.numpy())
print("Selected Indices:", selected_indices_2.numpy())
print("Selected Boxes:", result_2.numpy())
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt,binary_erosion,generate_binary_structure
from scipy.ndimage import _ni_support


def metrics_val(output,target,num_classes):
    num = output.shape[0]
    output = torch.argmax(torch.softmax(output, dim=1), dim=1)
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    metric_list = 0.0
    for i in range(num):
        prediction = output[i]
        label = target[i]
        metric_i = []
        for j in range(1,num_classes):
            metric_i.append(calculate_dice(prediction == j, label == j))
        metric_list += np.array(metric_i)
    metric_list = metric_list/num

    return metric_list
    
def calculate_dice(pred,gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dc = metric.binary.dc(pred, gt)
    return dc
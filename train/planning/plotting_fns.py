import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image

def save_topk_plot(x, y, lines, x_label, y_label, filename, k=0):
    plt.figure()
    if k > 0:
        argsort = np.argsort(x)
        plt.scatter(x[argsort[:k]], y[argsort[:k]], color='b')
        plt.scatter(x[argsort[k:]], y[argsort[k:]], color='grey')
    else:
        plt.scatter(x, y)
        
    if isinstance(lines, dict):
        for line_name, line_y in lines.items():
            plt.axhline(line_y, label=line_name, linestyle='--')
        plt.legend()
    else:
        plt.axhline(lines, color='r', linestyle='--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.close()
        
def save_rollout_images(context_images, pred_images, waypoint_annotated_images, goal_image, save_paths): 
    """
    
    Args:
        context_images: B, peva_context_size, 3, H, W
        pred_images: B, W or W-1, policy_pred_horizon, 3, H, W
        waypoint_annotated_images: B, W, 3, H, W
        goal_image: B, 3, H, W
        save_paths: list of strings, each string is the path to save the image of length B
    """
    pred_len = pred_images.shape[2]
    
    os.makedirs(os.path.dirname(save_paths[0]), exist_ok=True)
    B, _, C, H, Wimg = context_images.shape
    device = context_images.device
    W, a, b = waypoint_annotated_images.shape[1], context_images.shape[1], pred_images.shape[1]*pred_len
    max_len = max(a, b)
    
    # replace context and pred images with the appropriate waypoint annotated images
    context_images = context_images.clone()
    pred_images = pred_images.clone()
    context_images[:, -1] = waypoint_annotated_images[:, 0].clone()
    for i in range(W-1):
        pred_images[:, i, -1] = waypoint_annotated_images[:, i+1]
    
    image_list = [
        context_images, torch.zeros(B, max_len-a, C, H, Wimg, device=device), # B, max_len, 3, H, W
        pred_images.flatten(1,2), torch.zeros(B, max_len-b, C, H, Wimg, device=device), # B, max_len, 3, H, W
        goal_image[:, None] # B, 1, 3, H, W
    ]
    image = torch.cat(image_list, dim=1) # B, 2*max_len+1, C,  H, W
    for b in range(image.shape[0]):
        save_image(image[b], save_paths[b], nrow=max_len)
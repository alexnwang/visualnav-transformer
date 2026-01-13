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
        
def save_rollout_images(context_images, pred_images, waypoint_annotated_images, goal_image, save_paths, pred_len=None): 
    """
    
    Args:
        context_images: B, peva_context_size, 3, H, W
        pred_images: B, W*policy_pred_horizon, 3, H, W
        waypoint_annotated_images: B, W, 3, H, W
        goal_image: B, 3, H, W
        save_paths: list of strings, each string is the path to save the image of length B
    """    
    os.makedirs(os.path.dirname(save_paths[0]), exist_ok=True)
    B, _, C, H, Wimg = context_images.shape
    device = context_images.device
    a, b = context_images.shape[1], pred_images.shape[1]
    max_len = max(a, b)
    
    # replace context and pred images with the appropriate waypoint annotated images
    if waypoint_annotated_images is not None:
        W = waypoint_annotated_images.shape[1]
        context_images = context_images.clone()
        pred_images = pred_images.clone().unflatten(1, (-1, pred_len))
        context_images[:, -1] = waypoint_annotated_images[:, 0].clone()
        for i in range(W-1):
            pred_images[:, i, -1] = waypoint_annotated_images[:, i+1]
        pred_images = pred_images.flatten(1,2)
    
    image_list = [
        context_images, torch.zeros(B, max_len-a, C, H, Wimg, device=device), # B, max_len, 3, H, W
        pred_images, torch.zeros(B, max_len-b, C, H, Wimg, device=device), # B, max_len, 3, H, W
        goal_image[:, None] # B, 1, 3, H, W
    ]
    image = torch.cat(image_list, dim=1) # B, 2*max_len+1, C,  H, W
    for b in range(image.shape[0]):
        save_image(image[b], save_paths[b], nrow=max_len)
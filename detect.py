from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#去除重叠较为严重的框
def cal_iou(bboxes):
    delete = []
    for i  in range(len(bboxes)-1):
        for j in range(i+1, len(bboxes)):
            x1, y1, x2, y2, conf, cls_conf, cls_pred = bboxes[i]
            x1_, y1_, x2_, y2_, conf_, cls_conf_, cls_pred_ = bboxes[j]
            xmin = max(x1, x1_)
            xmax = min(x2, x2_)
            ymin = max(y1, y1_)
            ymax = min(y2, y2_)
            area_ = abs(x1-x2)*abs(y1-y2)
            area__ = abs(x1_-x2_)*abs(y1_-y2_)
            if (xmax-xmin)<=0 or (ymax-ymin)<=0:
                continue
            areajoin = abs(xmin-xmax)*abs(ymin-ymax)
            iou_ = areajoin/min(area_, area__)
            if iou_>=0.7:
                if area_<area__:
                    delete.append(bboxes[i])
                else:
                    delete.append(bboxes[j])
            elif (iou_<0.7) and (iou_>0.3):
                if (area_<area__) and (cls_conf.item() < 0.5):
                    delete.append(bboxes[i])
                elif (area_>area__) and (cls_conf_.item() < 0.5):
                    delete.append(bboxes[j])
    for d in delete:
        try:
            bboxes.remove(d)
        except:
            pass
    return bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/mchar_test_a", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_103.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)
    df = pd.DataFrame(columns=['file_name', 'file_code'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path,map_location = torch.device('cpu')))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            bboxes = []
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                if cls_conf<opt.conf_thres:
                    continue
                bboxes.append([x1, y1, x2, y2, conf, cls_conf, cls_pred])

            #去掉置信度低、重叠严重的框
            bboxes = cal_iou(bboxes)
            label = {}
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in bboxes:
            # for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)]+' '+str(round(cls_conf.item(), 2)),
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )
                tmp = {(int(x1) + int(x2))/2:str(classes[int(cls_pred)])}
                label.update(tmp)
            label = sorted(label.items(), key = lambda x: x[0])
            labels = ''
            for lae in label:
                labels+=lae[1]
        else:
            labels = ''
        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split(os.sep)[-1].split(".")[0]
        plt.savefig(f"output{os.sep}{filename}.png", bbox_inches = "tight", pad_inches = 0.0)
        plt.close()
        df.loc[img_i] = [f'{filename}.png', labels]
        df.to_csv(f"output{os.sep}mchar_sample_submit_A.csv", index = False)

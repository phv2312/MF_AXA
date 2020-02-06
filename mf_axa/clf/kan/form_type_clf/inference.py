import sys, os
sys.path.append(os.path.dirname(__file__))

from train_xentropy import data_transforms
from torchvision.models import resnet34
import torch.nn as nn
import torch
import cv2
import PIL.Image as Image

class FormTypeClassifer:
    def __init__(self, weight_path = ""):
        model_ft = resnet34(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 2)
        model_ft.eval()
        model_ft.load_state_dict(torch.load(weight_path, map_location='cpu'))

        self.model_ft = model_ft
        self.idx2cls = {0:'multi', 1:'single'}

    def process(self, np_image):
        pil_image = Image.fromarray(np_image)
        input_tensor = data_transforms['val'](pil_image) # (-> c,h,w)

        input_tensor = input_tensor.unsqueeze(0) # -> (1,c,h,w)
        outputs = self.model_ft(input_tensor)
        _, preds = torch.max(outputs, 1) # _, (1,)

        preds = preds.cpu().numpy().tolist()
        cls_preds = [self.idx2cls[pred] for pred in preds]

        return cls_preds


if __name__ == '__main__':
    clf = FormTypeClassifer(weight_path='/home/kan/Desktop/Cinnamon/mf/multiple_form/x/ahihi.ckpt')
    import glob, os

    folder_dir = "/home/kan/Desktop/split_data (1)/ahihi/val/single"
    for im_fn in glob.glob(os.path.join(folder_dir, "*.tif")):

        print (im_fn)
        sample_im = cv2.imread(im_fn)

        clf.process(sample_im)

    pass
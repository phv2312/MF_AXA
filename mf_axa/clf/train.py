import numpy as np
import os
from model import resnet34
import torch
from dataloader import FormLoader, imshow
from torchvision import transforms
from datetime import datetime
import torch.nn.functional as F


def end2endLoss(output1, output2, labels1, labels2):
    # Softmax fucntion
    prob1 = F.softmax(output1, 1)
    prob_multi = prob1[:, 1].unsqueeze(1)
    prob2 = F.softmax(output2, 1)
    prob2_ = torch.mul(prob2, prob_multi)

    # Entropy loss
    entropy_function = torch.nn.NLLLoss()

    # Prediction for single or multiple
    pred1 = - entropy_function(prob1, labels1.type(torch.cuda.LongTensor))

    # Prediction for vertical or horizontal
    multi_idx = torch.where(labels1 != 0)
    prob2 = prob2[multi_idx]
    labels2 = labels2[multi_idx]
    pred2 = - entropy_function(prob2, labels2.type(torch.cuda.LongTensor))

    loss = pred1 + pred2
    return loss

class FormTrainer():
    def __init__(self, model, data_train, data_valid, model_dir):
        """
        :param model: Model for feature extractor
        :param data_train: Path to data train
        :param data_valid: Path to data for validation
        :param log_dir: Path to save logs for tensorboard
        :param model_dir: Path to save model.pth
        """
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model = model.to(self.device)
        self.backbone = 'resnet34'
        self.data_train = data_train
        self.data_valid = data_valid
        # Version for model
        self.version = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # Path to save models
        self.model_dir = model_dir
        # Early stopping
        self.validating_loss_tmp = 0

    def train_one_epoch(self, dataloader, optimizer):
        """
        :param dataloader: Training data loader
        :param optimizer: Optimizer for training, you can actively config the hyper parameters after one epoch
        :return: Loss of training
        """
        debug = True
        self.model.train()
        total_loss = 0

        for batch_idx, (imgs, labels1, labels2) in enumerate(dataloader):
            imgs = imgs.to(self.device)
            labels1 = labels1.to(self.device)
            labels2 = labels2.to(self.device)

            # Debug images
            if debug:
                for i in range(2):
                    img = imgs[i].cpu()
                    transforms_pil = transforms.ToPILImage()
                    img = np.array(transforms_pil(img), dtype=np.uint8)
                    imshow(img)

            # Feed forward, output 1 for single and multi
            # output2 for vertical and horizontal
            output1, output2 = self.model(imgs)

            # Clear gradient
            optimizer.zero_grad()

            # Loss
            loss = end2endLoss(output1, output2, labels1, labels2)
            print(loss)
            total_loss += loss

            if (batch_idx + 1) % 100 == 0:
                print('Training triplet loss is ', total_loss / 100)
                self.writer.add_scalar('Local_loss_training', local_loss / 100, batch_idx)
                local_loss = 0

            loss.backward()
            optimizer.step()
        return total_loss / (batch_idx + 1)

    def validate(self, dataloader):
        """
        :param dataloader: Data loader for validating dataset
        :return: Loss of validation
        """
        debug = False
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch_idx, (imgs, labels) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # Debug images
                if debug:
                    for i in range(2):
                        img = imgs[i].cpu()
                        transforms_pil = transforms.ToPILImage()
                        img = np.array(transforms_pil(img), dtype=np.uint8)
                        imshow(img)

                # Feed forward
                output = self.model(img)
                # Loss
                loss = end2endLoss()
                total_loss += loss

        return total_loss / (batch_idx + 1)

    def save_model(self, model_dir):
        infor = {'model_state_dict': self.model.state_dict(),
                 'arch': self.backbone,
                 'validating_loss': self.validating_loss_tmp}
        torch.save(infor, model_dir)


def train():
    # Hyper parameters
    epochs = 30
    lr = 1e-3
    batch_size = 10

    # Path of data and models
    data_root = '/home/dotieuthien/Documents/AXA/MF_AXA/mf_axa/clf/dataset/train'
    data_valid = '/home/dotieuthien/Documents/AXA/MF_AXA/mf_axa/clf/dataset/valid'
    model_dir = '/home/dotieuthien/Documents/AXA/MF_AXA/mf_axa/clf/saved_models'

    # Data loaders
    train_loader = FormLoader(data_root, batch_size, True).loader()
    valid_loader = FormLoader(data_root, 1, True).loader()

    # Trainer
    model = resnet34()
    trainer = FormTrainer(model, data_root, data_valid, model_dir)

    # Adam optimizer
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(epochs):
        # Train one epoch
        training_loss = trainer.train_one_epoch(train_loader, optimizer)

        # Validate
        validating_loss = trainer.validate(valid_loader)

        print('Training loss and Validating loss in epoch %d: %f and %f' % (epoch, training_loss, validating_loss))
        trainer.save_model(os.path.join(model_dir, trainer.version + '.pth'))


if __name__ == '__main__':
    train()

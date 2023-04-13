import random
import os
import time
import torch
from torch import nn
from torchvision.transforms import *
from torchvision.datasets import *
from torch.utils.data import DataLoader
from logger import Logger, TrainingEpochMeters, EvalEpochMeters

from model import CNV

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Trainer(object):
    def __init__(self, model, epochs):
        
        self.model = model
        self.epochs = epochs
        if not os.path.exists('output'):
            os.makedirs('output')
        self.output_dir_path = os.path.join(os.getcwd(), 'output')
        self.logger = Logger(self.output_dir_path, True)
        # if checkpoints does not exist in output, create it
        if not os.path.exists(os.path.join(self.output_dir_path, 'checkpoints')):
            os.makedirs(os.path.join(self.output_dir_path, 'checkpoints'))
        self.checkpoints_dir_path = os.path.join(self.output_dir_path, 'checkpoints')

        # Randomness
        random_seed = 404
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

        # Datasets
        transform_to_tensor = transforms.Compose([transforms.ToTensor()])

        self.num_classes = 10
        train_transforms_list = [transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()]
        transform_train = transforms.Compose(train_transforms_list)
        builder = CIFAR10


        train_set = builder(root="./data/",
                            train=True,
                            download=True,
                            transform=transform_train)
        test_set = builder(root="./data/",
                           train=False,
                           download=True,
                           transform=transform_to_tensor)
        self.train_loader = DataLoader(train_set,
                                       batch_size=100,
                                       shuffle=True,
                                       num_workers=4)
        self.test_loader = DataLoader(test_set,
                                      batch_size=100,
                                      shuffle=False,
                                      num_workers=4)

        # Init starting values
        self.starting_epoch = 1
        self.best_val_acc = 0

        # Setup device
        if torch.cuda.is_available() is not None:
            self.device = torch.device('cuda:0')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = 'cpu'
        self.device = torch.device(self.device)

        model = model.to(device=self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(device=self.device)

        # Init optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                        lr=0.02,
                                        weight_decay=0)

        # LR scheduler
        self.scheduler = None
    
    def checkpoint_best(self, epoch, name):
        best_path = os.path.join(self.checkpoints_dir_path, name)
        self.logger.info("Saving checkpoint model to {}".format(best_path))
        torch.save({
            'state_dict': self.model.state_dict(),
            'optim_dict': self.optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_acc': self.best_val_acc,
        }, best_path)

    def train_model(self):

        # training starts

        for epoch in range(self.epochs):

            # Set to training mode
            self.model.train()
            self.criterion.train()

            # Init metrics
            epoch_meters = TrainingEpochMeters()
            start_data_loading = time.time()

            for i, data in enumerate(self.train_loader):
                (input, target) = data
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                target_var = target

                # measure data loading time
                epoch_meters.data_time.update(time.time() - start_data_loading)

                # Training batch starts
                start_batch = time.time()
                output = self.model(input)
                loss = self.criterion(output, target_var)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.model.clip_weights(-1, 1)

                # measure elapsed time
                epoch_meters.batch_time.update(time.time() - start_batch)

                if i % 10 == 0 or i == len(self.train_loader) - 1:
                    prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
                    epoch_meters.losses.update(loss.item(), input.size(0))
                    epoch_meters.top1.update(prec1.item(), input.size(0))
                    epoch_meters.top5.update(prec5.item(), input.size(0))
                    self.logger.training_batch_cli_log(epoch_meters, epoch, i,
                                                       len(self.train_loader))

                # training batch ends
                start_data_loading = time.time()

            # Set the learning rate
            if self.scheduler is not None:
                self.scheduler.step(epoch)
            else:
                # Set the learning rate
                if epoch % 40 == 0:
                    self.optimizer.param_groups[0]['lr'] *= 0.5

            # Perform eval
            with torch.no_grad():
                top1avg = self.eval_model(epoch)

            # checkpoint
            if top1avg >= self.best_val_acc:
                self.best_val_acc = top1avg
                self.checkpoint_best(epoch, "best.tar")
            else:
                self.checkpoint_best(epoch, "checkpoint.tar")

        # training ends
        return self.model
    
    def eval_model(self, epoch=None):
        eval_meters = EvalEpochMeters()

        # switch to evaluate mode
        self.model.eval()
        self.criterion.eval()

        for i, data in enumerate(self.test_loader):

            end = time.time()
            (input, target) = data

            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            target_var = target

            # compute output
            output = self.model(input)

            # measure model elapsed time
            eval_meters.model_time.update(time.time() - end)
            end = time.time()

            # compute loss
            loss = self.criterion(output, target_var)
            eval_meters.loss_time.update(time.time() - end)

            pred = output.data.argmax(1, keepdim=True)
            correct = pred.eq(target.data.view_as(pred)).sum()
            prec1 = 100. * correct.float() / input.size(0)

            _, prec5 = accuracy(output, target, topk=(1, 5))
            eval_meters.losses.update(loss.item(), input.size(0))
            eval_meters.top1.update(prec1.item(), input.size(0))
            eval_meters.top5.update(prec5.item(), input.size(0))

            # Eval batch ends
            self.logger.eval_batch_cli_log(eval_meters, i, len(self.test_loader))

        return eval_meters.top1.avg
    
'''

model = CNV(num_classes=10, weight_bit_width=2, act_bit_width=1, in_bit_width=8, in_ch=3)
epochs = 100
trainer = Trainer(model, epochs)
model = trainer.train_model()

'''
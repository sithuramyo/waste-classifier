import torch
import os
import matplotlib.pyplot as plt
import shutil

def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def plot_metrics(train_loss, val_loss, train_acc, val_acc):
    # Convert CUDA tensors to CPU tensors before converting to NumPy arrays
    train_acc = [acc.cpu().numpy() for acc in train_acc]
    val_acc = [acc.cpu().numpy() for acc in val_acc]
    train_loss = [loss.cpu().numpy() for loss in train_loss]
    val_loss = [loss.cpu().numpy() for loss in val_loss]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

class Logger:
    def __init__(self, filename):
        self.filename = filename
        
    def log(self, message):
        with open(self.filename, 'a') as f:
            f.write(message + '\n')
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
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join('reports', 'training_metrics.png'))
    plt.close()

class Logger:
    def __init__(self, filename):
        self.filename = filename
        
    def log(self, message):
        with open(self.filename, 'a') as f:
            f.write(message + '\n')
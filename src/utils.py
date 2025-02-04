import torch
import os
import matplotlib.pyplot as plt
import shutil
import torch
import shutil

def save_checkpoint(model, optimizer, epoch, best_acc, is_best=False, folder='models', filename='checkpoint.pth.tar'):
    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)
    
    checkpoint_path = os.path.join(folder, filename)
    
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_acc': best_acc
    }
    
    try:
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            shutil.copyfile(checkpoint_path, os.path.join(folder, 'model_best.pth.tar'))
        print(f"Checkpoint saved: {checkpoint_path} (Epoch {epoch}, Best Acc: {best_acc:.4f})")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")



def load_checkpoint(model, optimizer, folder='models', filename='checkpoint.pth.tar', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint_path = os.path.join(folder, filename)
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return model, optimizer, 0, 0.0  
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

        model.to(device)  # Ensure model is moved to the correct device
        print(f"Checkpoint loaded: {checkpoint_path} (Epoch {epoch}, Best Acc: {best_acc:.4f}) on {device}")
        return model, optimizer, epoch, best_acc
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return model, optimizer, 0, 0.0


def plot_metrics(train_loss, val_loss, train_acc, val_acc):
    # Convert CUDA tensors to CPU tensors before converting to NumPy arrays
    train_acc = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in train_acc]
    val_acc = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in val_acc]
    train_loss = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in train_loss]
    val_loss = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in val_loss]

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
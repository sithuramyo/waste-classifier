import argparse
import os
from train import train_model
from evaluate import evaluate_model
from verify_json import verify_annotation_compatibility

def main():
    parser = argparse.ArgumentParser(description='Waste Classification Training')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'verify'], required=True)
    parser.add_argument('--data_dir', default='dataset', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_path',default='dataset/json',help='Path to model checkpoint for evaluation')
    
    args = parser.parse_args()

    if args.mode == 'verify' or args.verify:
        verify_annotation_compatibility(
            annotation_path = os.path.join(args.data_dir, 'json/train_annotation.json'),
            data_root = args.data_dir
        )
        if args.mode == 'verify':
            exit()

    if args.mode == 'train':
        train_model(
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
    elif args.mode == 'evaluate':
        if not args.model_path:
            raise ValueError("Model path is required for evaluation mode")
        evaluate_model(
            model_path=args.model_path,
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )

if __name__ == '__main__':
    main()
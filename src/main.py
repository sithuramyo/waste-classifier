import argparse
from train import train_model
from evaluate import evaluate_model
from check_dataset import main as verify_dataset

def main():
    parser = argparse.ArgumentParser(description='Waste Classification Training')
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    parser.add_argument('--data_dir', default='dataset', help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_path', help='Path to model checkpoint for evaluation')
    
    args = parser.parse_args()

    if args.mode == 'verify' or args.verify:
        verify_dataset()
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
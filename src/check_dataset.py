# training/src/check_dataset.py
from verify_json import verify_annotation_compatibility

def main():
    splits = ['train', 'val', 'test']
    all_ok = True
    
    for split in splits:
        print(f"\nVerifying {split} annotations:")
        results = verify_annotation_compatibility(
            f"dataset/json/{split}_annotation.json",
            f"dataset/{split}"
        )
        
        print(f"Categories: {results['unique_categories']}")
        print(f"Images: {results['total_images']}")
        print(f"Annotations: {results['total_annotations']}")
        print(f"Issues found: {results['issues_found']}")
        
        if results['issues_found'] > 0:
            all_ok = False
            print("Sample issues:")
            for issue in results['sample_issues']:
                print(f" - {issue}")

    if not all_ok:
        print("\n❌ Dataset validation failed! Fix issues before training.")
        exit(1)
    else:
        print("\n✅ All dataset splits validated successfully!")

if __name__ == '__main__':
    main()
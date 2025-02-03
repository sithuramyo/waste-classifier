import json
import os

def verify_annotation_compatibility(annotation_path, data_root):
    with open(annotation_path) as f:
        data = json.load(f)
    
    issues = []
    
    # Verify category consistency
    unique_categories = set()
    for cat in data['categories']:
        cat_dir = os.path.join(data_root, cat['name'])
        if not os.path.isdir(cat_dir):
            issues.append(f"Category directory missing: {cat_dir}")
        unique_categories.add(cat['name'])
    
    # Verify image files
    for img in data['images']:
        found = False
        for cat in data['categories']:
            potential_path = os.path.join(data_root, cat['name'], img['file_name'])
            if os.path.exists(potential_path):
                found = True
                break
        if not found:
            issues.append(f"Image file not found: {img['file_name']}")
    
    # Verify annotation-image relationships
    image_ids = {img['image_id'] for img in data['images']}
    for ann in data['annotations']:
        if ann['image_id'] not in image_ids:
            issues.append(f"Annotation {ann['id']} references missing image ID {ann['image_id']}")
    
    return {
        'total_images': len(data['images']),
        'total_annotations': len(data['annotations']),
        'unique_categories': len(unique_categories),
        'issues_found': len(issues),
        'sample_issues': issues[:5]
    }
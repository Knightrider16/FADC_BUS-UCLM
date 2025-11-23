import json, random
from sklearn.model_selection import train_test_split

json_path = r"E:\S3\OT\FADC\FADC\data\BUS_COCO\annotations.json"

with open(json_path, 'r') as f:
    coco = json.load(f)

images = coco["images"]
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

def subset(coco, selected_images):
    selected_ids = {img["id"] for img in selected_images}
    return {
        "images": selected_images,
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in selected_ids],
        "categories": coco["categories"]
    }

train = subset(coco, train_images)
val = subset(coco, val_images)

with open(json_path.replace("annotations.json", "annotations_train.json"), "w") as f:
    json.dump(train, f, indent=2)

with open(json_path.replace("annotations.json", "annotations_val.json"), "w") as f:
    json.dump(val, f, indent=2)

print("✅ Train / Val split created")

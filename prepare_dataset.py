import os, cv2, json, numpy as np
from tqdm import tqdm

SRC_IMG = r"E:\S3\OT\FADC\FADC\data\BUS_RCNN2\images"
SRC_MASK = r"E:\S3\OT\FADC\FADC\data\BUS_RCNN2\masks"

OUT = r"E:\S3\OT\FADC\FADC\data\BUS_COCO"
os.makedirs(OUT, exist_ok=True)

os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)

coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "benign"},
        {"id": 2, "name": "malignant"}
    ]
}

ann_id = 1
img_id = 1

for fname in tqdm(sorted(os.listdir(SRC_IMG))):
    if not fname.endswith(".png"):
        continue

    img_path = os.path.join(SRC_IMG, fname)
    mask_path = os.path.join(SRC_MASK, fname)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    h, w = mask.shape
    coco["images"].append({
        "id": img_id,
        "file_name": fname,
        "height": h,
        "width": w
    })

    # Process benign (1) and malignant (2)
    for cls_id in [1, 2]:
        binary = (mask == cls_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 3:  # ignore tiny noise
                continue

            segmentation = cnt.flatten().tolist()
            x, y, w_box, h_box = cv2.boundingRect(cnt)

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "segmentation": [segmentation],
                "bbox": [x, y, w_box, h_box],
                "area": float(cv2.contourArea(cnt)),
                "iscrowd": 0
            })
            ann_id += 1

    cv2.imwrite(os.path.join(OUT, "images", fname), img)
    cv2.imwrite(os.path.join(OUT, "masks", fname), mask)

    img_id += 1

with open(os.path.join(OUT, "annotations.json"), "w") as f:
    json.dump(coco, f, indent=2)

print("✅ COCO dataset created at:", OUT)
print("🚀 Conversion Completed MUCH Faster!")

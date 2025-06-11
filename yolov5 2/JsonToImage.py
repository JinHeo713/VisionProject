import cv2
import json

image_path = "dataset/input_image/25cm_30.jpg"
json_path = "testJson.json"

img = cv2.imread(image_path)

with open(json_path, "r") as f:
    predictions = json.load(f)

for obj in predictions:
    class_label = obj["class_label"]
    confidence = obj["confidence"]
    x = int(obj["x"])
    y = int(obj["y"])
    w = int(obj["width"])
    h = int(obj["height"])

    x1, y1 = x, y
    x2, y2 = x + w, y + h

    if class_label == "connected":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    label = f"{class_label} {confidence:.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)

cv2.imshow("YOLO Visualization", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

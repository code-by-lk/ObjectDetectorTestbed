import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from class_names import ClassCategoryNames

CLASS_NAMES = ClassCategoryNames.get_category_names('coco')

# set random colors per class for visualization
COLORS = np.rando.uniform(0, 255, size=(len(CLASS_NAMES), 3)) # RGB colors

# define torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension

    # make prediction
    with torch.no_grad():
        outputs = model(image)

    # collect all bboxes with scores above threshold
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    threshold_indices = [scores.index(i) for i in scores if i > detection_threshold]
    bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    boxes = bboxes[np.array(scores) >= detection_threshold].astype(np.int32)

    # get all predicted class names
    labels = outputs[0]['labels'].cpu().numpy()
    pred_classes = [CLASS_NAMES[labels[i]] for i in threshold_indices]

    return boxes, pred_classes

def draw_boxes(boxes, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[CLASS_NAMES.index(classes[i])]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            2
        )
        cv2.putText(
            image,
            classes[i],
            (int(box[0])),
            int(box[i] - 5), # why minus 5?
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            lineType=cv2.LINE_AA
        )
    
    return image
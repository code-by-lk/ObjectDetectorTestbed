import argparse
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from .utils import detect_utils

def __main__():
    """
    Executing detect_images

        python detect_images.py --input input/image2.jpg

        python detect_images.py --input input/image2.jpg --min-size 1200 --threshold 0.5

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to input image/video')
    parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                        help='minimum input size for the RetinaNet network')
    parser.add_argument('-t', '--threshold', default=0.6, type=float,
                        help='minimum confidence score for detection')
    args = vars(parser.parse_args())
    print('USING:')
    print(f"Minimum image size: {args['min_size']}")
    print(f"Confidence threshold: {args['threshold']}")


    # download RetinaNet model
    model = torchvision.models.detection.retinanet_resnet50_fpn(
        pretrained=True,
        min_size=args['min_size']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    # import image
    image = Image.open(args['input']).convert('RGB')
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # run inference detection
    boxes, classes = detect_utils.predict(image, model, device, args['threshold'])
    result = detect_utils.draw_boxes(boxes, classes, image_array)

    cv2.imshow('Image', result)
    cv2.waitKey(0)
    save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}_t{int(args['threshold']*100)}"
    cv2.imwrite(f"outputs/{save_name}.jpg", result)


if __name__ == '__main__':
    __main__()

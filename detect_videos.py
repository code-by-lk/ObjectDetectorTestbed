import cv2
import argparse
import time
import torch
import torchvision
from PIL import Image

from .utils import detect_utils

def __main__():
    """
    Executing detect_videos

        python detect_videos.py --input input/video2.mp4

        python detect_videos.py --input input/video2.mp4 --min-size 800


    """

    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to input video')
    parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                        help='minimum input size for the RetinaNet network')
    parser.add_argument('-t', '--threshold', default=0.6, type=float,
                        help='minimum confidence score for detection')
    args = vars(parser.parse_args())
    print('USING:')
    print(f"Minimum frame size: {args['min_size']}")
    print(f"Confidence threshold: {args['threshold']}")
    # download or load the model from disk
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, 
                                                                num_classes=91, 
                                                                min_size=args['min_size'])
    
    # load the model onto the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)

    cap = cv2.VideoCapture(args['input'])
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')
    
    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}_t{int(args['threshold']*100)}"
    
    # define codec and create VideoWriter object 
    out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), 20, 
                        (frame_width, frame_height))
    frame_count = 0 # to count total frames
    total_fps = 0 # to get the final frames per second

    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()

        if ret == True:
            # convert the frame into PIL Image format
            pil_image = Image.fromarray(frame).convert('RGB')
            # get the start time
            start_time = time.time()
            
            # get predictions for the current frame
            boxes, classes = detect_utils.predict(pil_image, model, device, args['threshold'])
            # draw boxes and show current frame on screen
            result = detect_utils.draw_boxes(boxes, classes, frame)
            # get the end time
            end_time = time.time()
            # get the fps
            fps = 1 / (end_time - start_time)
            # add fps to total fps
            total_fps += fps
            # increment frame count
            frame_count += 1
            # press `q` to exit
            wait_time = max(1, int(fps/4))
            cv2.imshow('image', result)
            out.write(result)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

        else:
            break
    
        # release VideoCapture()
        cap.release()
        # close all frames and video windows
        cv2.destroyAllWindows()
        # calculate and print the average FPS
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")


if __name__ == '__main__':
    __main__()
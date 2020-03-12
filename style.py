import torch
import cv2
import argparse
import os
import os.path as osp
from torchvision.transforms import transforms
import glob

# Cloned modules
import utils
import transformer


#   To use this script just simply enter
#   python style.py --input video.mp4 --style hokusai-wave.pth
#   E.g python style.py -i data\Cannonbal.mp4 -s weights\hokusai-wave.pth
#   GPU Memory alert! if you have bigger memory, increase batch size , smaller memory, decrease batch size by using -b, default is batch size 8
#   Remove old folders because this script still cannot overwrite the exisiting folder

def stylize_folder(style_path, folder_containing_the_content_folder, save_folder, batch_size=1):
    # This function is cloned and modified from the original repo: https://github.com/rrmina/fast-neural-style-pytorch
    """Stylizes images in a folder by batch
    If the images  are of different dimensions, use transform.resize() or use a batch size of 1
    IMPORTANT: Put content_folder inside another folder folder_containing_the_content_folder
    """
    # Choose GPU if available else use CPU
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Image loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    # Please take note that the original frames must be inside a subfolder inside of the folder that you provide the path to --input
    # This is due to the nature of how pytorch's ImageFolder work
    image_dataset = utils.ImageFolderWithPaths(folder_containing_the_content_folder, transform=transform)
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size)

    # Load Transformer Network
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(style_path))
    net = net.to(device)

    # Stylize batches of images
    with torch.no_grad():
        for content_batch, _, path in image_loader:
            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()

            # Generate image
            generated_tensor = net(content_batch.to(device)).detach()

            # Create folder if not exist
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            # Save images
            for i in range(len(path)):
                generated_image = utils.ttoi(generated_tensor[i])
                image_name = os.path.basename(path[i])
                utils.saveimg(generated_image, osp.join(save_folder, image_name))


def getFrames(video_path):
    # This function is cloned and modified from the original repo: https://github.com/rrmina/fast-neural-style-pytorch
    if not os.path.exists(ORIGINAL_FRAMES_FOLDER_NAME):
        os.makedirs(osp.join(ORIGINAL_FRAMES_FOLDER_NAME))
    if not os.path.exists(osp.join(ORIGINAL_FRAMES_FOLDER_NAME, "content")):
        os.makedirs(osp.join(ORIGINAL_FRAMES_FOLDER_NAME, "content"))

    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()
    count = 1
    success = True
    # "content" is to make the subfolder that will be needed later
    while success:
        cv2.imwrite(osp.join(ORIGINAL_FRAMES_FOLDER_NAME, "content", "frame" + str(count) + ".jpg"), image)
        success, image = cap.read()
        count += 1
    print("Done extracting all frames")


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="None")
    parser.add_argument("-s", "--style", type=str, default="None")
    parser.add_argument("-o", "--output", type=str, default="output.mp4")
    parser.add_argument("-b", "--batchsize", type=int, default=8)

    args = parser.parse_args()

    # need to chg the input from folder into video
    print("Your input video: {}".format(args.input))
    print("Your style image: {}".format(args.style))

    ORIGINAL_FRAMES_FOLDER_NAME = "original_frames"
    STYLED_FRAMES_FOLDER_NAME = "styled_frames"

    # Information of your input video
    cap = cv2.VideoCapture(args.input)
    VIDEO_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    VIDEO_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    FPS = int(cap.get(cv2.CAP_PROP_FPS))

    getFrames(args.input)
    # Prepare Part 2 by inference the original frames into the network to generate styled frames
    stylize_folder(args.style, ORIGINAL_FRAMES_FOLDER_NAME, STYLED_FRAMES_FOLDER_NAME, batch_size=args.batchsize)

    # This is hack-ish because I have not find a way to use glob to order the images in 1,2,3... instead of 1,10,11...
    numberOfImgs_Part2 = len(glob.glob(osp.join(STYLED_FRAMES_FOLDER_NAME, "frame*.jpg")))
    part2Frames = []
    for i in range(1, numberOfImgs_Part2 + 1):
        filename = osp.join(STYLED_FRAMES_FOLDER_NAME, "frame{}.jpg").format(i)
        part2Frames.append(filename)

    # Uncomment this to only produce Part 2 without the Part 1 and Part 3
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter(args.output, fourcc, FPS, (int(VIDEO_WIDTH), int(VIDEO_HEIGHT)))
    #
    # # Write frames from Part2 the video
    # for image_name in part2Frames:
    #     out.write(cv2.imread(image_name))
    # out.release()
    # print("Done creating the Part 2 of your styled video for your music!")

    # Create Part1  that has a image blending effect transition
    # Get the first frame from the "original_frames" folder and the the first frame from the "styled_frames" folder
    first_ori = cv2.imread("original_frames/content/frame1.jpg")
    first_styled = cv2.imread("styled_frames/frame1.jpg")
    part1Frames = []
    alpha = 0.0
    for i in range(0, 100):
        alpha = i / 100
        beta = (1.0 - alpha)
        output = cv2.addWeighted(first_styled, alpha, first_ori, beta, 0.0)
        part1Frames.append(output)
        # cv2.imwrite("part1_{}.jpg".format(i),output)

    # Create Part3 that has an image blending effect transition but in an inverse order
    # Get the last frame from the "original_frames" folder and the the last frame from the "styled_frames" folder
    lastIndex = len(glob.glob(osp.join(ORIGINAL_FRAMES_FOLDER_NAME, "content/frame*.jpg")))
    last_ori = cv2.imread("original_frames/content/frame" + str(lastIndex) + ".jpg")
    last_styled = cv2.imread("styled_frames/frame" + str(lastIndex) + ".jpg")
    part3Frames = []
    alpha = 0.0
    for i in range(0, 100):
        alpha = i / 100
        beta = (1.0 - alpha)
        output = cv2.addWeighted(last_ori, alpha, last_styled, beta, 0.0)
        part3Frames.append(output)
        # cv2.imwrite("part3_{}.jpg".format(i), output)

    # It is time to write our frames into a video!
    #  Part1 and Part3 frames are stored in memory, there are of course better ways to do this, but oh well ...
    # Define the codec and create VideoWrite object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(args.output, fourcc, FPS, (int(VIDEO_WIDTH), int(VIDEO_HEIGHT)))

    for i in (range(len(part1Frames))):
        out.write(part1Frames[i])
    for image_name in part2Frames:
        out.write(cv2.imread(image_name))
    for i in (range(len(part3Frames))):
        out.write(part3Frames[i])
    out.release()
    print("Done creating your styled video for your music!")

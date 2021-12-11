import os

import cv2

if __name__ == '__main__':

    input_picture_path = "video_merge"
    out_video_path = "./demo_video.avi"

    fps = 30
    # 图片尺寸
    img_size = (1920, 1080)

    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G')#opencv2.4
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # opencv3.0
    video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, img_size)

    for file in os.listdir(input_picture_path):
        image = cv2.imread(os.path.join(input_picture_path, file))
        video_writer.write(image)

    video_writer.release()

    # 读视频
    video_cap = cv2.VideoCapture(out_video_path)
    while True:
        ret, img = video_cap.read()
        if img is None:
            break

        window_name = "demo video"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        cv2.resizeWindow(window_name, (800, 600))
        cv2.waitKey(25)

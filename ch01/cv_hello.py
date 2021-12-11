import cv2

if __name__ == '__main__':
    print(" Hello CV")
    image = cv2.imread("ch01.png")
    print(" image shape: {}".format(image.shape))
    cv2.imshow("ch01", image)
    cv2.waitKey(0)

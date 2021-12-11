import cv2

if __name__ == '__main__':
    print(" Hello CV")
    image = cv2.imread("ch01.png")
    print(" image shape: {}".format(image.shape))

    cv2.rectangle(image, (20, 20), (30, 30), (255, 0, 0), 1)
    cv2.circle(image, (50, 50), 15, (0, 255, 0), 1)
    cv2.putText(image, "hi CV", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))

    cv2.imshow("ch01", image)
    cv2.waitKey(0)

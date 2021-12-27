import cv2

# read image
img = cv2.imread('ch01.png', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('origin', img)

# SIFT
detector = cv2.xfeatures2d.SIFT_create()
keypoints = detector.detect(gray, None)
img = cv2.drawKeypoints(gray, keypoints,img)
# img = cv2.drawKeypoints(gray,keypoints,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

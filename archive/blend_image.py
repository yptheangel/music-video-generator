import cv2

ori=cv2.imread("data/ori.jpg")
styled=cv2.imread("data/styled.jpg")
styled=cv2.resize(styled,(ori.shape[1],ori.shape[0]))

# alpha = 0.0
# beta = (1.0 - alpha)
# output = cv2.addWeighted(ori, alpha, styled, beta, 0.0)
# cv2.imshow("Blended Image",output)
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()

alpha_slider_max = 100

window_title = "Blended image"

def on_trackbar(val):
    # alpha = val / alpha_slider_max
    alpha = val / alpha_slider_max

    beta = ( 1.0 - alpha )
    dst = cv2.addWeighted(ori, alpha, styled, beta, 0.0)
    cv2.imshow(window_title, dst)

cv2.namedWindow(window_title)
trackbar_name = 'Alpha x %d' % alpha_slider_max
cv2.createTrackbar(trackbar_name, window_title , 0, alpha_slider_max, on_trackbar)
on_trackbar(0)
cv2.waitKey()
import numpy as np
import cv2



def color_detection(frame):
    # img = cv2.imread(frame)
    pixel_colors = frame.reshape((frame.shape[0] * frame.shape[1], 3))
    # print(pixel_colors)
    color_counts = np.unique(pixel_colors, axis=0, return_counts=True)
    most_frequent_color = color_counts[0][np.argmax(color_counts[1])]
    # print(most_frequent_color)
    bValue=most_frequent_color[0]
    gValue = most_frequent_color[1]
    rValue = most_frequent_color[2]
    # hsv_color = cv2.cvtColor(np.uint8([[most_frequent_color]]), cv2.COLOR_BGR2HSV)[0][0]
    # # print(hsv_color)
    # hueValue = hsv_color[0]
    # sValue = hsv_color[1]
    # vValue = hsv_color[2]
    # color = "Undefined"
    # if sValue < 20:
    #     color = "White"
    # elif vValue < 20:
    #     color = "Black"
    # elif vValue < 150 & sValue < 20:
    #     color = "Grey"
    # elif hueValue < 5:
    #     color = "Red"
    # elif hueValue < 22:
    #     color = "Orange"
    # elif hueValue < 33:
    #     color = "Yellow"
    # elif hueValue < 78:
    #     color = "Green"
    # elif hueValue < 131:
    #     color = "Blue"
    # elif hueValue < 170:
    #     color = "Violet"
    # else:
    #     color = "Red"
    color=get_color(bValue,gValue,rValue)
    # print(color)
    return color


def get_color(b,g,r):
    # Get the color of the pixel at (0, 0)

    # Convert BGR to RGB
    rgb = (r, g, b)
    print(r,g,b)
    color="undefined"
    # Check if the color is red
    if rgb[0] in range(200, 256) and rgb[1] in range(0, 100) and rgb[2] in range(0, 100):
       color="Red"
    # Check if the color is green
    elif rgb[0] in range(0, 100) and rgb[1] in range(200, 256) and rgb[2] in range(0, 100):
        color = "Green"
    # Check if the color is blue
    elif rgb[0] in range(0, 100) and rgb[1] in range(0, 100) and rgb[2] in range(200, 256):
        color = "Blue"
    # Check if the color is yellow
    elif rgb[0] in range(200, 256) and rgb[1] in range(200, 256) and rgb[2] in range(0, 100):
        color = "Yellow"
    # Check if the color is magenta
    elif rgb[0] in range(200, 256) and rgb[1] in range(0, 100) and rgb[2] in range(200, 256):
        color = "Magenta"
    # Check if the color is cyan
    elif rgb[0] in range(0, 100) and rgb[1] in range(200, 256) and rgb[2] in range(200, 256):
        color = "Cyan"
    elif rgb[0] in range(200, 256) and rgb[1] in range(200, 256) and rgb[2] in range(200, 256):
        color = "White"
    elif rgb[0] in range(0, 99) and rgb[1] in range(0, 99) and rgb[2] in range(0, 99):
        color = "Black"
    elif rgb[0] in range(100, 199) and rgb[1] in range(100, 220) and rgb[2] in range(100, 220):
        color = "Grey"
    # elif rgb[0] in range(150, 199) and rgb[1] in range(0, 99) and rgb[2] in range(150, 199):
    #     color = "Black"
    else:
        color = "undefined"
    return color

# import cv2
# import numpy as np
#
# def get_dominant_color(frame):
#     pixels = np.float32(frame.reshape(-1, 3))
#     n_colors = 5
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
#     flags = cv2.KMEANS_RANDOM_CENTERS
#     _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
#     palette = np.uint8(centroids)
#     quantized = palette[labels.flatten()]
#     dominant_color = palette[np.argmax(np.bincount(labels.flatten()))]
#     print(dominant_color)
#     return dominant_color
#
# cap = cv2.VideoCapture("slow.mp4")
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     dominant_color = get_dominant_color(frame)
#     frame[:] = dominant_color
#
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# img = cv2.imread('scTest3.jpg')
# color= color_detection(img)
# print(color)
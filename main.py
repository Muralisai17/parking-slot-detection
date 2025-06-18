import cv2
import pickle
import cvzone
import numpy as np
from datetime import datetime

# Load video feed
cap = cv2.VideoCapture('carPark.mp4')

# Load saved parking positions
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

# Slot dimensions
width, height = 107, 48

# Reserved slot indices (0-based)
reservedSlots = [1, 3, 7, 9]

# Function to check parking space status
def checkParkingSpace(imgPro):
    spaceCounter = 0

    for idx, pos in enumerate(posList):
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        # Determine slot state
        if idx in reservedSlots:
            color = (255, 0, 0)      # Blue for reserved
            label = "Reserved"
        elif count < 900:
            color = (0, 255, 0)      # Green for free
            label = "Free"
            spaceCounter += 1
        elif count < 1500:
            color = (0, 165, 255)    # Orange for misaligned
            label = "Misaligned"
        else:
            color = (0, 0, 255)      # Red for occupied
            label = "Occupied"

        # Draw rectangle and label on original color image
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, 2)
        cvzone.putTextRect(img, label, (x, y + height - 3), scale=1, thickness=2, offset=0, colorR=color)

    # Show free space count on the image
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 200, 0))


while True:
    # Restart video if finished
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()

    if not success or img is None:
        print("Failed to capture frame from video")
        break

    # Preprocessing pipeline
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    # Check parking spaces on processed binary image
    checkParkingSpace(imgDilate)

    # Add timestamp on the original image near top-left corner (adjust position if needed)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cvzone.putTextRect(img, now, (10, 30), scale=1, thickness=2, offset=5, colorR=(255, 255, 255))

    # Show windows
    cv2.imshow("Gray", imgGray)
    cv2.imshow("Blurred", imgBlur)
    cv2.imshow("Adaptive Threshold", imgThreshold)
    cv2.imshow("Median", imgMedian)
    cv2.imshow("Dilated", imgDilate)
    cv2.imshow("Image", img)

    # Wait key with short delay
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release video and close windows
cap.release()
cv2.destroyAllWindows()

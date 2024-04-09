import os
import cv2

DATA_DIR = './data'
LAST_CLASS_FILE = 'last_class.txt'
number_of_classes = 50  # Define the total number of classes
dataset_size = 730  # Define the size of the dataset for each class

# Read the index of the last processed class from file
last_class = 0
if os.path.exists(LAST_CLASS_FILE):
    with open(LAST_CLASS_FILE, 'r') as file:
        last_class = int(file.read())

# Set up the camera
cap = cv2.VideoCapture(0)

# Set the dimensions of the camera window
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 800, 600)  # Set the desired dimensions



for j in range(last_class, number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "K" to start collecting data for class {} or "Esc" to exit'.format(j), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(35)
        if key == ord('k'):
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            # Save the index of the last processed class to file
            with open(LAST_CLASS_FILE, 'w') as file:
                file.write(str(j))  # Increment to the next class for the next run
            exit()

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(35)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

# Save the index of the last processed class to file
with open(LAST_CLASS_FILE, 'w') as file:
    file.write(str(j + 1))  # Increment to the next class for the next run

cap.release()
cv2.destroyAllWindows()

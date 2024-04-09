import pickle
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

# Load the trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: Model file not found.")
    exit()

# Initialize the video capture
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Blind', 1: 'Hello', 2: 'I love You', 3: 'Yes', 4:'No', 5:'thank you', 6:'Please',7:'Phone',8:'Horse',9:'ThumbsUp'
               ,10:'PeaceSign',11:'Walk', 12:'Up',13:'Four',14:'Sit',15:'Nine',16:'W',17:'X',18:'R',19:'Down',
               20:'Area', 21:'O',22:'Talk',23:'Hear',24:'Eat',25:'Male',26:'ThumbsDown',27:'Fish',28:'Lion',29:'King',30:'strong',31:'Sun',32:'Fireman'
               ,33:'Sky',34:'Gun',35:'Hospital',36:'Bed',37:'Snake',38:'Frog',39:'Light',40:'Pig',41:'Child',42:'Insect',43:'Black',44:'Red'
               ,45:'Juice',46:'Secret',47:'Stay',48:'Forget',49:'Police'
               }

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            # Make predictions with the model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw rectangle and text on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
        except ValueError as e:
            print("please just use one hand as RandomForestClassifier expecting 42 features as input", e)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) 
    if key == ord('q'):  # Press 'q' to quit
        break
    
cap.release()
cv2.destroyAllWindows()
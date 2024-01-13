import pickle
import os
import cv2
import mediapipe as mp
import numpy as np


# Create a white image
width, height = 640, 480
white_image = np.ones((height, width, 3), np.uint8) * 255

# Set the title text
title = "Sign Language Detection"

# Define font parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 0, 0)
font_thickness = 2

# Calculate the position to center the title text
text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
text_x = (width - text_size[0]) // 2
text_y = 40

# Render the title on the white image
cv2.putText(white_image, title, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

# Display the white image
cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.imshow(title, white_image)

output_folder = 'predicted_snapshots'
os.makedirs(output_folder, exist_ok=True)

print("Started predicting...")
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',
               7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',
               14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',
               21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',
               28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z',
               36: 'again', 37: 'you', 38: 'thank you', 39: 'sorry',
               40: 'meet', 41: 'how', 42: 'food', 43: 'boy',
               44: 'girl', 45: 'who'}
try:
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
            
            ##print("halfway 1")
            temp=[np.asarray(data_aux)]
            _max_temp=84
            ##print("_max_temp")
            ##print(_max_temp)

            temp_temp_max_temp=[]
            for i in temp:
                i=np.pad(i,(0,_max_temp-len(i)),'constant',constant_values=(0))
                temp_temp_max_temp.append(i)
            ##print("temp_temp_max_temp")
            ##print(temp_temp_max_temp)

            prediction = model.predict(temp_temp_max_temp)



            ##print("prediction passed")
            predicted_character = labels_dict[int(prediction[0])]

            output_filename = os.path.join(output_folder, f'predicted_{predicted_character}.jpg')
            cv2.imwrite(output_filename, frame)


            print(predicted_character)

            font_path = 'NIRMALA.TTF'  # Replace with the actual font file path
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.3
            font_color = (0, 0, 0)
            font_thickness = 3

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 10)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),font,font_scale, font_color, font_thickness, cv2.LINE_AA)

  
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
except Exception as X:
    print("something went wrong")
    print(X)

finally:
    cap.release()
    cv2.destroyAllWindows()

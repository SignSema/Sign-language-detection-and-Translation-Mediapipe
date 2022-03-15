# Import Dependencies

import cv2
import numpy as np
import mediapipe as mp
import pickle
import joblib
import csv
import pandas as pd
import pyttsx3

# # Definitions

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks_B(image, results):
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness = 2,circle_radius=3),
                             mp_drawing.DrawingSpec(color=(0,0,255), thickness = 2,circle_radius=1)
                             )
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness = 2,circle_radius=3),
                             mp_drawing.DrawingSpec(color=(0,0,255), thickness = 2,circle_radius=1)
                             )


# # # Voice


def speak(text):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', 150)

    #Setting the voice
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    #Text input
    engine.say(text)
    engine.runAndWait()


# # # # Load model

model = joblib.load('MP_model.pkl')
    

# # # # # Make Detections

sentence = []
predictions = []
threshold = 0.8
#minimum number of predictions for confirmation
pr = 3

cap = cv2.VideoCapture(0)
#set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        #read frame
        ret, frame = cap.read()

        #make detections
        image, results = mediapipe_detection(frame, holistic)
               
        #draw landmarks
        #draw_landmarks(image, results)
        draw_styled_landmarks_B(image, results)
        
        #Export Cordinates
        try:
             #Extract hand and face  landmarks
            lh_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3))
            rh_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3))
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3))
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3))
            
            #Concatenate rows
            row = lh_row + rh_row + face_row + pose_row
            
            #Make Detections
            X = pd.DataFrame([row])
            sign_class = model.predict(X)[0]
            sign_prob = model.predict_proba(X)[0]
                      
            #Sentence Logic
            if sign_prob[np.argmax(sign_prob)] > threshold:
                predictions.append(sign_class)
                print(sign_class, sign_prob[np.argmax(sign_prob)])
                if predictions[-pr:] == [sign_class]*pr:
                    if len(sentence) > 0:
                        if sign_class != sentence[-1]:
                            sentence.append(sign_class)
                            speak(sign_class)
                    else:
                        sentence.append(sign_class)
                        speak(sign_class)
                    
            
            if len(sentence) > 5:
                    sentence = sentence[-5:]
            
            cv2.rectangle(image, (0,0), (640,40),(0,0,0), -1 )
            cv2.putText(image,  ' '.join(sentence), (3,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        except:
            pass
        
        #show to screen
        cv2.imshow('OpenCV Feed', image)
        
        #break gracefully
        if cv2.waitKey(10) & 0xFF ==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()








import cv2
import numpy as np
import mediapipe as mp
import pickle
import csv
import pandas as pd
import pyttsx3
import joblib
import time


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



def draw_styled_landmarks_G(image, results):
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,255,0), thickness = 2,circle_radius=3),
                             mp_drawing.DrawingSpec(color=(0,255,0), thickness = 2,circle_radius=1)
                             )
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,255,0), thickness = 2,circle_radius=3),
                             mp_drawing.DrawingSpec(color=(0,255,0), thickness = 2,circle_radius=1)
                             )


def draw_styled_landmarks_np_nf_B(image, results):
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness = 2,circle_radius=3),
                             mp_drawing.DrawingSpec(color=(0,0,255), thickness = 2,circle_radius=1)
                             )
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness = 2,circle_radius=3),
                             mp_drawing.DrawingSpec(color=(0,0,255), thickness = 2,circle_radius=1)
                             )


# ## Voice


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


# # Make Detections


model_L = joblib.load('MP_model_head.pkl')



def sign_output(sign_list, sentence, sentence_out):
    with open('multi_sign.csv') as multisign_file:
        sign_list = csv.reader(multisign_file)
        for row in sign_list:
            if sentence[-1] == row[-1]:
                if sentence[-2] == row[-2]:
                    sentence_out.append(row[0])
                    break
            else:
                continue



def detect(vidsource):
    
    sentence = []
    sentence_out = []
    
    predictions = []
    
    last_sign_list = []
    one_sign_list = []
    
    #minimum probability
    threshold = 0.9
    
    #minimum number of predictions for confirmation
    pr = 3
    
    #for fps calculation
    pTime = 0
    cTime = 0
    
    #Loading complex signs mechanism
    with open('multi_sign.csv') as multisign_file:
        sign_list = csv.reader(multisign_file)
        for row in sign_list:
            last_sign_list.append(row[-1])
    
    #Loading simple signs
    with open('single_sign.csv') as singlesign_file:
        singlesign_list = csv.reader(singlesign_file)
        for row in singlesign_list:
            one_sign_list.append(row[0])
    
    #Detecting from source of video feed
    cap = cv2.VideoCapture(vidsource)
    
    #set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
        #for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):

            #read frame
            ret, frame = cap.read()

            #make detections
            image, results = mediapipe_detection(frame, holistic)
            
            #Draw for tracking
            draw_styled_landmarks_np_nf_B(image, results)

            #Extract landmarks
            lh_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3))
            rh_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3))
            
            if results.pose_landmarks:
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h,w,c = frame.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
         
                    if id == 0:
                        if lm.visibility > 0.8:
                            head = list(np.array([lm.x, lm.y, lm.z]))
                        else:
                            head =list(np.zeros(1*3))
                      
            
            #Concatenate rows
            row = lh_row + rh_row + head

            #Make Detections
            X = pd.DataFrame([row])
            sign_class = model_L.predict(X)[0]
            sign_prob = model_L.predict_proba(X)[0]

            #Sentence Logic
            if sign_prob[np.argmax(sign_prob)] > threshold:
                predictions.append(sign_class)

                #print(sign_class, sign_prob[np.argmax(sign_prob)])


                if predictions[-pr:] == [sign_class]*pr:
                    if len(sentence) > 0:
                        if sign_class != sentence[-1]:
                            sentence.append(sign_class)
                            
                            #Shows that a detection has been made
                            draw_styled_landmarks_G(image, results)
                            
                            #checks if a sign is a complex sign(last sign of the complex sign)
                            if sentence[-1] in last_sign_list:
                                sign_output(sign_list, sentence, sentence_out)
                            
                            #Checks if a sign is a simple sign
                            if sentence[-1] in one_sign_list:
                                sentence_out.append(sign_class)

                            #speak(sign_class)
                    else:
                        sentence.append(sign_class)
                        draw_styled_landmarks_np_nf_B(image, results)
                        if sentence[-1] in one_sign_list:
                                sentence_out.append(sign_class)
                        #speak(sign_class)


            if len(sentence) > 5:
                    sentence = sentence[-5:]
                    
                    
            if len(sentence_out) > 6:
                    sentence_out = sentence_out[-6:]
                    #speak(sentence_out)

            cv2.rectangle(image, (0,0), (640,40),(0,0,0), -1 )
            cv2.putText(image,  ' '.join(sentence), (3,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.rectangle(image, (0,80), (640,40),(255,0,0), -1 )
            cv2.putText(image,  ' '.join(sentence_out), (3,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            #fps
            cTime = time.time()
            fps = 1/ (cTime - pTime)
            pTime = cTime

            cv2.putText(image,"fps",(5,415), cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),2)
            cv2.putText(image,str(int(fps)),(10,460), cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),5)

   
            #show to screen
            cv2.imshow('OpenCV Feed', image)

            #break loop
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()



detect(0)


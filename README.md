# Sign Language Detection and Translation.
 Using mediapipe holistic we can estimate the position of various face, hand and body-pose keypoints.
 
![dance](https://user-images.githubusercontent.com/68475422/155277775-6f41e20a-4e85-499b-b37f-9711ec7239f0.gif)

![bandicam 2022-02-18 21-21-37-650 mp4_20220218_214008 mp4_20220223_104116 006](https://user-images.githubusercontent.com/68475422/155278423-fa1b081e-b393-40b6-ae47-94cc7fbfeb6f.png)


 The body movement of sign language gestures are estimated by the AI and the keypoints of these movements are extracted as numpy arrays.

 These numpy arrays are fed into an LSTM Neural Network as the datasets and the neural network is trained for 500 epochs.

 Using the camera feed the model can make predictions whose probabilities can also be displayed as a rectangle of dynamically changing length.
 
 [![Sign language](https://user-images.githubusercontent.com/68475422/155279503-c5ada12b-a87a-416a-920b-79de0e633951.png)](https://youtu.be/AKNrkSKYvuY)

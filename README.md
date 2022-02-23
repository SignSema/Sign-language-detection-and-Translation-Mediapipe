# Sign Language Detection and Translation.
 Using mediapipe holistic we can estimate the position of various face, hand and body-pose keypoints.
 
 [![bandicam 2022-02-12 11-00-15-551 mp4_20220212_111159 mp4_20220213_113443 143](https://user-images.githubusercontent.com/68475422/155275908-08d8c400-361e-4a95-b5ed-d0e5dbb8db8b.png)](https://www.youtube.com/watch?v=H7hxbDMA7Yo)


 The body movement of sign language gestures are estimated by the AI and the keypoints of these movements are extracted as numpy arrays.

 These numpy arrays are fed into an LSTM Neural Network as the datasets and the neural network is trained for 500 epochs.

 Using the camera feed the model can make predictions whose probabilities can also be displayed as a rectangle of dynamically changing length.

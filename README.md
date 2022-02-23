# Sign Language Detection and Translation.
 Using mediapipe holistic we can estimate the position of various face, hand and body-pose keypoints.

 The body movement of sign language gestures are estimated by the AI and the keypoints of these movements are extracted as numpy arrays.

 These numpy arrays are fed into an LSTM Neural Network as the datasets and the neural network is trained for 500 epochs.

 Using the camera feed the model can make predictions whose probabilities can also be displayed as a rectangle of dynamically changing length.

# Sign Language Detection and Translation.
 Using mediapipe holistic we can estimate the position of various face, hand and body-pose keypoints.
 
![dance](https://user-images.githubusercontent.com/68475422/155277775-6f41e20a-4e85-499b-b37f-9711ec7239f0.gif)

![bandicam 2022-02-18 21-21-37-650 mp4_20220218_214008 mp4_20220223_104116 006](https://user-images.githubusercontent.com/68475422/155278423-fa1b081e-b393-40b6-ae47-94cc7fbfeb6f.png)

By collecting the data extracted by the mediapipe library and training a machine learning model on sign language gestures a sign language interpreter can be made.


First try.

[![Sign language](https://user-images.githubusercontent.com/68475422/155279503-c5ada12b-a87a-416a-920b-79de0e633951.png)](https://youtu.be/AKNrkSKYvuY)
 
Second try, better... scalable.

[![bandicam 2022-03-11 16-39-57-143 mp4_20220311_173432 avi_20220311_174737 avi_20220311_175948 883](https://user-images.githubusercontent.com/68475422/157896151-47838fbd-274e-4d96-831b-53ef3845e14a.png)](https://youtu.be/7fn5HuKR7D4)

Third version with complex signs, the sign detector is also a custom neural network.

[![sign demo](https://user-images.githubusercontent.com/68475422/172781778-f0c4f75b-5c2b-4425-887c-64b9e0008897.png)](https://youtu.be/-XmaHa2LlSo)

# Instructions for running and testing.

1. Create and activate an environment in which all the imports are installed. (All the imports are at the top of the code. I used an anaconda environment but a standard python environment should work as well)
2. cd to the directory where both sign-language-detection-and-translation.py, MP_model_head.pkl and both csvs are stored
3. Python run sign-language-detection-and-translation.py or run all the cells of the jupytr notebook.(They're the same)

## Trained signs 
Hello (waving with palm facing camera), my (right hand palm flat on chest), name (both hands letter h with right-hand fingers resting on left-hand fingers), me (right hand pointing at your chest), The entire alphabet as shown [here](https://www.youtube.com/watch?v=WNigt-vfTX0&t=5s).

Added signs: you (pointing at camera), your (flat palm towards camera), 'Sup, What's up!(last two are in linkedin video), how (both hands thumbs up twisting against each other) mostly as shown [here](https://www.youtube.com/watch?v=nJx-XsxeajQ&t=7s)

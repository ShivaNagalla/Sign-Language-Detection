
Sign Language Detection:

• Welcome! This project lets you train your own sign language detection model and run real-time recognition using your webcam. The model is based on Transformers and uses hand landmarks detected by MediaPipe.

---

Prerequisites:

• You’ll need Python 3.8 or above for this project.  
• Before getting started, make sure to install all the required packages. You can do this with the following command:

```bash
pip install -r Requirements.txt
```

---

Project Overview:

• This repository contains code and data for detecting three American Sign Language (ASL) gestures: "please", "thank you", and "sorry".  
All the training data is stored as NumPy files in the `MP_Data` folder. You’ll find scripts for both training your model and running real-time detection.

Here’s what you’ll find inside the main project folder:

• The `MP_Data` directory contains all the gesture data, organized by gesture and by sequence.
• The script `train_model.py` is used to train your own sign detection model from the data.
• The script `real_time_detection.py` allows you to use your webcam to detect gestures in real time.
• The `Requirements.txt` file lists all the necessary dependencies.

---

Data Format:

• The gesture data is stored in the `MP_Data` folder. It’s already preprocessed and organized so each sign ("please", "thank you", or "sorry") has its own folder. Inside each gesture’s folder, you’ll find numbered folders representing different sequences. Each sequence folder contains 30 frames saved as `.npy` files (NumPy arrays).

For example:

```
MP_Data/
  ├── please/
  │     └── 0/
  │          ├── 0.npy
  │          ├── 1.npy
  │          └── ... (up to 29.npy)
  ├── thankyou/
  └── sorry/
```

• Each `.npy` file is a set of hand keypoints for one frame.

• If you want to add more data, you would need to create a new data collection script that saves hand keypoints using MediaPipe.

---

How to Train the Model:

• Once you have your data ready, you can train your sign language detection model.  
• Just run this command in your terminal:

```bash
python train_model.py
```

• The script will load the training data, extract hand features, train a transformer model, and save it as `asl_transformer_model.h5` in the project directory.

---

Real-time Detection:

• After training your model, you can use your webcam for live sign language detection.  
Simply run:

```bash
python real_time_detection.py
```

• This script will start your webcam and continuously look for hand signs. When it detects one of the three gestures ("please", "thank you", or "sorry"), it will display the result on the video feed in real time.

---
How the Scripts Work:

train_model_new.py

- Loads all gesture sequences from the `MP_Data` directory.
- Extracts and normalizes hand features for each frame.
- Uses a Transformer model with positional encoding to learn from the gesture data.
- Splits data into training, validation, and test sets and saves the trained model to disk.

real_time_detection_new.py

- Loads the trained model (`asl_transformer_model.h5`).
- Opens your webcam and uses MediaPipe to extract hand landmarks from each frame.
- Buffers the last 30 frames and predicts which gesture is being shown.
- If the prediction is confident enough, it displays the detected sign on the video window.

---

 Requirements:

Let’s talk about setting up your environment.  
First, make sure you have Python 3.8 or newer installed.

To make your life easy, all the libraries you need are already listed in the `Requirements.txt` file that comes with this project. Here’s what to do next:

1. Open your terminal or command prompt.
2. Go to the folder where you downloaded or cloned this project.
3. Run this command:

```bash
pip install -r Requirements.txt
```

That’s it! This single step will install everything the code needs. Here’s a quick rundown of the most important packages:

- NumPy: Helps you work with numerical data and arrays.
- OpenCV: Lets the code access and process webcam video.
- MediaPipe: Detects and tracks hand landmarks in real time.
- TensorFlow : Runs and trains the AI model for sign language recognition.
- Scikit-learn : Helps split your data and supports the training pipeline.

• If you see any errors about missing packages, double-check your Python version and try running the install command again. Sometimes, creating a new virtual environment (using `venv` or `conda`) can help avoid conflicts with old packages.

• If you get stuck during setup or have questions, don’t hesitate to ask or open an issue on GitHub. We’re here to help!

---

Customizing the Project:

• Currently, the project recognizes "please", "thank you", and "sorry". If you want to recognize other signs, you’ll need to collect new data and update the `ACTIONS` list in both scripts to include your new gesture labels.

• Make sure to use your webcam in a well-lit area with a plain background for the best results.

---

If you have any questions, want to contribute, or run into issues, feel free to open an issue or reach out!

---
"# Sign-Language-Detection" 

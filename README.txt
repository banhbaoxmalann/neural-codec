NEURAL AUDIO CODEC - README
Project Title: Neural Audio Codec for Cat vs Dog Sound Classification
Author: Trần Tuấn Anh
Date: 19/4/2026


REQUIREMENT:
------------
Python Version: 3.9 or higher

Required Libraries:
  - numpy (>=1.21.0)
  - librosa (>=0.9.0)
  - soundfile (>=0.10.0)
  - matplotlib (>=3.4.0)
  - scipy (>=1.7.0)
  - IPython (>=7.0.0)
  - streamlit (>=1.0.0) - for demo only
  - kaggle (>=1.5.0) - for dataset download

Installation command:
  pip install numpy librosa soundfile matplotlib scipy ipython streamlit kaggle

DATASET
-------
Source: Kaggle "Cats vs Dogs Audio Classification" dataset
URL: https://www.kaggle.com/datasets/stealthtechnologies/cats-vs-dogs-audio-classification
STEP 1: Create a Kaggle Account
  - Go to https://www.kaggle.com
  - Sign up for a free account
  - Verify your email address

STEP 2: Accept Dataset Terms
  - Navigate to the dataset URL above
  - Click "Add" or "Download" button
  - Read and accept the competition rules and dataset terms
  - You MUST accept the terms before downloading

STEP 3: Generate Kaggle API Token
  - Go to your Kaggle account settings: https://www.kaggle.com/account
  - Scroll to "API" section
  - Click "Create New Token"
  - This downloads a file called "kaggle.json" to your computer

STEP 4: Get Your API Token
  - Open the downloaded "kaggle.json" file
  - Look for the "key" field
  - Example content:
    {"username":"your_username","key":"40c900374d46e1e88f4b3c1c3d17d2f7"}
  - Copy the key value (long string of letters and numbers)

STEP 5: Set Token in Code
  In your Python script or notebook, set the token BEFORE downloading:
  
  import os
  os.environ['KAGGLE_API_TOKEN'] = "your_api_key_here"
  
  Replace "your_api_key_here" with the actual key from kaggle.json
 
STEP 6: Download the Dataset
  Run the following commands:
Download button

Dataset Statistics:
  - Total files: 484 audio files (242 cat, 242 dog)
  - Format: WAV, mono
  - Sample rate: Resampled to 16kHz (training) or 8kHz (demo)
  - Segment length: 16,384 samples (~1 second at 16kHz)


TRAINING CONFIGURATION
----------------------

Parameters:
  - Batch size: 4-8
  - Learning rate: 0.001
  - Epochs: 10-50 
  - Optimizer: SGD
  - Loss weighting (β): 1.0

Training Environment:
  - Platform: Google Colab (free tier)
  - Hardware: CPU only
  - RAM: 12 GB
  - Storage: Google Drive (for checkpoints)

Training Time:
  - ~12 hours for 50 epochs (CPU)
  - Checkpoints saved every 5 epochs

TRAIN MODEL
-----------

1. Open Test.ipynb in Google Colab
2. Mount Google Drive:
   from google.colab import drive
   drive.mount('/content/drive')
3. Set your Kaggle API token in the environment:
   os.environ['KAGGLE_API_TOKEN'] = "your_token_here"
4. Run all cells sequentially:
   - Cell 1: Download dataset from Kaggle
   - Cell 2: Define model architecture
   - Cell 3: Training loop
   - Cell 4: Evaluate on cat vs dog sounds
5. Wait until it finish, usually take a very long time. 

There will be likely that the google colab will automatically close before we 
finish our training session. That's why in the model we've already implemented 
the function to save parameter in each checkpoints on Google Drive to help the 
model keep on training from where it was stopped.

Interactive Demo 
----------------

1. Make sure that the weight folder(the folder that save all the weight) and the dataset folder is in the same folder as Live_Demo.py.
	main folder/
	|_____livedemo.py
	|_____ weight/
	|	|___training data
	|_____ Dvc/
		|___Dogs
		|___Cats
1. Open Live_Demo.py in Visual studio code.
2. Run the programme.
3. Paste the command "streamlit run c:\Users\Admin\Desktop\168225_25203_Tran_Tuan_Anh_Nguyen_Ngoc_Ninh\Code submission\livedemo.py" in the terminal.
4. Open the URL and enter the password to access the Streamlit web interface (The compiler may automatically open the url for you).
5. Select pre-loaded cat/dog samples.
6. Click the "Run Encode/Decode Comparison button"
7. Compare original vs reconstructed audio waveforms and spectrograms.

MODEL WEIGHTS
-------------
Saved checkpoints are stored in Google Drive at:
  /content/drive/MyDrive/Numpy_Codec_Weights_Balanced/

Files saved:
  - encoder_epoch{X}.npz    - Encoder weights (kernel and bias for 3 conv layers)
  - decoder_epoch{X}.npz    - Decoder weights (kernel and bias for 3 transposed conv layers)
  - vq_codebook_epoch{X}.npy - VQ codebook embeddings

Loading weights:
  model = NeuralCodec(num_embeddings=512, embedding_dim=128)
  model.load(SAVE_DIR, epoch=10)  


-------------
END OF README

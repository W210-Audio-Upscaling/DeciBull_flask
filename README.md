# DeciBull - Audio Upscaling
[Check out our website](decibull.app)

This flask app is designed to run the DeciBull audio upscaling algorithm. 

**How to Use**:

 **RECORD** and capture microphone input:
 1. Allow the website to record microphone input.
 2. Click "Record" and say a few words (just a few seconds)
 3. Click "Stop". This will stop recording and cause an audio player to appear. This clip is the original audio that you can compare to the final result. 
 4. Click "Predict". This will run the audio file through our model, and prompt you to download it. 
 5. Download and play the file to see results
 6. Click "Refresh". The app will then display how long your audio file was, and how long it took to be upsampled. 

If you choose to **BROWSE** and upload a file (.wav format):
1.	The "Browse" button currently does not work, but if I remove it it breaks the "Refresh" button so ¯\\\_(ツ)\_/¯

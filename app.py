from subprocess import run, PIPE
from flask import Flask, render_template, request, flash, redirect, url_for, send_file
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import time
from pydub import AudioSegment
from pydub.playback import play
import torch
import torchaudio
import torch.nn as nn
from tqdm.notebook import tqdm
from torch import tensor
import librosa

ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.secret_key = "VaishSughosh%$1234"
model = torch.load(open('model_gen.save', 'rb'))

class Generator(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.dropout1 = nn.Dropout(p=0.2)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.fc3 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.dropout1(x)
        x = self.relu4(x)
        return x


@app.route('/', methods=['GET', 'POST'])
def index():
    ip_address = "original"
    bird_path = ''
    if Path('tmp/audio'+ip_address+'.wav').is_file():
        Path('tmp/audio'+ip_address+'.wav').unlink()
        return render_template('index.html', bird = bird_path)
    return render_template('index.html', bird = bird_path)

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

@app.route('/help', methods=['GET', 'POST'])
def help():
    return render_template('help.html')

@app.route('/privacy',methods=['GET', 'POST'])
def privacy():
    return render_template('privacy.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    return render_template('feedback.html')

@app.route('/sitemap.xml', methods=['GET', 'POST'])
def sitemap():
    return render_template('sitemap.xml')

def allowed_file(f):
    return '.' in f and \
           f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/audio', methods=['POST'])
def audio():
    ip_address = "wavefile"
    with open('./tmp/audio'+ip_address+'.wav', 'wb') as f:
        f.write(request.data)
    proc = run(['ffprobe', '-of', 'default=noprint_wrappers=1', './tmp/audio'+ip_address+'.wav'], text=True, stderr=PIPE)
    return proc.stderr

@app.errorhandler(413)
def request_entity_too_large(error):
    flash("File too large", "error")
    return redirect(url_for("index"),413)

@app.errorhandler(500)
def request_internal_server_error(error):
    flash("Recording is too small. Please use another audio", "error")
    return redirect(url_for("index"),500)

@app.errorhandler(503)
def request_file_too_big(error):
    flash("Audio file is too large. Please use a file with size less than 5 MB", "error")
    return redirect(url_for("index"),503)

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    f = (request.files['file'])
    mfcc_=[]
    ip_address = "wavefile"
    PATH = Path('tmp/audio'+ip_address+'.wav')
    if f.filename == '' or Path('/tmp/audio'+ip_address+'.wav').is_file():
        if Path('tmp/audio'+ip_address+'.wav').is_file():
            t0 = time.time()
            AudioSegment.from_wav(PATH).export("tmp/downsample.mp3", format="mp3")
            audio_len = librosa.get_duration(filename=PATH)
            waveform_test, sample_rate_test = torchaudio.load("tmp/downsample.mp3")
            chunk_size = 2048
            total_chunk = int(waveform_test[0].size()[0]/2048)

            generator = Generator(chunk_size, chunk_size, chunk_size)
            generator.load_state_dict(model)
            generator.double()

            bucket = None
            for i in tqdm(range(1, total_chunk)):
                X_chunk = waveform_test[0][chunk_size*(i-1):chunk_size*i]
                prediction = generator(X_chunk.double())
                summed = X_chunk + prediction
                if i == 1:
                    bucket = summed
                    print(bucket.size())
                else:
                    bucket = torch.cat((bucket, summed), 0)
                    
            output = bucket.squeeze(0)
            torchaudio.save('tmp/saved.wav', output, 48000, precision=32)


            t1 = time.time()
            Path('tmp/audio'+ip_address+'.wav').unlink()
            flash("{:.2f} seconds of audio upsampled in {:.2f} seconds.".format(audio_len, t1-t0), "info")
            return send_file('tmp/saved.wav', as_attachment=True) #, send_file(Path("downsample.mp3"), as_attachment=True)

        else:
            flash("No audio recorded or uploaded", "error")
            return redirect(url_for("index"))
        
    if f and allowed_file(f.filename):
        X, sample_rate = librosa.load(f, sr = 44100, res_type='kaiser_best')
        mfcc_.append(np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T,axis=0))
        data_test= pd.DataFrame.from_records(mfcc_, columns=labels)
        prediction = model.predict(data_test)
        class_probabilities = model.predict_proba(data_test)
        max_prob = np.max(class_probabilities)
        if max_prob>0.7:
            output = prediction[0]
            if output == 'Cardinal':
                bird_path = 'https://lh3.googleusercontent.com/ej9UTPrx_kcYkYgr_berGZ-Y7T2q0Emi9yMuexW_3fzslYXBwOmOn9NBiHlnZNDPQzq6-BFghf6CSVfcuu-22-tEUgKDpuu_nnm5Tq1DjyG1R3pAhGqV4RrqaCUpSONQPWZwS2X2pg=w2400'
                bird_link = 'https://www.allaboutbirds.org/guide/Northern_Cardinal/overview'
            if output == 'Mourning Dove':
                bird_path = 'https://lh3.googleusercontent.com/SkBJmdsjOi8H6kBvKeUB0cdbxuj2W036ABWkEt1JZ-aTXq0L0Eyuv4i7pmU6HBwuaHcG9P7kYuz45Qm0izsHquYQy2qyun1C2UmgIEhB6qN2XGY-Gr42DI6-0-3bnww48SdmmuOnTg=w2400'
                bird_link = 'https://www.allaboutbirds.org/guide/Mourning_Dove'
            if output == 'Pigeon':
                bird_path = 'https://lh3.googleusercontent.com/8du1iStq3vN954_k-_DTD6PxEoz6JDG5URiwVqT1SlUPw62DwevItGwZw2r-5fAX3yPEmSHBNaeCnucYgwZ923NueykVt0uxH1Kf8-RS-Mn_tgXg4tW69h8K9j3n0njU8tsaYuDtAQ=w2400'        
                bird_link = 'https://www.allaboutbirds.org/guide/Rock_Pigeon'
            if output == 'Blue Jay':
                bird_path = 'https://lh3.googleusercontent.com/00uRzeBJWqlqEKiG44lt7ybQ-3-mZEUHprr8GxOb9D05RJ7whkjq5SkrycZiuU5qkx0RIsIdVZcpzBS4lIngYbe1bQeqeb4t7L0IGrY5YHxyJmzPaUCfDrQauXqRhab5No5VnbfZ8Q=w2400'        
                bird_link = 'https://www.allaboutbirds.org/guide/Blue_Jay'
            flash("It's a {}! ({:.2f}% probability)".format(output,max_prob*100), "info")

            return render_template('index.html', bird = bird_path, birdlink = bird_link, birdmore = bird_more)
        else:
            
            return redirect(url_for("index"))
    else:       
        flash("Use a WAV format file", "error")
        return redirect(url_for("index"))

if __name__ == "__main__":
    
    app.run(debug=True)

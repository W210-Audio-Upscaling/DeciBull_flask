from subprocess import run, PIPE
from flask import Flask, render_template, request, flash, redirect, url_for
import pickle
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request

ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.secret_key = "VaishSughosh%$1234"
model = pickle.load(open('Bird Predictor_neural_v4.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    ip_address = ip_address = urllib.request.urlopen('https://ident.me').read().decode('utf8').replace(':','')
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
    ip_address = ip_address = urllib.request.urlopen('https://ident.me').read().decode('utf8').replace(':','')
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
    labels=['var1','var2','var3','var4','var5','var6','var7','var8','var9','var10','var11','var12','var13']
    bird_more = 'https://lh3.googleusercontent.com/89AWJHL7bzdieXEV9GmJ5AcDa_pAh1GQcf5_YAfmanjo6GRFtxNzxO67QDsMV9SfOO9CrHDY5W0teRcwVlRxoG_zgdI6s9w8LcyDKL8XX6Dtjk4L6sMGBKToz3Hb-BfuWVsmtFQmFQ=w2400'
    ip_address = ip_address = urllib.request.urlopen('https://ident.me').read().decode('utf8').replace(':','')
    if f.filename == '' or Path('/tmp/audio'+ip_address+'.wav').is_file():
        if Path('tmp/audio'+ip_address+'.wav').is_file():
            X, sample_rate = librosa.load((Path('tmp/audio'+ip_address+'.wav')), sr = 44100, res_type='kaiser_best')
            mfcc_.append(np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T,axis=0))
            data_test= pd.DataFrame.from_records(mfcc_, columns=labels)
            prediction = model.predict(data_test)
            class_probabilities = model.predict_proba(data_test)
            max_prob = np.max(class_probabilities)
            Path('tmp/audio'+ip_address+'.wav').unlink()
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
                    bird_path = 'https://lh3.googleusercontent.com/7E6ZjfAeeHwenJl3DDTC_Ni9aQBzbBv100VMRNjpiY-be6fzHlGaVJek1V_bhQRBVSAGSqYZaoTbj_RtZC-WjN6-x-2vnFd3UcO7YVAna9u1vkpfC5dLxk4dSOYnpbF8HK2d8WoOD19GEGwdKNQ1bPKqS7A7RMidv8oKKOa5m0CWfh69Og93eMx6EpXFTOL9AiMTlKhJjZOt-6tbw10B88qRrP2jeDp6LAWLceRVf6kvQmoXtwm487y9ol3I0spbTqCQUXa7BNL3cbXdwq1p0rIecDdCMk2gXgIGSoKr0pPH_rCmdBYkXwG5mFovgjl_129b_j6hJ3EDn6Of5YPp1wby0cC_4q2UaIxQIzQ9JBt10za2y1mTzmUK3aDeq1ZNA96ML2T_YmZ7HjTVBKyIJHFhmGBCOI5Mt6Ibd__PwCRG461WCM1N46FPEk3PevVeM_N1EmAIYp1YsQo90rXJ4jJKbhz7nRf5lpEsg6hSTM32RF4kivaZWEdxHIqACV0WOoXyOZm9OBnsYiWV3vUU1SNQFjzHXAcxJfn9c1hDm0dodO76zVCINjF6hBmjta-7rWOFc-ecD5Nep_Lwsnuz54-gbKvF3l1k6LtMcKPhBUDii_dxjrPlRhMDMK-3R7Gwu6s8e35JDP2pESMWsmNY98_LAMmICNAjCndDEGLEGE98Q_qOohd3wejNE_Nv=w2171-h1643-no?authuser=2'        
                    bird_link = 'https://www.allaboutbirds.org/guide/Blue_Jay'
                flash("It's a {}! ({:.2f}% probability)".format(output,max_prob*100), "info")
                return render_template('index.html', bird = bird_path, birdlink = bird_link, birdmore = bird_more)
            else:
                flash("Audio not recognized", "info")
                return redirect(url_for("index"))
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
                bird_path = 'https://lh3.googleusercontent.com/7E6ZjfAeeHwenJl3DDTC_Ni9aQBzbBv100VMRNjpiY-be6fzHlGaVJek1V_bhQRBVSAGSqYZaoTbj_RtZC-WjN6-x-2vnFd3UcO7YVAna9u1vkpfC5dLxk4dSOYnpbF8HK2d8WoOD19GEGwdKNQ1bPKqS7A7RMidv8oKKOa5m0CWfh69Og93eMx6EpXFTOL9AiMTlKhJjZOt-6tbw10B88qRrP2jeDp6LAWLceRVf6kvQmoXtwm487y9ol3I0spbTqCQUXa7BNL3cbXdwq1p0rIecDdCMk2gXgIGSoKr0pPH_rCmdBYkXwG5mFovgjl_129b_j6hJ3EDn6Of5YPp1wby0cC_4q2UaIxQIzQ9JBt10za2y1mTzmUK3aDeq1ZNA96ML2T_YmZ7HjTVBKyIJHFhmGBCOI5Mt6Ibd__PwCRG461WCM1N46FPEk3PevVeM_N1EmAIYp1YsQo90rXJ4jJKbhz7nRf5lpEsg6hSTM32RF4kivaZWEdxHIqACV0WOoXyOZm9OBnsYiWV3vUU1SNQFjzHXAcxJfn9c1hDm0dodO76zVCINjF6hBmjta-7rWOFc-ecD5Nep_Lwsnuz54-gbKvF3l1k6LtMcKPhBUDii_dxjrPlRhMDMK-3R7Gwu6s8e35JDP2pESMWsmNY98_LAMmICNAjCndDEGLEGE98Q_qOohd3wejNE_Nv=w2171-h1643-no?authuser=2'        
                bird_link = 'https://www.allaboutbirds.org/guide/Blue_Jay'
            flash("It's a {}! ({:.2f}% probability)".format(output,max_prob*100), "info")
            return render_template('index.html', bird = bird_path, birdlink = bird_link, birdmore = bird_more)
        else:
            flash("Audio not recognized", "info")
            return redirect(url_for("index"))
    else:       
        flash("Use a WAV format file", "error")
        return redirect(url_for("index"))

if __name__ == "__main__":
    
    app.run(debug=True)

from __future__ import division, print_function
from werkzeug.exceptions import HTTPException

import numpy as np
import os
import io

import librosa
import keras
from keras import backend as K
from keras.models import Sequential,load_model
from flask import Flask,jsonify,redirect, url_for, request, render_template

import logging
#from playsound import playsound
debug = True
app = Flask(__name__)
logging.basicConfig(filename='demo.log', level=logging.DEBUG)
def get_model():
    global model555
    model555=load_model('/var/www/html/flaskproject1/covid/model5.h5')
    print("Model loaded")

get_model()



def wav2predict(sf):

  all_data=[]
  f, sr = librosa.load(sf, duration=2.5)
  st = librosa.stft(f[16000:38000])
  stft = np.abs(st)
  mfcc1 = librosa.feature.mfcc(f[16000:38000])
  spectral_bandwidth = librosa.feature.spectral_bandwidth(y=f[16000:38000], sr = sr)
  spectral_centroid = librosa.feature.spectral_centroid(y=f[16000:38000], sr = sr)
  zero_crossing =  librosa.feature.zero_crossing_rate(f[16000:38000])
  spec_roll = librosa.feature.spectral_rolloff(f[16000:38000])
  d = np.vstack((stft,mfcc1,spectral_bandwidth,spectral_centroid,zero_crossing,spec_roll))
  all_data.append(d)
  all_data = np.array(all_data)
  all_data = all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], 1)
  a = model555.predict(all_data)
  idx = np.argmax(a)
  if idx == 1:
    classs = 0
  elif idx == 0:
    classs = 1
  prop = np.max(a)

  return classs,prop

@app.route('/', methods=['GET'])
def index():
    app.logger.info('Processing default request')
    # Main page
    #return render_template('index.html')
    return "This is icovfy"

@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e

    res = {'code': 500,
           'errorType': 'Internal Server Error',
           'errorMessage': "Something went really wrong!"}
    if debug:
        res['errorMessage'] = e.message if hasattr(e, 'message') else f'{e}'

    return jsonify(res), 500

@app.route('/savewav', methods=['POST'])
def savewav():
  if request.method == 'POST':
    #print(request.files)
    imageData = io.BytesIO(request.get_data())
    data = imageData.getvalue()
    with open('/var/www/html/flaskproject1/covid/myfile.wav', mode='wb') as f:
      f.write(data)

    r = wav2predict('/var/www/html/flaskproject1/covid/myfile.wav')
    if r[0] == 1:
      cl='มีอาการโควิด-19'
      x = 1
    elif r[0] == 0:
      cl = 'ไม่มีอาการโควิด-19'
      x= 0

    pc = r[1]*100
    pci = int(pc)
    pr = str(pci)

    os.remove("/var/www/html/flaskproject1/covid/myfile.wav") 
    print(pr,cl)
  return jsonify(result=cl,prob=pr,x=str(x))



if __name__ == "__main__":
    app.run(port=5000, host='0.0.0.0',debug=True)
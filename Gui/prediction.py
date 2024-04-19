import librosa
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np

################################################

def get_split(path):
  myaudio = AudioSegment.from_file(path , "wav") 
  chunk_length_ms = 1500 # pydub calculates in millisec
  chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

  #Export all of the individual chunks as wav files
  paths =[]

  for i, chunk in enumerate(chunks):
      chunk_name = "chunk{0}.wav".format(i)
      paths.append(chunk_name)
      chunk.export(chunk_name, format="wav")

  return paths

##################################################

def get_mfcc(path):
    y, sr = librosa.load(path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    mfccs = mfccs.T
    return np.resize(mfccs, (65, 40))

##################################################

def get_xqs(paths):
  xqs = []
  for i in range(6): #only using first 6 chunks
    t = get_mfcc(paths[i])
    xqs.append(t)
  return xqs
 
##################################################

def get_probs(model, xqs):
  probs = []
  for xq in xqs:
    xq = xq.reshape((1,65,40))
    p = model.predict(xq, verbose = 0)
    probs.append(p)
  return np.array(probs)
  
##################################################

def predict(model, xqs, num_species):
  probs = get_probs(model, xqs)
  # aggregrate
  s = np.sum(probs, axis = 0)
  if num_species == 2: 
    n = 2
  else:
    n = 3
  labels = np.argsort(s[0])[::-1][:n]
  return list(labels)

##################################################
cls_label = {'Asiankoel' : 0, 'bluejay': 1, 'crow': 2, 'duck': 3, 'goaway': 4, 'lapwing': 5, 'owl': 6,
             'peafowl' : 7, 'sparrow':8, 'woodpeewe':9}

def get_bird_name(label):
    cls_names = []
    for k, v in cls_label.items():
      if v in label:
        cls_names.append(k)
    return cls_names        
##################################################

def demo(model, path, num_species):
    chunk_paths = get_split(path)
    print('Chunking done')
    xqs = get_xqs(chunk_paths)
    print('Getting MFCC done')
    y_pred = predict(model, xqs, num_species)
    print('Prediction done')
    y_pred = get_bird_name(y_pred)
    y_true =  path.name.split('_')[:-1]
    return y_true, y_pred
    



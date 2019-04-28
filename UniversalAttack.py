# %%
import numpy as np
import tensorflow as tf
import argparse
from shutil import copyfile

import scipy.io.wavfile as wav

import struct
import time
import os
import sys
from collections import namedtuple
sys.path.append("DeepSpeech")

# Okay, so this is ugly. We don't want DeepSpeech to crash.
# So we're just going to monkeypatch TF and make some things a no-op.
# Sue me.
tf.load_op_library = lambda x: x
tmp = os.path.exists
os.path.exists = lambda x: True
class Wrapper:
    def __init__(self, d):
        self.d = d
    def __getattr__(self, x):
        return self.d[x]
class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)
    def __getattr__(self, x):
        return self.do_define
    def do_define(self, k, v, *x):
        self.d[k] = v
tf.app.flags = HereBeDragons()
import DeepSpeech
os.path.exists = tmp

# More monkey-patching, to stop the training coordinator setup
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None


from util.text import ctc_label_dense_to_sparse
from tf_logits import get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

# %%
class Args:
    def __init__(self):
        self.input = ["0000.wav"]
        self.target = "example"
		self.out = ["adversarial.wav"]
		self.outprefix = ""
		self.funetune = [""]
		self.lr = 100
		self.iterations = 5000
		self.l2penalty = float('inf')

args = Args()
with tf.Session() as sess:
    finetune = []
    audios = []
    lengths = []
    
    if args.out is None:
        assert args.outprefix is not None
    else:
        assert args.outprefix is None
        assert len(args.input) == len(args.out)
    if args.finetune is not None and len(args.finetune):
        assert len(args.input) == len(args.finetune)
    

    # Load the inputs that we're given
    for i in range(len(args.input)):
        fs, audio = wav.read(args.input[i])
        assert fs == 16000
        assert audio.dtype == np.int16
        print('source dB', 20*np.log10(np.max(np.abs(audio))))
        audios.append(list(audio))
        lengths.append(len(audio))

        if args.finetune is not None:
            finetune.append(list(wav.read(args.finetune[i])[1]))

    maxlen = max(map(len,audios))
    audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
    finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])

    phrase = args.target

    # Loading
    def __init__(self, sess, loss_fn, phrase_length, max_audio_len,
                 learning_rate=10, num_iterations=5000, batch_size=1,
                 mp3=False, l2penalty=float('inf')):
    loss_fn = 'CTC'
    phrase_length = len(phrase)
    max_audio_len = maxlen
    learning_rate = args.lr
    num_iterations = args.iterations
    batch_size = 1
    l2penalty = args.l2penalty
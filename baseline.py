# %%
import numpy as np
import tensorflow as tf
import argparse
import random
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

class Args:
    def __init__(self):
        self.n_input = 128
        self.n_train = 85
        self.n_test = self.n_input - self.n_train
        self.input = ["./audios/%04d.wav"%i for i in range(self.n_input)]
        self.target = "example"
        self.out = ["./audios/final_adv%04d.wav"%i for i in range(self.n_input)]
        self.outprefix = None
        self.finetune = None
        self.lr = 100
        self.iterations = 100
        self.each_train = 21
        self.l2penalty = float('inf')
        self.rescale_original = 20
        self.rescale_noise = -8

args = Args()

print('rescale: original: %d, noise: %d, total: %d', args.rescale_original, args.rescale_noise, args.rescale_original - args.rescale_noise)
sess = tf.Session()
finetune = []
audios = []
audio_lengths = []
# project_eps = 10 ** (75/20.)
project_eps = 10 ** (75/20.) * (0.8 ** args.rescale_noise)
_, read_unipertur = np.array(wav.read("./audios/example.wav"))
read_unipertur = np.array(read_unipertur)

if args.out is None:
    assert args.outprefix is not None
else:
    assert args.outprefix is None
    assert len(args.input) == len(args.out)
if args.finetune is not None and len(args.finetune):
    assert len(args.input) == len(args.finetune)

def cal_dB(x):
    return 20*np.log10(np.max(np.abs(x)))

# Load the inputs that we're given
for i in range(len(args.input)):
    fs, audio = wav.read(args.input[i])
    #print("fs:", fs)
    assert fs == 16000
    assert audio.dtype == np.int16
    print('source dB', 20*np.log10(np.max(np.abs(audio))))
    audios.append(list(audio))
    audio_lengths.append(len(audio))

    if args.finetune is not None:
        finetune.append(list(wav.read(args.finetune[i])[1]))

maxlen = max(map(len,audios))
audios = np.array([x+[0]*(maxlen-len(x)) for x in audios])
print('audios.dtype', audios.dtype)
audios = np.array(audios * (0.8 ** args.rescale_original), dtype=np.int64)
print('audios.dtype', audios.dtype)
finetune = np.array([x+[0]*(maxlen-len(x)) for x in finetune])

phrase = args.target

# Loading
loss_fn = 'CTC'
phrase_length = len(phrase)
max_audio_len = maxlen
learning_rate = args.lr
num_iterations = args.iterations
batch_size = 1
l2penalty = args.l2penalty

# Create all the variables necessary
# they are prefixed with qq_ just so that we know which
# ones are ours so when we restore the session we don't
# clobber them.
tfdelta = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_delta')
tfmask = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_mask')
tfcwmask = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_cwmask')
tforiginal = tf.Variable(np.zeros((batch_size, max_audio_len), dtype=np.float32), name='qq_original')
tflengths = tf.Variable(np.zeros(batch_size, dtype=np.int32), name='qq_lengths')
tftarget_phrase = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.int32), name='qq_phrase')
tftarget_phrase_lengths = tf.Variable(np.zeros((batch_size), dtype=np.int32), name='qq_phrase_lengths')
tfrescale = tf.Variable(np.zeros((batch_size,1), dtype=np.float32), name='qq_phrase_lengths')
tfimportance = tf.Variable(np.zeros((batch_size, phrase_length), dtype=np.float32), name='qq_importance')

unipertur = np.zeros((batch_size, max_audio_len), dtype=np.float32)
unipertur = np.clip(np.random.normal(0, project_eps, size=unipertur.shape), -project_eps, project_eps)
unipertur[0, :read_unipertur.shape[0]] = read_unipertur * 10 ** ((cal_dB(project_eps) - cal_dB(read_unipertur))/20.)
wav.write("./audios/baseline_noise.wav", 16000, np.array(np.clip(np.round(unipertur[0]), -2**15, 2**15-1),dtype=np.int16))

# Initially we bound the l_infty norm by 2000, increase this
# constant if it's not big enough of a distortion for your dataset.
tfapply_delta = tf.clip_by_value(tfdelta, -2000, 2000)*tfrescale

# We set the new input to the model to be the abve delta
# plus a mask, which allows us to enforce that certain
# values remain constant 0 for length padding sequences.
tfnew_input = tfapply_delta*tfmask + tforiginal

# We add a tiny bit of noise to help make sure that we can
# clip our values to 16-bit integers and not break things.
tfnoise = tf.random_normal(tfnew_input.shape, stddev=2)
tfpass_in = tf.clip_by_value(tfnew_input+tfnoise, -2**15, 2**15-1)

# Feed this final value to get the logits.
tflogits = get_logits(tfpass_in, tflengths)

# And finally restore the graph to make the classifier
# actually do something interesting.
saver = tf.train.Saver([x for x in tf.global_variables() if 'qq' not in x.name])
saver.restore(sess, "models/session_dump")

# Choose the loss function we want -- either CTC or CW
if loss_fn == "CTC":
    tftarget = ctc_label_dense_to_sparse(tftarget_phrase, tftarget_phrase_lengths, batch_size)
    
    tfctcloss = tf.nn.ctc_loss(labels=tf.cast(tftarget, tf.int32),
                                inputs=tflogits, sequence_length=tflengths)

    # Slight hack: an infinite l2 penalty means that we don't penalize l2 distortion
    # The code runs faster at a slight cost of distortion, and also leaves one less
    # paramaeter that requires tuning.
    if not np.isinf(l2penalty):
        tfloss = tf.reduce_mean((tfnew_input-tforiginal)**2,axis=1) + l2penalty*tfctcloss
    else:
        tfloss = tfctcloss
    tfexpanded_loss = tf.constant(0)
    
elif loss_fn == "CW":
    raise NotImplemented("The current version of this project does not include the CW loss function implementation.")
else:
    raise

# Set up the Adam optimizer to perform gradient descent for us
start_vars = set(x.name for x in tf.global_variables())
print(start_vars)
optimizer = tf.train.AdamOptimizer(learning_rate)

grad,var = optimizer.compute_gradients(tfloss, [tfdelta])[0]
train = optimizer.apply_gradients([(tf.sign(grad),var)])

end_vars = tf.global_variables()
new_vars = [x for x in end_vars if x.name not in start_vars]

sess.run(tf.variables_initializer(new_vars+[tfdelta]))

# Decoder from the logits, to see how we're doing
tfdecoded, _ = tf.nn.ctc_beam_search_decoder(tflogits, tflengths, merge_repeated=False, beam_width=100)

# %%
audio = audios[0]
audio_lengths = audio_lengths
# target = [[toks.index(x) for x in phrase]]*len(audios)
target = [[toks.index(x) for x in phrase]]
finetune = finetune

print("audio_lengths:", audio_lengths)

# Here we'll keep track of the best solution we've found so far
final_deltas = [None]*batch_size

if finetune is not None and len(finetune) > 0:
    sess.run(tfdelta.assign(finetune-audio))

print('rescale: original: %d, noise: %d, total: %d', args.rescale_original, args.rescale_noise, cal_dB(unipertur) - cal_dB(audios[0]))

# We'll make a bunch of iterations of gradient descent here
now = time.time()
n_fooled_test = 0
n_fooled_train = 0
audio_indices = list(range(len(audios)))
for idx_audio in audio_indices:
    print("=" * 40)
    print("Training for audio: %d"%idx_audio)
    audio = audios[idx_audio]
    print("unipertur L2:", np.mean(np.square(unipertur)))
    print("unipertur dB:", cal_dB(unipertur))
    original = np.array(audio) + unipertur
    sess.run(tforiginal.assign(original))
    sess.run(tf.variables_initializer([tfdelta]))
    # sess.run(tforiginal.assign(np.array(audio)))
    sess.run(tflengths.assign((np.array([audio_lengths[idx_audio]])-1)//320))
    sess.run(tfmask.assign(np.array([[1 if i < l else 0 for i in range(max_audio_len)] for l in [audio_lengths[idx_audio]]])))
    sess.run(tfcwmask.assign(np.array([[1 if i < l else 0 for i in range(phrase_length)] for l in [(np.array(audio_lengths[idx_audio])-1)//320]])))
    sess.run(tftarget_phrase_lengths.assign(np.array([len(x) for x in target])))
    sess.run(tftarget_phrase.assign(np.array([list(t)+[0]*(phrase_length-len(t)) for t in target])))
    c = np.ones((batch_size, phrase_length))
    sess.run(tfimportance.assign(c))
    sess.run(tfrescale.assign(np.ones((batch_size,1))))

    new, delta, r_out, r_logits = sess.run((tfnew_input, tfdelta, tfdecoded, tflogits))
    lst = [(r_out, r_logits)]

    for out, logits in lst:
        chars = out[0].values

        res = np.zeros(out[0].dense_shape)+len(toks)-1
    
        for ii in range(len(out[0].values)):
            x,y = out[0].indices[ii]
            res[x,y] = out[0].values[ii]

        # Here we print the strings that are recognized.
        res = ["".join(toks[int(x)] for x in y).replace("-","") for y in res] 
        if res[0] == args.target:
            if idx_audio < args.n_train:
                n_fooled_train += 1
            else:
                n_fooled_test += 1
        print("\n".join(res)) 
        # And here we print the argmax of the alignment.
        res2 = np.argmax(logits,axis=2).T
        res2 = ["".join(toks[int(x)] for x in y[:(l-1)//320]) for y,l in zip(res2,audio_lengths)]
        print("\n".join(res2))

    feed_dict = {}
    # Minimize delta
    # Actually do the optimization ste
    d, el, cl, l, logits, new_input, _ = sess.run((tfdelta, tfexpanded_loss,
                                                    tfctcloss, tfloss,
                                                    tflogits, tfnew_input,
                                                    train),
                                                    feed_dict)
    # Report progress
    cur_loss = np.mean(cl)
    print("%.3f"%np.mean(cl), "\t", "\t".join("%.3f"%x for x in cl))
    if idx_audio >= args.n_train:
        print("It was TEST")
    if idx_audio == 0:
        wav.write("./audios/baseline_adv%04d.wav"%idx_audio, 16000, np.array(np.clip(np.round(original[0]), -2**15, 2**15-1),dtype=np.int16))


fool_rate_train = float(n_fooled_train) / args.n_train
fool_rate_test = float(n_fooled_test) / args.n_test
print("training fooling rate: %f, testing fooling rate: %f, project_eps: %f"%(fool_rate_train, fool_rate_test, project_eps))
print('rescale: original: %d, noise: %d, total: %d', args.rescale_original, args.rescale_noise, cal_dB(unipertur) - cal_dB(audios[0]))

sess.close()

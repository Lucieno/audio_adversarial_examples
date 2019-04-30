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

# %%
def projection(v, eps, p):
    """
    Project the values in `v` on the L_p norm ball of size `eps`.

    :param v: Array of perturbations to clip.
    :type v: `np.ndarray`
    :param eps: Maximum norm allowed.
    :type eps: `float`
    :param p: L_p norm to use for clipping. Only 1, 2 and `np.Inf` supported for now.
    :type p: `int`
    :return: Values of `v` after projection.
    :rtype: `np.ndarray`
    """
    # Pick a small scalar to avoid division by 0
    tol = 10e-8
    v_ = v.reshape((v.shape[0], -1))

    if p == 2:
        v_ = v_ * np.expand_dims(np.minimum(1., eps / (np.linalg.norm(v_, axis=1) + tol)), axis=1)
    elif p == 1:
        v_ = v_ * np.expand_dims(np.minimum(1., eps / (np.linalg.norm(v_, axis=1, ord=1) + tol)), axis=1)
    elif p == np.inf:
        v_ = np.sign(v_) * np.minimum(abs(v_), eps)
    else:
        raise NotImplementedError('Values of `p` different from 1, 2 and `np.inf` are currently not supported.')

    v = v_.reshape(v.shape)
    return v


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

args = Args()

sess = tf.Session()
finetune = []
audios = []
audio_lengths = []
project_eps = 10 ** (75/20.)

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

# We'll make a bunch of iterations of gradient descent here
now = time.time()
MAX = num_iterations
for epoch in range(MAX):
    print("Start of epcoh: %d"%epoch)
    n_fooled_train = 0
    n_fooled_test = 0
    shuffled_indices = list(range(len(audios)))
    random.shuffle(shuffled_indices)
    for idx_audio in shuffled_indices:
        print("=" * 40)
        print("Training for audio: %d"%idx_audio)
        audio = audios[idx_audio]
        print("unipertur L2:", np.mean(np.square(unipertur)))
        print("unipertur dB:", cal_dB(unipertur))
        sess.run(tforiginal.assign(np.array(audio) + unipertur))
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

        for i in range(args.each_train):
            iteration = i
            now = time.time()

            # Print out some debug information every 10 iterations.
            if i%10 == 0:
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
                    if i == 0 and res[0] == args.target:
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
            if i % 10 == 0:
                print("%.3f"%np.mean(cl), "\t", "\t".join("%.3f"%x for x in cl))
            if idx_audio >= args.n_train:
                print("It was TEST")
                break

            logits = np.argmax(logits,axis=2).T
            for ii in range(batch_size):
                # Every 100 iterations, check if we've succeeded
                # if we have (or if it's the final epoch) then we
                # should record our progress and decrease the
                # rescale constant.
                if (loss_fn == "CTC" and i == 0 and res[ii] == "".join([toks[x] for x in target[ii]])) \
                    or (i == MAX-1 and final_deltas[ii] is None):
                    # Get the current constant
                    rescale = sess.run(tfrescale)
                    if rescale[ii]*2000 > np.max(np.abs(d)):
                        # If we're already below the threshold, then
                        # just reduce the threshold to the current
                        # point and save some time.
                        print("It's way over", np.max(np.abs(d[ii]))/2000.0)
                        rescale[ii] = np.max(np.abs(d[ii]))/2000.0

                    # Otherwise reduce it by some constant. The closer
                    # this number is to 1, the better quality the result
                    # will be. The smaller, the quicker we'll converge
                    # on a result but it will be lower quality.
                    rescale[ii] *= .8

                    # Adjust the best solution found so far
                    final_deltas[ii] = new_input[ii]

                    print("Worked i=%d ctcloss=%f bound=%f"%(ii,cl[ii], 2000*rescale[ii][0]))
                    #print('delta',np.max(np.abs(new_input[ii]-audio[ii])))
                    sess.run(tfrescale.assign(rescale))

                    # Just for debugging, save the adversarial example
                    # to /tmp so we can see it if we want
                    wav.write("./audios/adv%04d.wav"%idx_audio, 16000,
                                np.array(np.clip(np.round(new_input[ii]), -2**15, 2**15-1),dtype=np.int16))

            if cur_loss < 1:
                break

        if idx_audio < args.n_train:
            print("delta L2:", np.mean(np.square(d)))
            print("delta dB:", cal_dB(d))
            unipertur += d
            unipertur = projection(unipertur, project_eps, np.inf)
    fool_rate_train = float(n_fooled_train) / args.n_train
    fool_rate_test = float(n_fooled_test) / args.n_test
    print("End of epcoh: %d, training fooling rate: %f, testing fooling rate: %f, project_eps: %f"%(epoch, fool_rate_train, fool_rate_test, project_eps))
    if fool_rate_test > 0.75:
        # project_eps /= float(10 ** (5/20))
        project_eps *= 0.8
        wav.write("./audios/unipertur.wav", 16000, np.array(np.clip(np.round(unipertur[0]), -2**15, 2**15-1),dtype=np.int16))

sess.close()

# Claim
The code is based on the implementation of Nicholas Carlini and David Wagner's "Audio Adversarial Examples: Targeted Attacks on Speech-to-Text"

# How to run

1. Install the dependencies
```
pip3 install --user numpy scipy tensorflow-gpu==1.8.0 pandas python_speech_features
```

2. Clone the Mozilla DeepSpeech repository into a folder called DeepSpeech:
```
git clone https://github.com/mozilla/DeepSpeech.git
```

2b. Checkout the correct version of the code:

```
(cd DeepSpeech; git checkout tags/v0.1.1)
```

3. Download the DeepSpeech model
```
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz
tar -xzf deepspeech-0.1.0-models.tar.gz
```

4. Verify that you have a file models/output_graph.pb, its MD5 sum should be
```
08a9e6e8dc450007a0df0a37956bc795.
```

5. Convert the .pb to a TensorFlow checkpoint file
```
python3 make_checkpoint.py
```

6. Generate adversarial examples
```
python UniversalAttack.py
```

# WARNING

THE CODE TO HOOK INTO DEEPSPEECH IS UGLY. This means I require a
very specific version of DeepSpeech (0.1.1) and TensorFlow (1.8.0) using
python 3.5. I can't promise it won't set your computer on fire if you use
any other versioning setup. (In particular, it WILL NOT work with
DeepSpeech 0.2.0+, and WILL NOT work with TensorFlow 1.10+.)

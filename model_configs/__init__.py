"""
Problems:
---------
1. CNN:
    1.1. small scale:
        filter length (kernel size, dilation), downsampling (stride),
        these mainly depend on frequency bands of regions of interest,
        like QRS complex, P wave, T wave, or even intervals like qt interval,
        and also finer structures like notches on QRS complexes
    1.2. large scale:
        network depth, and block structures (e.g. ResNet v.s. VGG);
        upsampling?
2. RNN:
    2.1. choice between LSTM and attention
    2.2. use the last state for the last classifying layer or use the whole sequence

Frequency bands:
----------------
QRS complex: 8 - 25 Hz
P wave: 5 - 20 Hz
T wave: 2.5 - 7 Hz
notch: 30 - 50 Hz (?)
NOTE that different literatures have different conlusions,
the above takes into considerations of many literatures

TODO: experiment on frequency analysis using the whole CINC2020 training data,
or using the (annotations of) ludb (Lobachevsky University Electrocardiography Database)

References:
-----------
[1] Lin, Chia-Hung. "Frequency-domain features for ECG beat discrimination using grey relational analysis-based classifier." Computers & Mathematics with Applications 55.4 (2008): 680-690.
[2] Elgendi, Mohamed, Mirjam Jonkman, and Friso De Boer. "Frequency Bands Effects on QRS Detection." BIOSIGNALS 2003 (2010): 2002.
[3] Tereshchenko, Larisa G., and Mark E. Josephson. "Frequency content and characteristics of ventricular conduction." Journal of electrocardiology 48.6 (2015): 933-937.
"""

from .ati_cnn import *


__all__ = [s for s in dir() if not s.startswith('_')]

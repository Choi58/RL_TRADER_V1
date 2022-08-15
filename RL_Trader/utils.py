import time
import datetime
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm
import collections
import os
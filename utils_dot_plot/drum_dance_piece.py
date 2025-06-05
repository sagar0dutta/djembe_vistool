import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils_subdivision.gen_distribution_subplot import analyze_single_type
from utils_dot_plot.kinematic_dot_plot import *
from utils_dot_plot.drum_merged import *

def get_subdiv_color(subdiv):
    if subdiv in [1, 4, 7, 10]:
        return 'black'
    elif subdiv in [2, 5, 8, 11]:
        return 'green'
    elif subdiv in [3, 6, 9, 12]:
        return 'red'
    return 'gray'



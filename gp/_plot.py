import mplhep
mplhep.style.use('ROOT')

import matplotlib.pyplot as plt

def label(ax = None, stage = 'Internal', dataset = '2016 6.5%'):
    return mplhep.label.exp_label('HPS', llabel=stage, rlabel=dataset, ax=ax)

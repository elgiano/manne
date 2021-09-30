from os import listdir
from os.path import isdir, join, splitext
import numpy as np
from manne_model import ManneModel
import pandas as pd

model_paths = [join('models', p)
               for p in listdir('models') if isdir(join('models', p))]

num_models = len(model_paths)

print(f'Benchmarking {num_models} models')

models = [ManneModel(p) for p in model_paths]

model_data = None
for m in models:
    b = m.benchmark(verbose=False)
    b = np.array(b).reshape(-1)
    if model_data is None:
        model_data = np.array([b])
    else:
        model_data = np.vstack((model_data, b))

model_names = [p for p in listdir('models') if isdir(join('models', p))]
info = []
for name in model_paths:
    skip = name.split('_')[1] == 'skip'
    latent = int(name.split('_')[2][1:])
    aug = ''.join([c[0].upper()
                   for c in splitext(name.split('_')[-1])[0].split("+")])
    info += [[latent, skip, aug]]
h = pd.DataFrame(info, columns=['latent', 'skip', 'aug'])

d = pd.DataFrame(model_data, columns=['enc1', 'enc1v', 'dec1', 'dec1v', 'ae1', 'ae1v', 'enc10', 'enc10v',
                                      'dec10', 'dec10v', 'ae10', 'ae10v', 'enc100', 'enc100v', 'dec100', 'dec100v', 'ae100', 'ae100v'])
d = pd.concat((h, d), axis=1)
d.to_csv('models_benchmark.csv')

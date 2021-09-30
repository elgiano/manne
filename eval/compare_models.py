from os import listdir
from os.path import isdir, join, splitext
from manne_model import ManneModel
import pandas as pd

model_paths = [join('models', p)
               for p in listdir('models') if isdir(join('models', p))]

num_models = len(model_paths)
print(f'Comparing {num_models} models')
models = [ManneModel(p) for p in model_paths]
model_data = []
for m in models:
    aug = splitext(m.name.split('_')[-1])[0]
    vl = m.history['val_loss'][-1]
    e = len(m.history['val_loss'])
    t = m.history.get('train_dur', 0)
    model_data += [[m.latent_size, m.skip, aug, vl, e, t]]

d = pd.DataFrame(model_data, columns=[
                 'latent', 'skip', 'augmentations', 'val_loss', 'epochs', 'train dur'])
d = d.sort_values('val_loss')
d.to_csv('compare_model_loss.csv')

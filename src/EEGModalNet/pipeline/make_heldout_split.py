"""Create the held-out participant split for LEMON (stratified by age group x sex).

Run from the repository root:  python -m src.EEGModalNet.pipeline.make_heldout_split
Writes splits/lemon_heldout40.json with the held-out and training participant ids.
"""
import json
import os

import pandas as pd
import xarray as xr
from sklearn.model_selection import StratifiedShuffleSplit

N_HELDOUT, SEED = 40, 0
DATA = 'data/LEMON_DATA/EC_ch-8_sf-128.nc5'
DEMOG = 'data/LEMON_DATA/Demographics.csv'
OUT = 'splits/lemon_heldout40.json'

if __name__ == '__main__':
    subjects = [str(s) for s in xr.open_dataarray(DATA, engine='h5netcdf').subject.values]
    demog = pd.read_csv(DEMOG).set_index('ID').loc[subjects]
    sex = demog['Gender_ 1=female_2=male'].map({1: 'F', 2: 'M'})
    age_group = demog['Age'].astype(str).str[:2].astype(int).map(lambda a: 'young' if a < 45 else 'old')
    strata = (age_group + '_' + sex).to_numpy()

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=N_HELDOUT, random_state=SEED)
    train_idx, held_idx = next(splitter.split(subjects, strata))
    heldout = sorted(subjects[i] for i in held_idx)
    train = sorted(subjects[i] for i in train_idx)

    counts = pd.crosstab(pd.Series(strata, name='stratum'),
                         pd.Series(['heldout' if s in set(heldout) else 'train' for s in subjects], name='set'))
    print(counts)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'description': 'LEMON held-out participants, stratified by age group x sex',
                   'seed': SEED, 'n_total': len(subjects), 'n_heldout': len(heldout),
                   'strata_counts': counts.to_dict(), 'heldout': heldout, 'train': train}, f, indent=1)
    print(f'wrote {OUT}: {len(heldout)} held out, {len(train)} for training')

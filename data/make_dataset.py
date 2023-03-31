import pandas as pd
import glob
import logging
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap


@timing
def make_dataset_from_curves(curve_list, name_of_dataset='dataset'):
    """
    For each folder in data/data, read pickle files and concat into one dataframe
    """
    big_df = pd.DataFrame()
    for i, curve in enumerate(curve_list):
        df_curves_list = []
        for file in glob.glob(f"data/data/{curve}/*"):
            #df = pd.concat([df, pd.read_pickle(file)], axis=1)
            tmp_df = pd.read_pickle(file)
            # check if curve contains numeric values
            if tmp_df.dtypes[0] == 'float64':
                df_curves_list.append(tmp_df)

            # prepend curve name to column names
        if len(df_curves_list) > 0:
            df = pd.concat(df_curves_list, axis=1)
            df.columns = [curve + '_' + col for col in df.columns]
            # merge with other curves on index (date)
            big_df = pd.merge(big_df, df, how='outer', left_index=True, right_index=True)
        logging.info(f"[{i + 1}/{len(curve_list)}] {curve} done.")

    # save to pickle
    big_df.to_pickle(f"C:/Users/frede/PycharmProjects/Masters/data/datasets/{name_of_dataset}.pkl")

    return big_df


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    , handlers=[logging.FileHandler('make_dataset.log'), logging.StreamHandler()])

# try with all curves
some_curves = ['AcceptedAggregatedOffers_17.1.D',
               'ActivatedBalancingEnergy_17.1.E',
               'ActualGenerationOutputPerGenerationUnit_16.1.A',
               'ActualTotalLoad_6.1.A']

all_curves = [file[file.rfind('\\') + 1:] for file in glob.glob('data/data/*')]
tmp_df = make_dataset_from_curves(all_curves, name_of_dataset='all_numeric_curves')

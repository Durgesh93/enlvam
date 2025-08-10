import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from typing import Optional,Any,List,Dict,Union
from dotenv import find_dotenv

envpath = find_dotenv()
sys.path.insert(0,os.path.dirname(envpath))

import echotools

def parse_args():
    parser = argparse.ArgumentParser(description="Process gev2_private data paths and settings.")
    
    parser.add_argument('--input_dir', 
                             type=str, 
                             default='dirs/data_storage/rawfiles/gev2_private/movie_data',
                            help='Path to the input directory containing raw files.'
                        )

    parser.add_argument('--output_dir', 
                              type=str, 
                              default='dirs/data_storage',
                              help='Path to the output directory.'
                        )

    parser.add_argument('--metadata_path', 
                                type=str, 
                                default='dirs/data_storage/rawfiles/gev2_private/movie_data/movie_single_cycle_dataset.xlsx',
                                help='Path to the metadata CSV file.'
                        )

    return parser.parse_args()



class PlaxMovieDataset:
    """A class to create a dataset of movies and paired coordinates."""

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.movies: List[np.ndarray] = []
        self.coords: List = []
        self.indices: List[int] = []
        self.index: int = 0
        self.movie_counter: int = 0

    def load(self, output_dir: str, indices: List[int]) -> None:
        movies: np.ndarray = np.load(os.path.join(output_dir, self.name, "movies.npy"))
        total_frames: int = movies.shape[0]
        self.movies = [movies[ind0:ind1] for ind0, ind1 in zip(indices, indices[1:] + [total_frames])]
        self.index = total_frames

    def __repr__(self) -> str:
        return f"PlaxMovieDataset for {self.name} with {len(self)} images and coords"

    def __len__(self) -> int:
        return len(self.movies)


def load_coords_from_dset_str(phase: str, s: str) -> Union[np.ndarray, None]:
    """
    Parses a string of coordinates and returns a formatted NumPy array
    based on the specified cardiac phase.

    Parameters:
        phase (str): Cardiac phase, either 's' (systole) or 'd' (diastole).
        s (str): String representation of coordinates.

    Returns:
        np.ndarray: Transformed coordinates for the specified phase.
    """
    s = s.replace("[", "").replace("]", "").replace("\n", "")
    coords = [c for c in s.split() if len(c) > 1]
    coords = np.array(coords).astype(float).reshape((12, 2))
    coords = np.vstack([coords[::2], coords[1::2]]).reshape((2, 3, 2, 2))
    coords_ed = echotools.six_to_four_format(coords[0])
    coords_es = echotools.six_to_four_format(coords[1])

    if phase == 's':
        return coords_es
    elif phase == 'd':
        return coords_ed
    return None

def check_invalid_distances(c: np.ndarray) -> np.bool_:
    """
    Checks if all coordinate pairs have non-zero distances.

    Parameters:
        c (np.ndarray): Array of coordinates expected to be shaped for 4 points.

    Returns:
        bool: True if any pair has zero distance, False otherwise.
    """
    pair_idx = [0, 1, 1, 2, 2, 3]
    coord_pair = c[pair_idx].reshape(len(pair_idx) // 2, 2, 2).astype('int')  # 3x2x2
    fc_c = coord_pair[:, 0]  # 3x2
    sc_c = coord_pair[:, 1]  # 3x2
    d = np.linalg.norm(fc_c - sc_c, axis=-1)  # shape: (3,)
    return ~np.all(d)

def gev2moviemetadata(movie_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and reshapes a movie dataset by removing entries with invalid distances
    and converting frame annotations into a long format.

    Parameters:
        movie_dataset (pd.DataFrame): Input DataFrame containing movie metadata and coordinates.

    Returns:
        pd.DataFrame: Cleaned and reshaped DataFrame ready for analysis.
    """
    ids: list[int] = []

    for i, row in movie_dataset.iterrows():
        coord_es: np.ndarray = load_coords_from_dset_str('s', row.coords)
        vb = echotools.VBScanLine(coord_es)
        mcoord_es: np.ndarray = vb.B2AMM_coords(coord_es)

        coord_ed: np.ndarray = load_coords_from_dset_str('d', row.coords)
        vb = echotools.VBScanLine(coord_ed)
        mcoord_ed: np.ndarray = vb.B2AMM_coords(coord_ed)

        if check_invalid_distances(mcoord_es) or check_invalid_distances(mcoord_ed):
            print(f"Skipping Movie {i:04d} due to invalid distances in Motion mode", flush=True)
            ids.append(i)

    movie_dataset = movie_dataset.drop(ids, axis=0).reset_index(drop=True)
    movie_dataset['movieid'] = movie_dataset.index

    id_vars = set(movie_dataset.columns) - {'ED_frame', 'ES_frame'}
    movie_dataset = pd.melt(
        movie_dataset,
        id_vars=id_vars,
        value_vars=['ED_frame', 'ES_frame'],
        var_name='phase',
        value_name='aidx'
    )

    movie_dataset.loc[movie_dataset['phase'] == 'ES_frame', 'phase'] = 's'
    movie_dataset.loc[movie_dataset['phase'] == 'ED_frame', 'phase'] = 'd'
    movie_dataset['dataid'] = movie_dataset.index

    return movie_dataset


def movie_data(args: Any)->None:
    input_dir  = args.input_dir
    output_dir = args.output_dir
    data_dict = {
        'X_bmovie':   [],
        'X_bcoords': [],
        'y_movieid':  [],
        'y_dataid':  [],
        'y_frames':  [],
        'y_height':  [],
        'y_width': [],
        'y_keyframe_idx':[],
        'y_phase':[],
        'y_annotime':[],
        'y_split': [],
        'y_series_uid': [],
        'y_labeler': [],
        'y_pix2cm_y':[],
        'y_pix2cm_x':[],
        'y_hospital':[]
    }

    movie_dataset = pd.read_excel(os.path.join(input_dir, 'movie_single_cycle_dataset.xlsx'), usecols='B:W')
    directories = set(movie_dataset.directory)

    # Load datasets
    dsets = {}
    for d in directories:
        dsets[d] = PlaxMovieDataset(d)
        indices_d = list(movie_dataset[movie_dataset['directory'] == d]['dataset_index'])
        dsets[d].load(input_dir, indices_d)
    dsets = {d: dset for d, dset in dsets.items() if len(dset) > 0}

    # Process the movie dataset
    movie_dataset = gev2moviemetadata(movie_dataset)
    print('Extracting train movie clip ES and ED', flush=True)

    # Populate dictionaries
    for i, row in tqdm(movie_dataset.iterrows(),total=len(movie_dataset)):
        hospital,labeler,*time = row.directory.split('_')
        time                   = '.'.join(time)
        movie                  = dsets[row.directory].movies[row.dataset_counter]
        frames,height,width    = movie.shape 
        coord                  = load_coords_from_dset_str(row.phase, row.coords)

        data_dict['X_bmovie'].append(movie)
        data_dict['X_bcoords'].append(coord)
        data_dict['y_movieid'].append(row.movieid)
        data_dict['y_frames'].append(frames)
        data_dict['y_height'].append(height)
        data_dict['y_width'].append(width)
        data_dict['y_keyframe_idx'].append(row.aidx)
        data_dict['y_phase'].append(row.phase)
        data_dict['y_pix2cm_y'].append(row['height_cm']/height)
        data_dict['y_pix2cm_x'].append(row['width_cm']/width)
        data_dict['y_annotime'].append(time)
        data_dict['y_hospital'].append(hospital)
        data_dict['y_split'].append(row.dataset_split)
        data_dict['y_series_uid'].append(row.series_uid)
        data_dict['y_labeler'].append(row.labeler)
        data_dict['y_dataid'].append(i)

    _, indices             = np.unique(data_dict['y_movieid'], return_index=True)
    movies = []
    movie_ids = []
    for ind in indices:
        movie = data_dict['X_bmovie'][ind]
        movies.append(movie)
        movie_ids.append(data_dict['y_movieid'][ind])

    data_dict['X_bmovie']  = movies
    data_dict['X_preid']   = movie_ids

    h5_file     = 'gev2_private.h5'
    h5_filepath = os.path.join(args.output_dir, h5_file)
    h5_container = echotools.H5Container(h5_filepath)
    try:
        h5_container.store_as_chunked_dataset(data_dict,target_keys=[
            'y_movieid',
            'y_frames',
            'y_height',
            'y_width',
            'y_keyframe_idx',
            'y_phase',
            'y_pix2cm_y',
            'y_pix2cm_x',
            'y_annotime',
            'y_hospital',
            'y_split',
            'y_series_uid',
            'y_labeler',
            'y_dataid',
            'X_bcoords',                                    
        ])

        h5_container.store_as_dataset(data_dict,target_keys=['X_bmovie'])
    finally:
        # closing h5 container
        h5_container.close()

    
if __name__ == '__main__':
    print('GeV2 Private Data Script', flush=True)
    print('++++++++++++++++++++++++', flush=True)
    args = parse_args()
    movie_data(args)
    print('Script finished', flush=True)
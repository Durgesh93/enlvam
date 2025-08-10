import sys
import os
import cv2
import argparse
import numpy as np
import pandas as pd
import concurrent
from tqdm import tqdm
import albumentations as A
from typing import Optional,Any,List,Dict
from dotenv import find_dotenv

envpath = find_dotenv()
sys.path.insert(0,os.path.dirname(envpath))

from data.prepare_data import constants
import echotools

def parse_args():
    parser = argparse.ArgumentParser(description="Process echonetlvh data paths and settings.")
    
    parser.add_argument('--input_dir', 
                             type=str, 
                             default='dirs/data_storage/rawfiles/echonetlvh',
                            help='Path to the input directory containing raw files.'
                        )

    parser.add_argument('--output_dir', 
                              type=str, 
                              default='dirs/data_storage',
                              help='Path to the output directory.'
                        )

    parser.add_argument('--metadata_path', 
                                type=str, 
                                default='dirs/data_storage/rawfiles/echonetlvh/MeasurementsList.csv',
                                help='Path to the metadata CSV file.'
                        )

    parser.add_argument('--num_workers', 
                               type=int, 
                               default=8,
                               help='Number of worker threads to use.'
                        )

    parser.add_argument('--num_annot_structures', 
                                type=str, 
                                default='3',
                                help='Number of annotated structures.'
                        )

    return parser.parse_args()


def get_avi_path(filename: str, input_dir: str) -> Optional[str]:
    """
    Appends '.avi' to the filename and searches for the file
    in Batch1 to Batch4 directories under the input directory.

    Parameters:
        filename (str): Base name of the file (without extension).
        input_dir (str): Root directory containing Batch folders.

    Returns:
        Optional[str]: Full path to the .avi file if found, else None.
    """
    filename += ".avi"
    for i in range(1, constants.NUM_BATCHES_IN_ECHONET_LVH):
        file_path = os.path.join(input_dir, f'Batch{i}', filename)
        if os.path.exists(file_path):
            return file_path
    return None

def create_df(df: pd.DataFrame, args: Any) -> pd.DataFrame:
    """
    Processes a DataFrame containing echocardiographic landmark data.

    Parameters:
        df (pd.DataFrame): Raw input DataFrame with landmark and metadata.
        args (Any): Arguments object with at least an 'input_dir' attribute.

    Returns:
        pd.DataFrame: Processed and reshaped DataFrame.
    """

    def check_coords(row: pd.Series) -> pd.Series:
        """
        Checks and interpolates missing coordinates for IVS, LVID, and LVPW.

        Parameters:
            row (pd.Series): A row from the DataFrame.

        Returns:
            pd.Series: Updated row with combined coordinates and flags.
        """
        ivs_f = not np.array_equal(row['coords.IVS'], np.zeros((2, 2)))
        lvid_f = not np.array_equal(row['coords.LVID'], np.zeros((2, 2)))
        lvpw_f = not np.array_equal(row['coords.LVPW'], np.zeros((2, 2)))
        interp_coords = False

        if not lvid_f and ivs_f and lvpw_f:
            row['coords.LVID'][0] = row['coords.IVS'][1]
            row['coords.LVID'][1] = row['coords.LVPW'][0]
            interp_coords = True
            lvid_f = True

        if ivs_f and lvid_f and not np.array_equal(row['coords.IVS'][1], row['coords.LVID'][0]):
            mid = 0.5 * (row['coords.IVS'][1] + row['coords.LVID'][0])
            row['coords.IVS'][1] = mid
            row['coords.LVID'][0] = mid
            interp_coords = True
        if lvid_f and lvpw_f and not np.array_equal(row['coords.LVID'][1], row['coords.LVPW'][0]):
            mid = 0.5 * (row['coords.LVID'][1] + row['coords.LVPW'][0])
            row['coords.LVID'][1] = mid
            row['coords.LVPW'][0] = mid
            interp_coords = True

        row['coords'] = np.concatenate([
            row['coords.IVS'],
            row['coords.LVID'],
            row['coords.LVPW']
        ])

        row = row.drop(['coords.IVS', 'coords.LVID', 'coords.LVPW'])

        row['IVSm'] = ivs_f
        row['LVIDm'] = lvid_f
        row['LVPWm'] = lvpw_f
        row['interp'] = interp_coords
        row['count'] = sum([ivs_f, lvid_f, lvpw_f])

        return row

    # Compute file paths
    df["file_path"] = df["HashedFileName"].apply(
                                                lambda filename: get_avi_path(
                                                                    filename,
                                                                    args.input_dir
                                                                )
                                                )
    
    # Extract phase and clean Calc column
    df["phase"] = df["Calc"].apply(lambda x: x[-1])
    df["Calc"] = df["Calc"].apply(lambda x: x[:-1])

    # Merge train and val splits
    df.loc[df["split"].isin(["train", "val"]), "split"] = "trainval"

    # Convert X/Y coordinates into 2x2 arrays
    df["coords"] = df[["X1", "X2", "Y1", "Y2"]].apply(
        lambda row: np.array([[row["Y2"], row["X2"]], [row["Y1"], row["X1"]]]),
        axis=1
    )
    df = df.drop(columns=["X1", "X2", "Y1", "Y2"])

    # Pivot table to separate IVS, LVID, and LVPW
    index_cols = [
        "HashedFileName", "Frame", "Frames", "FPS",
        "Width", "Height", "split", "file_path", "phase"
    ]
    df = df.pivot(index=index_cols, columns=["Calc"], values=["coords", "Frame"]).reset_index()

    # Flatten multi-level column names
    df.columns = [
        ".".join([col[0], col[1]]) if col[1] else col[0]
        for col in df.columns
    ]

    # Ensure all coordinate columns are valid arrays
    coord_columns = ["coords.IVS", "coords.LVID", "coords.LVPW"]
    df[coord_columns] = df[coord_columns].map(
        lambda x: x if isinstance(x, np.ndarray) else np.zeros((2, 2))
    )

    # Apply coordinate validation and interpolation
    df = df.apply(check_coords, axis=1)

    # Rename hashed filename to movie ID
    df = df.rename(columns={"HashedFileName": "movieid"})

    return df

resize_transform = A.Compose(
                            [
                                A.ToGray(num_output_channels=1, p=1.0),
                                A.Resize(height=768, width=1024, p=1), 
                                A.CenterCrop(width=768, height=768),  
                                A.LongestMaxSize(max_size=128, interpolation=1, p=1.0),
                            ], keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
                            )


def resize_transform_movies(movie_frames: List[np.ndarray]) -> np.ndarray:
    """
    Applies a resize transformation to each frame in a sequence of movie frames.

    Args:
        movie_frames (List[np.ndarray]): A list of image frames (NumPy arrays), typically representing a movie or video sequence.

    Returns:
        np.ndarray: A NumPy array containing the transformed frames stacked along the first axis.
    """
    movie: List[np.ndarray] = []

    for frame in movie_frames:
        data_dict = {
            'image': frame,
            'keypoints': np.zeros((2, 2), dtype=float).tolist()  # Dummy keypoints for compatibility
        }

        tframe = resize_transform(**data_dict)['image']
        movie.append(tframe)

    return np.stack(movie)

def read_video(video_file_path: str) -> Optional[np.ndarray]:
    """
    Reads a video file and returns its frames as a NumPy array in RGB format.

    Parameters:
        video_file_path (str): Path to the video file.

    Returns:
        Optional[np.ndarray]: 4D NumPy array of shape (num_frames, height, width, 3)
                              containing RGB frames, or None if the video cannot be read.
    """
    if not os.path.isfile(video_file_path):
        print(f"Error: File '{video_file_path}' does not exist.")
        return None

    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    if not str(video_file_path).lower().endswith(valid_extensions):
        print(f"Error: File '{video_file_path}' is not a supported video format.")
        return None

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_file_path}'.")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"Error: No frames were read from '{video_file_path}'.")
        return None

    frames_np = np.array(frames, dtype=np.uint8)
    return frames_np

def generate_data(movieid: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Generate structured data for a single movie ID from the DataFrame.
    Applies resizing and coordinate transformation to video frames and annotations.

    Parameters:
        movieid (str): Unique identifier for the video.
        df (pd.DataFrame): DataFrame containing annotations and metadata.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing processed video and annotation data,
                                  or None if the video or data is missing.
    """
    row_data: Dict[str, Any] = {
        'X_bmovie': None,
        'X_preid': None,
        'X_bcoords': None,
        'y_movieid': [],
        'y_frames': [],
        'y_width': [],
        'y_height': [],
        'y_keyframe_idx': [],
        'y_phase': [],
        'y_mask_ivs': [],
        'y_mask_lvid': [],
        'y_mask_lvpw': [],
        'y_split': [],
        'y_pix2cm_x': [],
        'y_pix2cm_y': [],
    }

    dfmovie = df[df['movieid'] == movieid]
    if dfmovie.empty:
        return None

    filepath = dfmovie['file_path'].iloc[0]
    movie_frames = read_video(filepath)
    if movie_frames is None:
        return None

    movie = resize_transform_movies(movie_frames)
    frames, height, width = movie.shape
    row_data['X_bmovie'] = movie
    row_data['X_preid'] = movieid

    tcoords: Optional[np.ndarray] = None
    for _, row in dfmovie.iterrows():
        row_data['y_keyframe_idx'].append(row.Frame)
        row_data['y_split'].append(row.split)
        row_data['y_phase'].append(row.phase)
        row_data['y_width'].append(width)
        row_data['y_height'].append(height)
        row_data['y_frames'].append(frames)
        row_data['y_movieid'].append(movieid)

        data_dict = {
            'image': np.zeros((int(row.Height), int(row.Width), 3), dtype=np.uint8),
            'keypoints': row.coords.astype(float).tolist()
        }
        resize_transform(**data_dict)
        tcoords = np.array(data_dict['keypoints'], dtype=float).reshape(6, 2)

        pix2cm_y = row.Height / height
        pix2cm_x = row.Width / width

        row_data['y_pix2cm_x'].append(pix2cm_x)
        row_data['y_pix2cm_y'].append(pix2cm_y)
        row_data['y_mask_ivs'].append(row.IVSm)
        row_data['y_mask_lvid'].append(row.LVIDm)
        row_data['y_mask_lvpw'].append(row.LVPWm)

    row_data['X_bcoords'] = tcoords
    return row_data




def aggregate_to_dict(results):
    """
    Aggregates a list of dictionaries into a single dictionary by combining values under the same keys.

    Args:
        results: A list of dictionaries (or None), where each dictionary contains key-value pairs.
                 Values can be either lists or single items.

    Returns:
        A dictionary where each key maps to a list of aggregated values from all input dictionaries.
    """
    data_dict = {}

    for result in results:
        if result is not None:
            # Iterate through each key-value pair in the dictionary
            for key, value in result.items():
                # If the value is a list, extend the existing list
                if isinstance(value, list):
                    data_dict.setdefault(key, []).extend(value)
                else:
                    # Otherwise, append the single value to the list
                    data_dict.setdefault(key, []).append(value)

    return data_dict


def generate_h5(args : Any) -> None:
    """
    Main function to generate an HDF5 dataset from a metadata CSV file.

    Steps:
    - Loads and preprocesses metadata.
    - Filters annotations based on the number of structures.
    """
    # Load and preprocess metadata
    df = pd.read_csv(args.metadata_path, index_col=0)
    df = create_df(df,args)

    # Parse annotation structure counts
    args.num_annot_structures = [
        int(s) for s in args.num_annot_structures.split('+')
    ]
    df = df[df['count'].isin(args.num_annot_structures)].reset_index(drop=True)

    movie_ids = df['movieid'].unique()
    results = [None] * len(movie_ids)

    # Construct output filename
    h5_filepath = 'plaxlv_echonetlvh'
    if len(args.num_annot_structures) < 3:
        suffix = '_'.join(str(v) for v in args.num_annot_structures)
        h5_filepath += f"_mini_{suffix}"
    h5_filepath += '.h5'

    h5_filepath = os.path.join(args.output_dir, h5_filepath)
    h5_container = echotools.H5Container(h5_filepath)
    try:
        # Process each movie in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(generate_data, movieid, df): i
                for i, movieid in enumerate(movie_ids)
            }

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(movie_ids)):
                i = futures[future]
                results[i] = future.result()

                if results[i] is not None:
                    h5_container.store_as_dataset(results[i], target_keys=['X_bmovie'])
                    del results[i]['X_preid']
                    del results[i]['X_bmovie']
        
        # Aggregate results into a single dictionary
        data_dict = aggregate_to_dict(results)

        h5_container.store_as_chunked_dataset(data_dict,target_keys=[

                                                            'y_keyframe_idx',
                                                            'y_split',
                                                            'y_phase',
                                                            'y_width',
                                                            'y_height',
                                                            'y_frames',
                                                            'y_movieid',
                                                            'y_pix2cm_x',
                                                            'y_pix2cm_y',
                                                            'y_mask_ivs',
                                                            'y_mask_lvid',
                                                            'y_mask_lvpw',
                                                            'X_bcoords',
        ])
    finally:
        # closing h5 container
        h5_container.close()
    

if __name__ == "__main__":
    print('staring bmode echonet LVH measurement buidling data',flush=True)
    args = parse_args()
    generate_h5(args)
    
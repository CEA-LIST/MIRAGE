import copy
import glob
import os
import torch
import numpy as np
import pickle
import logging
from tqdm import tqdm
import json
from torch_geometric.data import Dataset, Data
from mmrnet.utils import TQDMBytesReader
from mmrnet.dataset.augmentation import *
from functools import partial
from .voxelize import pointcloud_to_spatiotemporal_voxel

class MIRAGEData(Dataset):
    raw_data_path = 'data/MIRAGE/raw'
    processed_data = None
    seed = 42
    stacks = None
    forced_rewrite = False
    include_time = True
    total_features = 9 # max dataset features # # x, y, z, frame_idx, range, velocity, doppler_bin, bearing, intensity
    num_features = 9
    num_pos_features = 9
    num_classes = 12
    data = None
    statistics = None
    zero_padding = 'no'
    max_points=22
    kfold = None
    model_type = 'graph'

    def _parse_config(self, c):
        c = {k: v for k, v in c.items() if v is not None}
        self.seed = c.get('seed', self.seed)
        self.processed_data = c.get('processed_data', self.processed_data)
        self.stacks = c.get('stacks', self.stacks)
        self.forced_rewrite = c.get('forced_rewrite', self.forced_rewrite)
        self.augmentations = json.loads(c.get('data_augmentation', '[]'))
        self.max_points = c.get('max_points', self.max_points)
        self.zero_padding = c.get('zero_padding', self.zero_padding)
        self.input_features = json.loads(c.get('input_features', '[]'))
        self.posconv_features = json.loads(c.get('posconv_features', '[]'))
        self.posgraph_features = json.loads(c.get('posgraph_features', '[]'))
        self.kfold = c.get('kfold', self.kfold)
        self.model_type = c.get('model_type', self.model_type)

    def __init__(
            self, root, partition, 
            transform=None, pre_transform=None, pre_filter=None,
            mmr_dataset_config = None):
        
        # Initialize superclass 
        super(MIRAGEData, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        # Parse the provided configuration
        self._parse_config(mmr_dataset_config)
        
        # LOAD THE DATASET
        if MIRAGEData.data is None:
            if (not os.path.isfile(self.processed_data)) or self.forced_rewrite:
                # Create the processed/ directory if it does not exist:
                os.makedirs(os.path.dirname(self.processed_data), exist_ok=True)
                # Process the raw dataset and write it to file
                MIRAGEData.data, _ = self._process()
                logging.info(f'Saving processed data to {self.processed_data}')
                with open(self.processed_data, 'wb') as f:
                    pickle.dump(MIRAGEData.data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logging.info('Processed data saved successfully.')
            else:
                # Load the already available processed dataset from file
                with open(self.processed_data, 'rb') as f:
                    f_size = os.path.getsize(self.processed_data)
                    with TQDMBytesReader(f, total=f_size) as pbfd:
                        up = pickle.Unpickler(pbfd)
                        MIRAGEData.data = up.load()
            # Compute and save statistics for the training set
            MIRAGEData.statistics = self._compute_statistics(MIRAGEData.data['train'])
            logging.info(f"Statistics computed: {MIRAGEData.statistics}")
        self.data = MIRAGEData.data

        ## compute feature mask
        self.input_features_mask = self._initialize_features(self.input_features)
        self.posconv_features_mask = self._initialize_features(self.posconv_features)
        self.posgraph_features_mask = self._initialize_features(self.posgraph_features)

        # MIRAGEData.statistics_test = self._compute_statistics(MIRAGEData.data['test'])
        # logging.info(f"Statistics computed on test set: {MIRAGEData.statistics_test}")
        
        # Initialize the parameters
        total_samples = len(self.data['train']) + len(self.data['val']) + len(self.data['test'])
        self.class_weights = self.data['class_weights'] if 'class_weights' in self.data else None
        self.data = self.data[partition]
        self.num_samples = len(self.data)
        self.target_dtype = torch.float
        self.info = {
            'num_samples': self.num_samples,
            'num_classes': self.num_classes,
            'stacks': self.stacks,
            'partition': partition,
            'class_weights': self.class_weights,
            'num_features': self.num_features,
            'num_pos_features': self.num_pos_features,
            'augmentations': self.augmentations,
            'seed': self.seed,
        }
        logging.info(
            f'Loaded {partition} data with {self.num_samples} samples,'
            f' where the total number of samples is {total_samples}.\n'
            )
        if partition == 'train':
            logging.info(f'Augmentations: {self.augmentations}')

    
    def len(self):
        return self.num_samples
        
    def _build_augmentation_map(self):
        """Pre-build augmentation map to avoid recreation on each get() call."""
        # Initialize epoch-related attributes if not set
        if not hasattr(self, 'max_epochs'):
            self.max_epochs = 100  # Default value
        if not hasattr(self, 'epoch'):
            self.epoch = 0  # Default value
        
        self._augmentation_map = {
            'random_translation': random_translation,
            'random_rotation': random_rotation,
            'random_scaling': random_scaling,
            'time_stretching': time_stretching,
            'random_node_dropout': partial(random_node_dropout, dropout_rate=0.3),
            'add_random_points': partial(add_random_points, percentage=0.3),
        }
        

    def get(self, idx):
        """Retrieve a data point by index, apply augmentations, and return the processed data.
        Point cloud features are expected to be: {x, y, z, frame_idx, range, velocity, doppler_bin, bearing, intensity}.
        Note: range, doppler_bin and bearing are not recorded in MIRAGE (have 0 value)."""
        data_point = self.data[idx]
        y = torch.tensor(data_point['y'], dtype=torch.long)
        x = data_point['new_x']

        # rescale T
        if x.shape[1] > 3:
            # remove the time offset (put time indexes in 0->nb_stack)
            x[:, 3] = x[:, 3] - x[:, 3].min()
            # rescale to have time between 0 and 1
            x[:, 3] = x[:, 3] / self.stacks # rescale time to be [0,1] based on stack length

        # DATA AUGMENTATION
        if self.info['partition'] == 'train':
            x = x.copy()  # Ensure x is a copy to avoid modifying the original data
            if not hasattr(self, '_augmentation_map') or self._augmentation_map is None:
                self._build_augmentation_map()
            for augmentation_name in self.augmentations:
                augmentation = self._augmentation_map.get(augmentation_name)
                x, _ = augmentation(x)

        # NORMALIZE OTHER FEATURES (after data augmentation because data augmentation use min/max statistics computed on unnormalized data)
        stats = MIRAGEData.statistics['new_x']
        x[:, 5] = (x[:, 5] - stats["mean"][5]) / stats["std"][5] # normalize velocity
        x[:, 8] = (x[:, 8] - stats["mean"][8]) / stats["std"][8] # normalize intensity

        # SELECT FEATURES
        x_tensor = torch.tensor(x, dtype=torch.float32)
        pos_tensor = torch.tensor(x, dtype=torch.float32)
        pos_graph_tensor = torch.tensor(x, dtype=torch.float32)
        # put unused features to zero using features masks
        x_tensor[:, torch.logical_not(self.input_features_mask)] = 0
        pos_tensor[:, torch.logical_not(self.posconv_features_mask)] = 0
        pos_graph_tensor[:, torch.logical_not(self.posgraph_features_mask)] = 0

        if 'voxel' in self.model_type:
            x_numpy = x_tensor.numpy()
            if 'voxellstm' in self.model_type:
                x_numpy = pointcloud_to_spatiotemporal_voxel(x_numpy, self.stacks, LSTM=True) # (X,Y,Z,T,C)
            else:  
                x_numpy = pointcloud_to_spatiotemporal_voxel(x_numpy, self.stacks) # (X,Y,Z,T,C)
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
            x_tensor = x_tensor.permute(3,2,1,0,4) # (T,Z,Y,X,C)
            return x_tensor,y
        elif 'padded' in self.model_type:
            x_tensor = torch.tensor(x[:, :self.num_features], dtype=torch.float32)
            return x_tensor,y
        else:
            x_data = Data(x=x_tensor, pos=pos_tensor, pos_graph=pos_graph_tensor)
            return x_data, y

    
    def _compute_statistics(self, data):
        """
        Compute statistics for the training set.

        Args:
            data (list or array-like): The dataset to compute statistics for.

        Returns:
            dict: A dictionary containing the computed statistics.
        """

        data_tmp = [item['new_x'] for item in data]
        data_tmp = np.concatenate(data_tmp, axis=0)
        print("## points per frame avg on the split", data_tmp.shape[0] / len(data))
        statistics = {}
        # Compute statistics for 'new_x'
        statistics['new_x'] = {
            "mean": np.mean(data_tmp, axis=0),  # Mean across all samples
            "std": np.std(data_tmp, axis=0),   # Standard deviation across all samples
            "min": np.min(data_tmp, axis=0),   # Minimum value across all samples
            "max": np.max(data_tmp, axis=0),   # Maximum value across all samples
        }

        return statistics

    def _process(self):

        train_list, _ = self._parse_entire_dataset(partition='Train')
        val_list, _ = self._parse_entire_dataset(partition='Validation')
        test_list, _ = self._parse_entire_dataset(partition='Test')

        # stack and pad frames based on config
        train_list = self.stack_and_padd_frames(train_list)
        val_list = self.stack_and_padd_frames(val_list)
        test_list = self.stack_and_padd_frames(test_list)        

        num_samples = len(train_list) + len(val_list) + len(test_list)

        data_map = {
            'train': train_list,
            'val': val_list,
            'test': test_list,
        }
        return data_map, num_samples


    def _parse_file_by_frame(self, txt_path, label_idx, file_id):
        """
        Parses a txt file and returns a list of frame dictionaries,
        each with 'x' (point cloud with features) and 'y' (class label).
        """
        frames = []
        current_points = []
        current_frame_idx = -1

        with open(txt_path, 'r') as f:
            content = f.read()

        # Split the file into point blocks
        blocks = content.split('---')

        for block in blocks:
            lines = block.strip().splitlines()
            if not lines:
                continue

            data = {}
            for line in lines:
                if ':' in line:
                    key, val = line.split(':', 1)
                    data[key.strip()] = val.strip()

            if 'point_id' not in data:
                continue

            if data['point_id'] == '0':
                # New frame starting
                if current_points:
                    frames.append({
                        'x': np.array(current_points, dtype=np.float32),
                        'y': label_idx,
                        "file_id": file_id,
                    })
                    current_points = []
                current_frame_idx += 1

            try:
                x = float(data.get('x', 0))
                y = float(data.get('y', 0))
                z = float(data.get('z', 0))
                r = float(data.get('range', 0))
                v = float(data.get('velocity', 0))
                d = float(data.get('doppler_bin', 0))
                b = float(data.get('bearing', 0))
                i = float(data.get('intensity', 0))
                current_points.append([x, y, z, current_frame_idx, r, v, d, b, i])
            except ValueError:
                continue  # Skip corrupted values

        # Add last frame
        if current_points:
            frames.append({
                'x': np.array(current_points, dtype=np.float32),
                'y': label_idx,
                "file_id": file_id,
            })

        return frames


    def _parse_entire_dataset(self, partition):
        """
        Automatically parses all action subfolders for the given partition (train/test/val)
        and builds the dataset.
        """
        action_map = {}
        data = []
        label_counter = 0

        partition_key = {
            'Train': 'Train',
            'Validation': 'Train',
            'Test': 'Test'
        }

        kfold_key = {
            0: ['subject0', 'subject1'],
            1: ['subject2', 'subject3'],
            2: ['subject4', 'subject5'],
            3: ['subject6', 'subject7'],
            4: ['subject8', 'subject9'],
        }

        partition_dir = os.path.join(self.raw_data_path, partition_key[partition])
        print(f"Parsing MIRAGE dataset partition: {partition_dir}")

        ### Find the path of the subjects in the corresponding split
        listsubject = []
        for environment in os.listdir(partition_dir):
            print("environment:", environment)
            environment_path = os.path.join(partition_dir, environment)
            if not os.path.isdir(environment_path):
                continue

            for subject in os.listdir(environment_path):
                subject_path = os.path.join(environment_path, subject)
                if (self.kfold is not None) and (partition != 'Test'):
                    if subject in kfold_key[self.kfold]:
                        if partition == 'Validation':
                            listsubject.append(subject_path)
                    elif partition == 'Train':
                        listsubject.append(subject_path)
                else:
                    listsubject.append(subject_path)

        print(f"split: {partition}, kfold: {self.kfold}, subjects: {listsubject}")
            
        for subject_path in listsubject:
            if not os.path.isdir(subject_path):
                continue

            for action in sorted(os.listdir(subject_path)):
                if "z" in action: #remove background sequences recordings
                    continue
                action_path = os.path.join(subject_path, action)
                if not os.path.isdir(action_path):
                    continue

                if action not in action_map:
                    action_map[action] = label_counter
                    label_counter += 1

                label_idx = action_map[action]
                txt_files = sorted(glob.glob(os.path.join(action_path, "*.txt")))

                for i,txt in enumerate(txt_files):
                    filename = os.path.basename(txt)
                    # to keep only clean or noise samples
                    if "evalnoisy" in self.processed_data:
                        if "nois" not in filename: # skip clean samples
                            continue
                    elif "evalclean" in self.processed_data:
                        if ("nois" in filename): # skip noisy samples
                            continue
                    frames = self._parse_file_by_frame(txt, label_idx, i)
                    data.extend(frames)

        print("## Number of actions in this split:", label_counter)
        return data, action_map


    
    def stack_and_padd_frames(self, data_list):
        """Stacks and zero-pads the point clouds

        Returns a list where each point cloud is mapped to its respective stack of zero-padded point clouds.
        The zero-padding is performed with respect to the parameter 'self.max_points', and that shape is enforced to each point cloud.
        The stacking is performed with respect to the parameter 'self.stacks', and that is the number of data points that are mapped to each
        original data point in the returned list.
        
        """

        if self.stacks is None:
            # if stacking is not requested do nothing
            return data_list
        
        # TAKE MULTIPLE FRAMES FOR EACH Xs
        
        xs = [d['x'] for d in data_list]    # Store all the point clouds in the variable xs
        action_labels = [d['y'] for d in data_list]  # Store all the action labels
        file_id = [d['file_id'] for d in data_list] # Store all file ids
        assert(len(xs) == len(action_labels) == len(file_id))
        new_data_list = []  # This will hold the new data points after stacking and padding
        print("Stacking and padding frames...")

        # NO ZERO PADDING PERFORMED, GRAPH TYPE MODEL NEEDED
        for i in tqdm(range(len(xs))):
            data_point_x = []
            for j in range(self.stacks):
                if i - j >= 0 and action_labels[i] == action_labels[i-j] and file_id[i] == file_id[i-j]:
                    # CHECK THAT WE ARE STACKING ONLY FRAMES BELONGING TO THE SAME ACTION AND SUBJECT
                    # check also that we are stacking frames belonging to the same file (file_id)
                    tmp_x = xs[i - j]
                    data_point_x.append(tmp_x)
                else:
                    break

            if self.zero_padding in ['per_data_point', 'data_point']:
                # We are padding each frame individually
                for k in range(len(data_point_x)):
                    diff = self.max_points - data_point_x[k].shape[0]
                    data_point_x[k] = np.pad(data_point_x[k], ((0, max(diff, 0)), (0, 0)), 'constant')
                    data_point_x[k] = data_point_x[k][np.random.choice(len(data_point_x[k]), self.max_points, replace=False)]

            if data_point_x and len(data_point_x)==self.stacks:  # Must contain exactly self.stacks frames
                new_x = np.concatenate(data_point_x, axis=0)
                new_data_list.append({
                    'x': xs[i],  # Original x data
                    'y': action_labels[i],  # Original y data
                    'new_x': new_x,  # Stacked and processed x data
                })
        print("Stacking and padding frames done")
        return new_data_list
    
    def _initialize_features(self, features):
        # x, y, z, frame_idx, range, velocity, doppler_bin, bearing, intensity
        mask = torch.zeros(self.total_features, dtype=bool)
        if "xyz" in features:
            mask[:3] = 1
        if "time" in features:
            mask[3] = 1
        if "velocity" in features:
            mask[5] = 1
        if "intensity" in features:
            mask[8] = 1
        return mask
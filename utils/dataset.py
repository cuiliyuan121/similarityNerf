import contextlib
import pickle
import os
from PIL import Image
import numpy as np
import yaml
import networkx as nx
import random
from scipy.spatial.transform import Rotation
import inspect
import matplotlib.pyplot as plt
import re
from external import blenderproc_utils
import json
from collections import defaultdict
from glob import glob
from .transform import crop_camera, total3d_world_rotation, uv2cam, cam2uv, homotrans, get_backend, WORLD_FRAME_TO_TOTAL3D, CAMERA_FRAME_TO_TOTAL3D
from itertools import chain
from contextlib import suppress


def check_data(data_dir, strict=True):
    # check if the data directory exists
    data_dirs = {}
    if strict and not os.path.exists(data_dir):
        raise Exception('Data directory does not exist: {}'.format(data_dir))
    data_dirs['front_dir'] = os.path.join(data_dir, '3D-FRONT')
    data_dirs['future_dir'] = os.path.join(data_dirs['front_dir'], '3D-FUTURE-model')
    data_dirs['front_texture_dir'] = os.path.join(data_dirs['front_dir'], '3D-FRONT-texture')
    data_dirs['front_scene_dir'] = os.path.join(data_dirs['front_dir'], '3D-FRONT')
    data_dirs['cc_material_dir'] = os.path.join(data_dir, 'cctextures')
    for dir in data_dirs.values():
        if strict and not os.path.exists(dir):
            raise Exception('Directory does not exist: {}'.format(dir))
    return data_dirs


def load_splits(dir, split_names=None):
    if split_names is None:
        json_files = glob(os.path.join(dir, '*.json'))
        split_names = [os.path.splitext(os.path.basename(s))[0] for s in json_files]
        split_names = [s for s in split_names if s in ('train', 'val', 'test')]

    # scan folder if split files don't exist
    if not split_names:
        scenes = glob(os.path.join(dir, '*/'))
        scene_ids = [os.path.basename(os.path.normpath(s)) for s in scenes]
        scene_ids.sort()
        return {'all': scene_ids}

    # use existing split files
    splits = {}
    for split in split_names:
        split_json = os.path.join(dir, split + '.json')
        with open(split_json, 'r') as f:
            splits[split] = json.load(f)
    return splits


def save_splits(dir, split_object_ids):
    for split_name, split_object_id in split_object_ids.items():
        split_object_id.sort()
        print(f"Saving split {split_name} with {len(split_object_id)} objects...")
        with open(os.path.join(dir, f'{split_name}.json'), 'w') as f:
            json.dump(split_object_id, f, indent=4)


class RecursiveCall(dict):
    def recursive_copy_call(self, func, unique_args=None, **kwargs):
        d = dict(self).copy()
        for key, value in d.items():
            unique_kwargs = kwargs.copy()
            if unique_args is not None and key in unique_args and unique_args[key] is not None:
                unique_kwargs.update(unique_args[key])

            if isinstance(value, RecursiveCall):
                d[key] = getattr(value, func)(**unique_kwargs)
            elif isinstance(value, dict):
                d[key] = {k: getattr(v, func)(**unique_kwargs) if isinstance(v, RecursiveCall) else v for k, v in value.items()}
            elif isinstance(value, list):
                d[key] = [getattr(v, func)(**unique_kwargs) if isinstance(v, RecursiveCall) else v for v in value]
        return d

    def recursive_inplace_call(self, func, *args, **kwargs):
        for value in self.values():
            if isinstance(value, RecursiveCall) and hasattr(value, func):
                getattr(value, func)(*args, **kwargs)
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, RecursiveCall) and hasattr(v, func):
                        getattr(v, func)(*args, **kwargs)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, RecursiveCall) and hasattr(v, func):
                        getattr(v, func)(*args, **kwargs)


class DataFilter(RecursiveCall):
    def data_dict(self, keys=None, **kwargs):
        if keys is None or keys == 'all':
            data = {k: self[k] for k in self.keys()}  # call __getitem__ in case preprocessing is needed
        else:
            data = {}
            for k in set(keys) | set(kwargs.keys()):
                # skip keys that are not in the data
                with contextlib.suppress(Exception):
                    data[k] = self[k]
        sub_keys = {k: v for k, v in keys.items() if v is not None} if isinstance(keys, dict) else {}
        sub_keys.update(kwargs)
        data = RecursiveCall(data).recursive_copy_call('data_dict', unique_args=sub_keys)
        return data

    def pickle_dict(self):
        return dict(self).copy()


class DictOfTensor(RecursiveCall):
    def __init__(self, data, backend=np, device=None):
        super().__init__(data)
        self.backend = backend
        self.if_detach = False
        self.if_clone = False
        self.device = device

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if self.backend is np and value.__class__.__name__ == 'Tensor':
            value = value.detach().cpu().numpy()
            super().__setitem__(key, value)
        if isinstance(value, np.ndarray) and value.dtype == np.float64:
            value = value.astype(np.float32)
            super().__setitem__(key, value)
        if self.backend is not np and isinstance(value, np.ndarray):
            value = self.backend.from_numpy(value)
            super().__setitem__(key, value)
        if self.device is not None and value.__class__.__name__ == 'Tensor' and value.device != self.device:
            value = value.to(self.device)
            super().__setitem__(key, value)
        if self.if_detach and value.__class__.__name__ == 'Tensor' and key not in self.detached:
            value = value.detach()
            super().__setitem__(key, value)
            self.detached.add(key)
        if self.if_clone and value.__class__.__name__ == 'Tensor' and key not in self.cloned:
            value = value.clone()
            super().__setitem__(key, value)
            self.cloned.add(key)
        return value

    def tensor(self, now=False, device=None):
        import torch
        d = type(self)(self.recursive_copy_call('tensor', now=now, device=device), backend=torch, device=device)
        if now:
            for k in d:
                d[k]
        return d

    def numpy(self, now=False):
        d = type(self)(self.recursive_copy_call('numpy', now=now), backend=np)
        if now:
            for k in d:
                d[k]
        return d

    def detach(self, now=False):
        d = type(self)(self.recursive_copy_call('detach', now=now), backend=self.backend, device=self.device)
        d.if_detach = True
        d.detached = set()
        if now:
            for k in d:
                d[k]
            d.if_detach = False  # if now, then no need to detach again
            d.detached = None
        return d

    def clone(self, now=False):
        d = type(self)(self.recursive_copy_call('clone', now=now), backend=self.backend, device=self.device)
        d.if_clone = True
        d.cloned = set()
        if now:
            for k in d:
                d[k]
            d.if_clone = False  # if now, then no need to clone again
            d.cloned = None
        return d

    def _requires_grad(self, requires_grad=True):
        self.recursive_inplace_call('_requires_grad', requires_grad=requires_grad)
        return self

    def copy(self):
        new_dict = self.recursive_copy_call('copy')
        return type(self)(new_dict, backend=self.backend, device=self.device)


class GetFailedError(Exception):
    pass


class AutoGetSetDict(DictOfTensor):
    ref_graph = None
    _get_prefix = '_get_'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ref_graph is None:
            self._generate_ref_graph()
        self.ref_stack = []

    @classmethod
    def _generate_ref_graph(cls):
        # Scan for reference groups in get methods
        attr_ref_groups = defaultdict(list)
        for c in inspect.getmro(cls):
            if not issubclass(c, AutoGetSetDict):
                continue
            scaned_func = set()
            for name, func in inspect.getmembers(
                    c, predicate=lambda f: inspect.isfunction(f) and f.__name__.startswith(cls._get_prefix)):
                if func in scaned_func:
                    continue
                scaned_func.add(func)
                attribute = name[len(c._get_prefix):]

                # scan for renferences of attribute in the code
                lines = inspect.getsource(func).splitlines()
                refs = set()
                for line in lines:
                    # find all the reference like self['*'] or self["*"] in each line with regex
                    # add an edge from the attribute to the reference
                    matches = re.findall(r"self\[['\"]([^\[\]]*)['\"]\]", line)
                    refs.update(matches)

                    if 'return' in line and refs:
                        attr_ref_groups[attribute].append(refs)
                        refs = set()

                if refs:
                    attr_ref_groups[attribute].append(refs)

        ref_graph = nx.DiGraph()

        # add nodes to the graph
        for attribute, ref_groups in attr_ref_groups.items():
            ref_graph.add_node(attribute)
            ref_graph.add_nodes_from(set().union(*ref_groups))

        # add references to the graph
        ref_stack = []
        
        def add_refs(now):
            # avoid cycle
            if now in ref_stack:
                return
            # add root attribute to ref_stack
            if ref_stack:
                ref_graph.add_edge(ref_stack[0], now, indirect=len(ref_stack) - 1)
            # skip leaf attributes
            if now not in attr_ref_groups:
                return
            ref_stack.append(now)
            for ref_group in attr_ref_groups[now]:
                # avoid references with other attributes in the same group
                if len(ref_stack) >= 2 and ref_stack[-2] in ref_group:
                    continue
                for ref in ref_group:
                    add_refs(ref)
            ref_stack.pop()

        for attribute in attr_ref_groups:
            add_refs(attribute)

        cls.ref_graph = ref_graph
        
    @classmethod
    def visualize_ref_graph(cls, dir):
        if cls.ref_graph is None:
            cls._generate_ref_graph()
        pos = nx.nx_agraph.graphviz_layout(cls.ref_graph)
        options = {
            "with_labels": True,
            "font_weight": "bold",
            "node_color": "#1f78b4",
            "width": 3,
            "node_size": 300
        }
        nx.draw(cls.ref_graph, pos, **options)
        edge_labels = nx.get_edge_attributes(cls.ref_graph, 'indirect')
        nx.draw_networkx_edge_labels(cls.ref_graph, pos, edge_labels)
        output_image_dir = os.path.join(dir, f"{cls.__name__}_ref_graph.png")
        plt.savefig(output_image_dir)
        plt.close()
    
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            # if the attribute is not in the dict, calculate and cache the attribute
            if hasattr(self, self._get_prefix + key):
                if key in self.ref_stack:
                    raise GetFailedError(f"Recursive reference detected: {self.ref_stack + [key]}")
                self.ref_stack.append(key)
                try:
                    value = getattr(self, self._get_prefix + key)()
                finally:
                    self.ref_stack.pop()
                super().__setitem__(key, value)
                return super().__getitem__(key)
            else:
                raise e

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # If the attribute is in the reference graph and has renferences
        if key in self.ref_graph:
            # update its predecessors
            self.update_predecessors(key)

    def update_predecessors(self, key):
        for predecessor in self.ref_graph.predecessors(key):
            for successor in nx.node_boundary(self.ref_graph, list(self.ref_graph.predecessors(key)) + [key]):
                # cache other direct references if the predecessor to pop has multiple references
                if self.ref_graph.get_edge_data(predecessor, successor, {}).get('indirect') == 0:
                    with contextlib.suppress(Exception):
                        self[successor]
            self.pop(predecessor, None)


class CategoryMapping:
    object_category2future = {}
    object_future2category = {}
    object_categories = []

    @classmethod
    def load_category_mapping(cls, dir):
        # load category mapping if not loaded
        if not cls.object_category2future:
            with open(os.path.join(dir, 'object_category2future.yaml'), 'r') as f:
                cls.object_category2future.update(yaml.load(f, Loader=yaml.FullLoader))
            with open(os.path.join(dir, 'object_categories.yaml'), 'r') as f:
                cls.object_categories.extend(yaml.load(f, Loader=yaml.FullLoader))
            for category, futures in cls.object_category2future.items():
                for future in futures:
                    # get the first string before ' / ' or '/'
                    future = future.split(' / ')[0].split('/')[0]
                    cls.object_future2category[future] = category


class Object(AutoGetSetDict, DataFilter, CategoryMapping):
    orientation_bin = None

    @classmethod
    def generate_orientation_bin(cls, num_bins):
        if cls.orientation_bin is None:
            cls.orientation_bin = np.linspace(-np.pi, np.pi, num_bins + 1, dtype=np.float32)

    def _get_category(self):
        if 'future_category' in self:
            future_category = self['future_category']
            # get the first string before ' / ' or '/'
            future_category = future_category.split(' / ')[0].split('/')[0]
            if future_category in self.object_future2category:
                return self.object_future2category[future_category]
            # loose match
            for category in self.object_categories:
                if category in future_category.lower():
                    return category
            return 'other'
        return self.object_categories[self['category_id']]

    def _get_category_id(self):
        if 'category_onehot' in self:
            return self['category_onehot'].argmax()
        try:
            return self.object_categories.index(self['category'])
        except ValueError:
            return None

    def _get_category_onehot(self):
        onehot = np.zeros(len(self.object_categories), dtype=np.float32)
        onehot[self['category_id']] = 1
        return onehot

    def _get_orientation_cls(self):
        if 'orientation_score' in self:
            return self['orientation_score'].argmax()
        cls = np.digitize(self['orientation'], self.orientation_bin) - 1
        assert cls >= 0 and cls < len(self.orientation_bin) - 1
        return cls

    def _get_orientation(self):
        orientation = self.orientation_bin[[int(self['orientation_cls']), int(self['orientation_cls']) + 1]].mean()
        return self.backend.tensor(orientation, device=self.device) if self.backend is not np else orientation


class Rotation6D(AutoGetSetDict):
    def _get_rotation_6d(self):
        from pytorch3d.transforms import matrix_to_rotation_6d
        import torch
        rotation_mat = torch.tensor(self['rotation_mat']) if self.backend is np else self['rotation_mat']
        rotation_6d = matrix_to_rotation_6d(rotation_mat)
        rotation_6d = rotation_6d.numpy() if self.backend is np else rotation_6d
        return rotation_6d

    def _get_rotation_mat(self):
        from pytorch3d.transforms import rotation_6d_to_matrix
        import torch
        rotation_6d = torch.tensor(self['rotation_6d']) if self.backend is np else self['rotation_6d']
        rotation_mat = rotation_6d_to_matrix(rotation_6d)
        rotation_mat = rotation_mat.numpy() if self.backend is np else rotation_mat
        return rotation_mat


class Object3D(Object, Rotation6D):
    def _get_local2world_mat(self):
        if 'world2local_mat' in self:
            return self.backend.linalg.inv(self['world2local_mat'])
        local2world_mat = self.backend.eye(4, dtype=self['center'].dtype)
        if self.backend is not np:
            local2world_mat = local2world_mat.to(self['center'].device)
        local2world_mat[:3, :3] = self['rotation_mat']
        local2world_mat[:3, 3] = self['center']
        return local2world_mat

    def _get_world2local_mat(self):
        return self.backend.linalg.inv(self['local2world_mat'])

    def _get_center(self):
        return self['local2world_mat'][:3, 3]

    def _get_rotation_mat(self):
        try:
            if self.backend is np:
                return Rotation.from_euler('z', self['orientation']).as_matrix()
            import pytorch3d
            obj_euler = self.backend.cat([(self['orientation']).unsqueeze(0), self.backend.zeros(2, device=self.device)], dim=0)
            return pytorch3d.transforms.euler_angles_to_matrix(obj_euler, 'ZXY').squeeze(0)
        except GetFailedError:
            if 'rotation_6d' in self:
                return super()._get_rotation_mat()
            return self['local2world_mat'][:3, :3]

    def _get_orientation(self):
        try:
            return super()._get_orientation()
        except GetFailedError:
            if self.backend is np:
                return Rotation.from_matrix(self['rotation_mat']).as_euler('zxy')[0]
            import pytorch3d
            return -pytorch3d.transforms.matrix_to_euler_angles(self['rotation_mat'].T, 'ZXY')[0]

    def _get_down_vec(self):
        return -self['rotation_mat'][:, 2]

    def data_dict(self, keys=None, **kwargs):
        if keys is None:
            keys = ['rotation_mat', 'center', 'category_id', 'category', 'size', 'jid']
        return super().data_dict(keys, **kwargs)

    def mmdet3d(self):
        backend = get_backend(self['center'])
        xyz = backend.copy(self['center'])
        xyz[..., 2] -= self['size'][..., 2] / 2
        return backend.concatenate([xyz, self['size'], self['orientation'][..., None]])


class Object2D(Object):
    mean_size = {}

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        if 'camera' in self and not isinstance(self['camera'], Camera):
            self['camera'] = Camera(self['camera'], *args, **kwargs)

    @classmethod
    def load_mean_size(cls, dir):
        # load mean size if not loaded
        if not cls.mean_size:
            with open(os.path.join(dir, 'object_mean_size.json'), 'r') as f:
                cls.mean_size.update(json.load(f))
            for category in cls.mean_size:
                cls.mean_size[category] = np.array(cls.mean_size[category], dtype=np.float32)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if key == 'segmentation' and isinstance(value, dict):
            value = blenderproc_utils.rle_to_binary_mask(value)
        return value

    def pickle_dict(self):
        pickle_dict = super().pickle_dict()
        if isinstance(pickle_dict['segmentation'], np.ndarray):
            pickle_dict['segmentation'] = blenderproc_utils.binary_mask_to_rle(pickle_dict['segmentation'])
        return pickle_dict

    def data_dict(self, keys=None, **kwargs):
        if keys is None:
            keys = ['bdb2d', 'category', 'category_id', 'segmentation']
        return super().data_dict(keys, **kwargs)

    def _get_size_scale(self):
        return self['size'] / self.mean_size[self['category']]

    def _get_size(self):
        mean_size = self.mean_size[self['category']]
        if self.backend is not np:
            mean_size = self.backend.tensor(mean_size, device=self['size_scale'].device)
        return mean_size * self['size_scale']


class Camera(Rotation6D, DataFilter):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        if 'image' in self and not isinstance(self['image'], ImageIO):
            self['image'] = ImageIO(self['image'], *args, **kwargs)
        if 'object' in self:
            self['object'] = {k: v if isinstance(v, Object2D) else Object2D(v, *args, **kwargs) for k, v in self['object'].items()}

    def pickle_dict(self):
        pickle_dict = super().pickle_dict()
        pickle_dict.pop('image', None)
        if 'object' in pickle_dict:
            pickle_dict['object'] = {k: v.pickle_dict() for k, v in pickle_dict['object'].items()}
        return pickle_dict
    
    def _get_cam2world_mat(self):
        if 'cam2world_mat' in self:
            return self.backend.linalg.inv(self['world2cam_mat'])
        cam2world_mat = self.backend.eye(4, dtype=self['position'].dtype)
        if self.backend is not np:
            cam2world_mat = cam2world_mat.to(self['position'].device)
        cam2world_mat[:3, :3] = self['rotation_mat']
        cam2world_mat[:3, 3] = self['position']
        return cam2world_mat

    def _get_world2cam_mat(self):
        return self.backend.linalg.inv(self['cam2world_mat'])

    def _get_position(self):
        return self['cam2world_mat'][:3, 3]
    
    def _get_rotation_mat(self):
        if 'rotation_6d' in self:
            return super()._get_rotation_mat()
        return self['cam2world_mat'][:3, :3]
    
    def data_dict(self, keys=None, **kwargs):
        if keys is None:
            keys = ['cam2world_mat', 'K', 'height', 'width', 'image', 'object']
        return super().data_dict(keys, **kwargs)


class ImageIO(DictOfTensor, DataFilter):
    def __getitem__(self, key):
        image = super().__getitem__(key)
        if isinstance(image, str):
            image = np.array(Image.open(image))
            if key == 'depth':
                image = image.astype(np.float32) / 1000
            self[key] = image
        if key == 'color' and isinstance(image, np.ndarray) and image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        return image

    def save(self, dir, camera_id=None, key=None):
        os.makedirs(dir, exist_ok=True)
        if key is None:
            keys = self.keys()
        elif isinstance(key, str):
            keys = [key]
        else:
            keys = key

        for key in keys:
            image_name = f"{key}.png" if camera_id is None else f"{camera_id:04d}-{key}.png"
            if isinstance(self.get(key), str):
                if os.path.exists(self.get(key)):
                    os.link(self.get(key), os.path.join(dir, image_name))
            else:
                image = self[key]
                mode = None
                if key == 'depth':
                    image = (image * 1000).astype(np.uint16)
                elif key in ('color', 'masked_color'):
                    image = (image * 255).astype(np.uint8)
                elif key == 'alpha':
                    image = (np.concatenate([self['color'], image], axis=-1) * 255).astype(np.uint8)
                    image_name = image_name.replace('alpha', 'rgba')
                    mode = 'RGBA'
                Image.fromarray(image, mode=mode).save(os.path.join(dir, image_name))
    
    def data_dict(self, keys=None, **kwargs):
        if keys is None:
            keys = ['color']
        return super().data_dict(keys, **kwargs)

    def preprocess_image(self, keys=None, resize_width=None):
        keys = keys or ['color_norm']
        from torchvision import transforms
        import torch
        input_images = []
        for key in keys:
            if key == 'color_norm':
                color_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                input_images.append(color_transform(self['color']))
            elif key == 'mask':
                input_images.append(transforms.ToTensor()(self['instance_segmap'].astype(np.float32)))
            elif key == 'depth':
                input_images.append(transforms.ToTensor()(self['depth']))
            elif key == 'mask_color':
                # 目前使用的encoder checkpoints是在whitebackground下训练的
                mask = self['instance_segmap'].astype(np.bool)
                mask_color = self['color'].copy()
                mask_color[~mask, :] = 1.
                mask_color = transforms.ToTensor()(mask_color)
                mask_color = mask_color * 2 - 1
                input_images.append(mask_color)
            else:
                raise ValueError(f"Unknown key: {key}")
        input_image = torch.cat(input_images, dim=0)
        if resize_width is not None:
            input_image = transforms.Resize((resize_width, resize_width))(input_image)
        return input_image


class Affinity(DictOfTensor, DataFilter):
    def __getitem__(self, key):
        return super().__getitem__(key) if super().__contains__(key) else super().__getitem__(key[::-1]).T

    def __contains__(self, key):
        return super().__contains__(key) or super().__contains__(key[::-1])

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def subscene(self, camera_ids):
        return Affinity({
            k: v for k, v in self.items()
            if k[0] in camera_ids and k[1] in camera_ids
        }, backend=self.backend, device=self.device)


def sample_connected_component_of_camera_overlap_graph(connected_component, n_camera, max_try=None):
    assert n_camera < len(connected_component), f"n_camera should be smaller than {len(connected_component)}"
    assert nx.is_connected(connected_component), "connected_component should be connected"
    if max_try is None:
        max_try = len(connected_component) * 10

    # get camera proposals in the connected_component with n_camera cameras
    camera_proposals = set()
    # randomly choose a node to start
    start_node = random.choice(list(connected_component.nodes))
    camera_proposals.add(start_node)
    for _ in range(n_camera - 1):
        neighbors = list(nx.node_boundary(connected_component, camera_proposals))
        neighbor_to_add = random.choice(neighbors)
        camera_proposals.add(neighbor_to_add)

    return camera_proposals


class Scene(AutoGetSetDict, DataFilter, CategoryMapping):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        if 'camera' in self:
            # use __setitem__ to avoid popping attributes referencing camera
            dict.__setitem__(self, 'camera', {k: v if isinstance(v, Camera) else Camera(v, *args, **kwargs) for k, v in self['camera'].items()})
        if 'object' in self:
            self['object'] = {k: v if isinstance(v, Object3D) else Object3D(v, *args, **kwargs) for k, v in self['object'].items()}
        if 'affinity' in self:
            self['affinity'] = Affinity(self['affinity'], *args, **kwargs)

    @classmethod
    def from_dir(cls, dir):
        # load pickle file
        pickle_dir = os.path.join(dir, 'data.pkl')

        # scan images if pickle does not exist
        if not os.path.exists(pickle_dir):
            images = glob(os.path.join(dir, '*.png'))
            data = {
                'camera': defaultdict(lambda: {'image': {}}),
                'uid': os.path.basename(dir)
            }
            for image in images:
                name = os.path.splitext(os.path.basename(image))[0]
                camera_id, key = name.split('-')
                data['camera'][int(camera_id)]['image'][key] = image
            return cls(data)

        CategoryMapping.load_category_mapping(os.path.dirname(os.path.normpath(dir)))

        # load from pickle
        with open(pickle_dir, 'rb') as f:
            data = pickle.load(f)
        
        # generate image file dirs
        for camera_id, camera in data['camera'].items():
            camera['image'] = {}
            for key in ['color', 'instance_segmap', 'depth']:
                image_dir = os.path.join(dir, f"{camera_id:04d}-{key}.png")
                camera['image'][key] = image_dir
        scene = cls(data)
        
        # remove objects not in the object category list
        if 'object' in scene:
            scene['object'] = {k: v for k, v in scene['object'].items() if v['category'] in scene.object_categories}
        for camera in scene['camera'].values():
            if 'object' in camera:
                camera['object'] = {k: v for k, v in camera['object'].items() if v['category'] in scene.object_categories}
        if 'seg2obj_id' in scene:
            scene['seg2obj_id'] = {k: v for k, v in scene['seg2obj_id'].items() if v in scene['object']}
        
        return scene
            
    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        
        # save images
        self.save_images(dir)
        
        # write data as pickle
        with open(os.path.join(dir, 'data.pkl'), 'wb') as f:
            pickle.dump(self.pickle_dict(), f)
    
    def save_images(self, dir, key=None):
        for camera_id, camera in self['camera'].items():
            if 'image' in camera:
                camera['image'].save(dir, camera_id=camera_id, key=key)
            
    def pickle_dict(self):
        pickle_dict = super().pickle_dict()
        if 'camera' in pickle_dict:
            pickle_dict['camera'] = {k: v.pickle_dict() for k, v in pickle_dict['camera'].items()}
        if 'object' in pickle_dict:
            pickle_dict['object'] = {k: v.pickle_dict() for k, v in pickle_dict['object'].items()}
        if 'affinity' in pickle_dict:
            pickle_dict['affinity'] = pickle_dict['affinity'].pickle_dict()
        return pickle_dict

    def _get_camera_graph(self):
        camera_graph = nx.DiGraph()
        for cam1_idx, (cam1_id, cam1) in enumerate(self['camera'].items()):
            camera_graph.add_node(cam1_id, camera=cam1)
            for cam2_idx, (cam2_id, cam2) in enumerate(self['camera'].items()):
                if cam1_idx == cam2_idx:
                    continue
                edge_properties = {
                    k: self[k][cam1_idx, cam2_idx]
                    for k in self.keys()
                    if k.startswith('relative_yaw')
                }
                edge_properties['affinity'] = self['affinity'][(cam1_id, cam2_id)]
                camera_graph.add_edge(cam1_id, cam2_id, **edge_properties)
        return camera_graph

    @property
    def camera_overlap_graph(self):
        camera_overlap_graph = nx.Graph()
        for cam1_id, cam1 in self['camera'].items():
            camera_overlap_graph.add_node(cam1_id, camera=cam1)
            for cam2_id, cam2 in self['camera'].items():
                if cam1_id == cam2_id or camera_overlap_graph.has_edge(cam1_id, cam2_id):
                    continue
                overlapped_obj_ids = set(cam1['object'].keys()) & set(cam2['object'].keys())
                if overlapped_obj_ids:
                    camera_overlap_graph.add_edge(cam1_id, cam2_id, overlapped_obj_ids=overlapped_obj_ids)
        return camera_overlap_graph

    @property
    def object_association_graph(self):
        # generate object association graph
        object_association_graph = nx.Graph()
        if 'affinity' in self:
            obj_id2idx = {
                cam_id: {obj_id: idx for idx, obj_id in enumerate(cam['object'].keys())}
                for cam_id, cam in self['camera'].items()
            }
        for camera_id, camera in self['camera'].items():
            for obj_id in camera['object']:
                new_node = (camera_id, obj_id)
                object_association_graph.add_node(new_node)
                for node in object_association_graph.nodes:
                    if node[0] != camera_id and node[1] == obj_id:
                        attr = {
                            'affinity': float(self['affinity'][(node[0], new_node[0])][
                                obj_id2idx[node[0]][node[1]], obj_id2idx[new_node[0]][new_node[1]]])
                        } if 'affinity' in self else {}
                        object_association_graph.add_edge(node, new_node, **attr)
        return object_association_graph

    def subscene(self, camera_ids, relabel_camera=False):
        subscene = self.copy()
        subscene['camera'] = {new_id if relabel_camera else i: subscene['camera'][i] for new_id, i in enumerate(camera_ids)}
        if 'object' in subscene:
            obj_ids = {
                obj_id
                for camera in subscene.get('camera', {}).values()
                for obj_id in camera.get('object', {})
            }
            subscene['object'] = {obj_id: subscene['object'][obj_id] for obj_id in obj_ids}
        if self.get('origin_camera_id', None) not in camera_ids:
            subscene.pop('origin_camera_id', None)
        if 'affinity' in self:
            subscene['affinity'] = subscene['affinity'].subscene(camera_ids)
        return subscene
    
    def random_subscene(self, n_camera, min_overlap_object=1, relabel_camera=False, min_object=0):
        assert n_camera > 0, f"n_camera should be positive, but got {n_camera}"
        if n_camera == 1:
            return self.subscene([random.choice(list(self['camera'].keys()))], relabel_camera=relabel_camera)

        camera_proposals = [cam_id for cam_id, cam in self['camera'].items() if len(cam['object']) >= min_object]
        if not camera_proposals:
            return self.subscene(camera_proposals, relabel_camera=relabel_camera)

        assert min_overlap_object >= 0, f"min_overlap_object should be non-negative, but got {min_overlap_object}"
        if min_overlap_object == 0:
            random.shuffle(camera_proposals)
            camera_proposals = camera_proposals[:n_camera]
            return self.subscene(camera_proposals, relabel_camera=relabel_camera)

        # disconnect cameras in camera_overlap_graph with low overlap
        camera_overlap_graph = self.camera_overlap_graph.subgraph(camera_proposals).copy()
        to_remove = []
        for edge, edge_attr in camera_overlap_graph.edges.items():
            if len(edge_attr['overlapped_obj_ids']) < min_overlap_object:
                to_remove.append(edge)
        camera_overlap_graph.remove_edges_from(to_remove)
        
        # find connected components of camera_overlap_graph
        connected_components = nx.connected_components(camera_overlap_graph)
        connected_components = sorted(connected_components, key=lambda x: len(x), reverse=True)
        
        # randomly choose the largest connected component if it's the best we can get
        if len(connected_components[0]) <= n_camera:
            node_proposals = [c for c in connected_components if len(c) == len(connected_components[0])]
            node_proposal = random.choice(node_proposals)
            return self.subscene(node_proposal, relabel_camera=relabel_camera)
        
        # randomly choose a connected component with more than n_camera cameras
        connected_components = [c for c in connected_components if len(c) > n_camera]
        num_nodes = [len(c) for c in connected_components]
        probabilities = [n / sum(num_nodes) for n in num_nodes]
        node_proposal = np.random.choice(connected_components, p=probabilities)
        connected_component = camera_overlap_graph.subgraph(node_proposal)
        
        # get camera proposals in the connected_component with n_camera cameras
        camera_proposals = sample_connected_component_of_camera_overlap_graph(connected_component, n_camera)
        
        return self.subscene(camera_proposals, relabel_camera=relabel_camera)
    
    def set_world_coordinate_frame(self, origin_camera_id=None):
        try:
            next(iter(self['camera'].values()))['position']
        except Exception:
            self['origin_camera_id'] = origin_camera_id
            return

        if origin_camera_id is None:
            # get the average of camera positions
            world_center = self.backend.stack([camera['position'] for camera in self['camera'].values()]).mean(axis=0)
            # find the camera closest to the average
            origin_camera_id = min(self['camera'].keys(), key=lambda x: self.backend.linalg.norm(self['camera'][x]['position'] - world_center))
        self['origin_camera_id'] = origin_camera_id
        origin_camera = self['camera'][origin_camera_id]

        # set the origin of the world coordinate frame to the origin camera
        translation = -origin_camera['position']
        for camera in self['camera'].values():
            camera['position'] = camera['position'] + translation
        if 'object' in self:
            for obj in self['object'].values():
                obj['center'] = obj['center'] + translation

        # set a new world coordinate frame like total3d
        old2new_mat = get_backend(origin_camera['rotation_mat']).eye_on(
            4, dtype=origin_camera['rotation_mat'].dtype, device=getattr(origin_camera['rotation_mat'], 'device', None))
        old2new_mat[:3, :3] = total3d_world_rotation(origin_camera['rotation_mat'])

        # get the rotation of the origin camera around Z axis
        for camera in self['camera'].values():
            camera['cam2world_mat'] = old2new_mat @ camera['cam2world_mat']
        if 'object' in self:
            for obj in self['object'].values():
                obj['local2world_mat'] = old2new_mat @ obj['local2world_mat']

        return origin_camera_id

    def remove_small_object2d(self, min_area=1.):
        # remove small objects in each camera
        visible_obj = set()
        for cam_id, camera in self['camera'].items():
            if 'object' not in camera:
                continue
            frame_area = camera['height'] * camera['width']
            new_objs = {}
            for obj_id, obj in camera['object'].items():
                if obj['area'] / frame_area * 100 >= min_area:
                    new_objs[obj_id] = obj
                    visible_obj.add(obj_id)
            camera['object'] = new_objs

        # remove 3D objects that are not visible in any camera
        if 'object' in self:
            self['object'] = {obj_id: obj for obj_id, obj in self['object'].items() if obj_id in visible_obj}

    def proximity_check_camera(self, min_depth=1.):
        if not min_depth:
            return

        # remove cameras with too small depth
        new_cameras = {}
        for cam_id, camera in self['camera'].items():
            if 'depth' not in camera['image']:
                return
            if camera['image']['depth'].min() >= min_depth:
                new_cameras[cam_id] = camera
        self['camera'] = new_cameras

        # remove 3D objects that are not visible in any camera
        visible_obj = set(obj_id for camera in self['camera'].values() for obj_id in camera['object'])
        if 'object' in self:
            self['object'] = {obj_id: obj for obj_id, obj in self['object'].items() if obj_id in visible_obj}

    def data_dict(self, keys=None, **kwargs):
        if keys is None:
            keys = ['uid', 'camera', 'object']
        return super().data_dict(keys, **kwargs)

    def enable_optimization(self, lr_config):
        params = []
        for lr_name, lr in lr_config.items():
            if not lr:
                continue
            prefix = lr_name.split('_')[0]
            p = []
            for obj_id, obj in self[prefix].items():
                if prefix == 'camera' and obj_id == self.get('origin_camera_id', None):
                    continue
                p.append(obj[lr_name[len(prefix)+1:]].requires_grad_())
            if p:
                params.append({'params': p, 'lr': lr, 'name': lr_name})
        self.optimize_params = [v['name'] for v in params]
        self.step()
        return params

    def step(self):
        # update attributes referencing the optimized parameters
        for param_name in self.optimize_params:
            prefix = param_name.split('_')[0]
            for obj_id, obj in self[prefix].items():
                if prefix == 'camera' and obj_id == self.get('origin_camera_id', None):
                    continue
                k = param_name[len(prefix)+1:]
                if k in obj.ref_graph:
                    obj.update_predecessors(k)

    def add_noise_to_3dscene(self, noise_std_config):
        for noise_name, noise_std in noise_std_config.items():
            if not noise_std:
                continue
            if noise_name.startswith('object'):
                for object3d in self['object'].values():
                    if noise_name == 'object_rotation':
                        rotation_noise = Rotation.from_euler(
                            'Z', np.random.normal() * noise_std, degrees=True).as_matrix().astype(object3d['rotation_mat'].dtype)
                        object3d['rotation_mat'] = rotation_noise @ object3d['rotation_mat']
                    elif noise_name == 'object_center':
                        object3d['center'] = object3d['center'] + np.random.normal(
                            size=object3d['center'].shape).astype(object3d['center'].dtype) * noise_std
                    elif noise_name == 'object_size':
                        object3d['size'] = object3d['size'] * (np.random.normal(
                            size=object3d['size'].shape).astype(object3d['size'].dtype) * noise_std + 1)
                    else:
                        raise ValueError(f'Unknown noise {noise_name}')
            elif noise_name.startswith('camera'):
                for camera in self['camera'].values():
                    if noise_name == 'camera_rotation':
                        rotation_noise = Rotation.from_euler('XYZ', np.random.normal(
                            size=3) * noise_std, degrees=True).as_matrix().astype(camera['rotation_mat'].dtype)
                        camera['rotation_mat'] = rotation_noise @ camera['rotation_mat']
                    elif noise_name == 'camera_position':
                        camera['position'] = camera['position'] + np.random.normal(
                            size=camera['position'].shape).astype(camera['position'].dtype) * noise_std
                    else:
                        raise ValueError(f'Unknown noise {noise_name}')
            elif noise_name == 'depth_scale':
                for camera in self['camera'].values():
                    camera['image']['depth'] = camera['image']['depth'] * (np.random.normal() * noise_std + 1)
            else:
                raise ValueError(f'Unknown noise {noise_name}')

    def crop_object_images(self):
        for cam_id, camera in self['camera'].items():
            if 'object' not in camera:
                continue
            for obj in camera['object'].values():
                obj['camera'] = crop_camera(camera, obj)

    def aggregate_object_images(self):
        for obj_id, obj_3d in self['object'].items():
            objnerf = ObjectNeRF(obj_3d, backend=self.backend, device=self.device)
            objnerf['camera'] = {}
            self['object'][obj_id] = objnerf

        for camera_id, camera in self['camera'].items():
            # crop image in each camera
            for obj_id, obj_2d in camera['object'].items():
                self['object'][obj_id]['camera'][camera_id] = obj_2d['camera']

    def zoom_out_bdb2d(self, ratio):
        for cam_id, camera in self['camera'].items():
            if 'object' not in camera:
                continue
            for obj_id, obj in camera['object'].items():
                bdb2d_center = obj['bdb2d'][:2] + obj['bdb2d'][2:] / 2
                bdb2d_wh = (obj['bdb2d'][2:] * ratio)
                bdb2d_xy1 = np.clip(bdb2d_center - bdb2d_wh / 2, 0, None)
                bdb2d_xy2 = np.clip(bdb2d_xy1 + bdb2d_wh, None, np.array([camera['width'], camera['height']]))
                bdb2d_wh = bdb2d_xy2 - bdb2d_xy1
                obj['bdb2d'] = np.concatenate([bdb2d_xy1, bdb2d_wh]).astype(int)

    def add_noise_to_bdb2d(self, noise_std):
        for cam_id, camera in self['camera'].items():
            for obj_id, obj in camera['object'].items():
                bdb2d_center = obj['bdb2d'][:2] + obj['bdb2d'][2:] / 2
                bdb2d_xy1 = bdb2d_center - obj['bdb2d'][2:] / 2
                bdb2d_xy1 = bdb2d_xy1 + np.random.normal(size=2) * obj['bdb2d'][2:] * noise_std
                bdb2d_xy1 = np.clip(bdb2d_xy1, 0, None)
                bdb2d_xy2 = bdb2d_xy1 + obj['bdb2d'][2:]
                bdb2d_xy2 = bdb2d_xy2 + np.random.normal(size=2) * obj['bdb2d'][2:] * noise_std
                bdb2d_xy2 = np.clip(bdb2d_xy2, None, np.array([camera['width'], camera['height']]))
                bdb2d_wh = bdb2d_xy2 - bdb2d_xy1
                bdb2d_wh = np.clip(bdb2d_wh, 1, None)
                obj['bdb2d'] = np.concatenate([bdb2d_xy1, bdb2d_wh]).astype(int)


class ObjectNeRF(Object3D):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        if 'camera' in self:
            self['camera'] = {k: v if isinstance(v, Camera) else Camera(v, *args, **kwargs) for k, v in self['camera'].items()}

    @classmethod
    def from_dir(cls, dir):
        CategoryMapping.load_category_mapping(os.path.dirname(os.path.normpath(dir)))

        # load pickle file
        pickle_dir = os.path.join(dir, 'data.pkl')
        with open(pickle_dir, 'rb') as f:
            data = pickle.load(f)
        
        # generate image file dirs
        for camera_id, camera in data['camera'].items():
            camera['image'] = {k: os.path.join(dir, f"{camera_id:04d}-{k}.png") for k in ('color', 'instance_segmap')}
        
        return cls(data)

    def save(self, dir):
        os.makedirs(dir, exist_ok=True)
        
        # save images
        self.save_images(dir)

        # write data as pickle
        with open(os.path.join(dir, 'data.pkl'), 'wb') as f:
            pickle.dump(self.pickle_dict(), f)

    def save_images(self, dir, key=None):
        for camera_id, camera in self['camera'].items():
            camera['image'].save(dir, camera_id=camera_id, key=key)

    def pickle_dict(self):
        pickle_dict = super().pickle_dict()
        if 'camera' in pickle_dict:
            pickle_dict['camera'] = {k: v.pickle_dict() for k, v in pickle_dict['camera'].items()}
        return pickle_dict
    
    def subset(self, camera_ids, relabel_camera=True):
        subset = self.copy()
        subset['camera'] = {new_id if relabel_camera else i: self['camera'][i] for new_id, i in enumerate(camera_ids)}
        return subset
    
    def random_subset(self, n_camera, relabel_camera=True):
        n_camera = min(n_camera, len(self['camera']))
        camera_ids = random.sample(list(self['camera'].keys()), n_camera)
        return self.subset(camera_ids, relabel_camera=relabel_camera)


def object3d_center_from_prediction(object2d, camera):
    center_2d = (object2d['offset'] * object2d['bdb2d'][2:]) + object2d['bdb2d'][:2] + object2d['bdb2d'][2:] / 2
    return uv2cam(camera['K'], center_2d + 0.5, object2d['center_depth'])


def object3d_from_prediction(object2d, camera):
    object3d = {
        'category': object2d['category'],
        'size': object2d['size'],
        'center': homotrans(camera['cam2world_mat'], object3d_center_from_prediction(object2d, camera))
    }

    if camera.backend is np:
        cam_rotation = WORLD_FRAME_TO_TOTAL3D @ camera['rotation_mat'] @ CAMERA_FRAME_TO_TOTAL3D.T
        cam_yaw = Rotation.from_matrix(cam_rotation).as_euler('xzy')[-1]
    else:
        import pytorch3d
        cam_rotation = camera.backend.tensor(WORLD_FRAME_TO_TOTAL3D, device=camera.device) @ camera['rotation_mat'] \
            @ camera.backend.tensor(CAMERA_FRAME_TO_TOTAL3D.T, device=camera.device)
        cam_yaw = -pytorch3d.transforms.matrix_to_euler_angles(cam_rotation.T, 'XZY')[-1]

    if 'orientation_score' in object2d:
        # only support numpy backend for orientation score interpolation
        cam_yaw = cam_yaw if camera.backend is np else cam_yaw.detach().cpu().item()
        # orientation is not able to backpropagate through score
        # interpolate orientation score with rotated orientation bins
        orientation_bin_center = (object2d.orientation_bin[:-1] + object2d.orientation_bin[1:]) / 2
        object3d['orientation_score'] = np.interp(
            orientation_bin_center - cam_yaw,
            orientation_bin_center,
            object2d.numpy()['orientation_score'],
            period=2 * np.pi
        )
    else:
        object3d['orientation'] = object2d['orientation'] + cam_yaw
        if camera.backend is np:
            object3d['orientation'] = np.array(object3d['orientation'])

    for key in ('score', 'embedding'):
        if key in object2d:
            object3d[key] = object2d[key]
    return Object3D(object3d, object2d.backend, object2d.device)


def parameterize_object2d(scene, target_scene=None):
    target_scene = target_scene or scene
    for cam_id in scene['camera'].keys():
        total3d_scene = scene.subscene([cam_id])
        total3d_scene.set_world_coordinate_frame(cam_id)
        total3d_cam = total3d_scene['camera'][cam_id]
        for obj_id, obj2d in target_scene['camera'][cam_id]['object'].items():
            total3d_obj2d = total3d_cam['object'][obj_id]
            total3d_obj3d = total3d_scene['object'][obj_id]
            obj2d['orientation'] = total3d_obj3d['orientation']
            if total3d_obj3d.backend is np:
                obj2d['orientation'] = np.array(obj2d['orientation'])
            center_in_cam = homotrans(total3d_cam['world2cam_mat'], total3d_obj3d['center'])
            obj2d['center_depth'] = -center_in_cam[2]
            obj2d['size'] = total3d_obj3d['size']
            center = homotrans(total3d_cam['world2cam_mat'], total3d_obj3d['center'])
            center_2d = cam2uv(total3d_cam['K'], center) - 0.5
            obj2d['offset'] = (center_2d - (total3d_obj2d['bdb2d'][:2] + total3d_obj2d['bdb2d'][2:] / 2)) / total3d_obj2d['bdb2d'][2:]


def parameterize_camera(camera):
    if camera.backend is np:
        cam_rotation = WORLD_FRAME_TO_TOTAL3D @ camera['rotation_mat'] @ CAMERA_FRAME_TO_TOTAL3D.T
        camera['roll'], camera['pitch'], camera['yaw'] = Rotation.from_matrix(cam_rotation).as_euler('xzy')
    else:
        import pytorch3d
        cam_rotation = camera.backend.tensor(WORLD_FRAME_TO_TOTAL3D, device=camera.device) @ camera['rotation_mat'] \
            @ camera.backend.tensor(CAMERA_FRAME_TO_TOTAL3D.T, device=camera.device)
        camera['roll'], camera['pitch'], camera['yaw'] = -pytorch3d.transforms.matrix_to_euler_angles(cam_rotation.T, 'XZY')

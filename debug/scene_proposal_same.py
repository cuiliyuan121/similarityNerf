import wandb
from .camera_pose_proposal import CameraPoseProposalDataset, CameraPoseProposal
from .object_proposal import ObjectProposalDataset, ObjectProposal, affinity_from_id_for_scene
from utils.general_utils import recursive_merge_dict
from utils.dataset import Scene, Object3D, object3d_center_from_prediction, object3d_from_prediction, parameterize_camera
from utils.visualize_utils import SceneVisualizer, image_grid
import tempfile
from utils.transform import WORLD_FRAME_TO_TOTAL3D, CAMERA_FRAME_TO_TOTAL3D, uv2cam, cam2uv, homotrans, bdb3d_corners, rotation_mat_dist
from utils.metrics import seg_iou, bdb3d_iou
from scipy.spatial.transform import Rotation
import numpy as np
from tqdm import tqdm
import torch
import copy
import networkx as nx
from utils.torch_utils import LossModule
from collections import defaultdict
import torchmetrics
from .base import SceneEstimationModule


class SceneProposalDataset(ObjectProposalDataset, CameraPoseProposalDataset):
    def get_data(self, input_scene, gt_scene):
        # add color image to input_data for visualization, and depth image for saving results
        input_data = input_scene.data_dict(
            keys=['origin_camera_id', 'affinity', 'split'],
            camera={'keys': 'all', 'image': {'keys': ['color', 'depth']}},
            object={'keys': {'rotation_mat', 'center', 'category_id', 'size', 'category_onehot', 'score'}}  # for evaluating saved results
        )

        # remove novel views from gt_scene when using gt as input
        gt_subscene = None if gt_scene is None else gt_scene.subscene(camera_ids=input_data['camera'].keys())

        # add object pose proposal GTs from Front3DObjectDataset
        input_data_obj, gt_data = ObjectProposalDataset.get_data(self, input_scene, gt_subscene)
        input_data = recursive_merge_dict(input_data, input_data_obj)
        if self.config['gt_affinity'] or self.config['gt_object_pose']:
            if self.config['gt_object_pose']:
                gt_data_obj = gt_subscene.data_dict(keys=[], camera={'keys': [], 'object': {'keys': 'all'}})
                if not self.config['gt_affinity']:
                    gt_data_obj = recursive_merge_dict(gt_data_obj, ObjectProposalDataset.get_data(self, gt_subscene, gt_subscene)[0])
            else:
                gt_data_obj = ObjectProposalDataset.get_data(self, gt_subscene, gt_subscene)[0]
            for cam in input_data['camera'].values():
                cam.pop('object', None)  # update detection results
            input_data = recursive_merge_dict(input_data, gt_data_obj)
            for cam in input_data['camera'].values():
                for obj in cam['object'].values():
                    obj['score'] = 1.  # set score to 1 for NMS and evaluation
        if self.config['gt_affinity']:
            input_data['affinity'] = gt_data['affinity'].copy()

        # add camera pose proposal GTs from Front3DCameraDataset
        input_data_cam, gt_data_cam = CameraPoseProposalDataset.get_data(self, input_scene, gt_subscene)
        assert not (self.config['gt_camera_pose'] and (self.config['gt_camera_yaw'] or self.config['gt_pitch_roll'])), \
            'gt_camera_pose and gt_camera_yaw/gt_pitch_roll cannot be both True'
        if self.config['gt_camera_pose']:
            for i_cam, cam in input_data_cam['camera'].items():
                cam['cam2world_mat'] = gt_scene['camera'][i_cam]['cam2world_mat'].copy()
        if self.config['gt_pitch_roll']:
            input_data_cam['camera'] = recursive_merge_dict(input_data_cam['camera'], gt_data_cam['camera'])
        if self.config['gt_camera_yaw']:
            input_data_cam['relative_yaw'] = gt_data_cam['relative_yaw'].copy()
        input_data = recursive_merge_dict(input_data, input_data_cam)
        gt_data = recursive_merge_dict(gt_data, gt_data_cam)

        # add camera and object pose GTs
        gt_data_pose = gt_scene.data_dict(keys=['camera', 'object', 'origin_camera_id', 'split'])
        gt_data = recursive_merge_dict(gt_data, gt_data_pose)

        return input_data, gt_data


def set_camera_as_origin(camera):
    if 'roll' not in camera:
        parameterize_camera(camera)
    rotation_mat = Rotation.from_euler('xzy', [camera['roll'], camera['pitch'], 0.]).as_matrix()
    camera['rotation_mat'] = WORLD_FRAME_TO_TOTAL3D.T @ rotation_mat @ CAMERA_FRAME_TO_TOTAL3D
    camera['position'] = np.zeros(3, dtype=np.float32)
    camera['yaw'] = 0.


def camera_proposal_from_object(object3d, object2d, camera, cam_yaw=None):
    camera = camera.copy()
    if cam_yaw is None:
        obj_yaw = Rotation.from_matrix(object3d['rotation_mat']).as_euler('zxy')[0]
        cam_yaw = obj_yaw - object2d['orientation']
    camera['yaw'] = cam_yaw
    cam2world_mat = np.eye(4, dtype=np.float32)
    cam_rotation_mat = Rotation.from_euler('xzy', [camera['roll'], camera['pitch'], camera['yaw']]).as_matrix()
    cam_rotation_mat = WORLD_FRAME_TO_TOTAL3D.T @ cam_rotation_mat @ CAMERA_FRAME_TO_TOTAL3D
    cam2world_mat[:3, :3] = cam_rotation_mat
    cam2world_mat[:3, 3] = object3d['center'] - cam_rotation_mat @ object3d_center_from_prediction(object2d, camera)
    camera['cam2world_mat'] = cam2world_mat
    return camera


def camera_affinity_score(affinity_mat):
    grid = np.indices(affinity_mat.shape)
    indices = grid.reshape(2, -1).T
    affinity = affinity_mat.reshape(-1)
    order = affinity.argsort()[::-1]
    indices = indices[order]
    affinity = affinity[order]

    matched_cam1 = set()
    matched_cam2 = set()
    total = 0.
    for (i, j), score in zip(indices, affinity):
        if i not in matched_cam1 and j not in matched_cam2:
            matched_cam1.add(i)
            matched_cam2.add(j)
            total += score
        if len(matched_cam1) >= affinity_mat.shape[0] or len(matched_cam2) >= affinity_mat.shape[1]:
            break
    return total


def bdb3d_proj_loss(obj3d, obj2d, cam):
    corners = bdb3d_corners(obj3d)
    corners = homotrans(cam['world2cam_mat'], corners)
    corners_2d = cam2uv(cam['K'], corners) - 0.5
    est_bdb2d = np.concatenate([corners_2d.min(0), corners_2d.max(0)])
    est_bdb2d[[0, 2]] = est_bdb2d[[0, 2]].clip(0, cam['width'] - 1)
    est_bdb2d[[1, 3]] = est_bdb2d[[1, 3]].clip(0, cam['height'] - 1)
    gt_bdb2d = obj2d['bdb2d'].copy()
    gt_bdb2d[2:] += obj2d['bdb2d'][:2]
    est_bdb2d[:2] = est_bdb2d[:2].clip(min=gt_bdb2d[:2])
    est_bdb2d[2:] = est_bdb2d[2:].clip(max=gt_bdb2d[2:])
    return np.abs(est_bdb2d / cam['width'] - gt_bdb2d / cam['width']).mean()


class Object3DConfidenceLoss(LossModule):
    def __init__(self, func=None, weight=None, **kwargs):
        for key in ('bdb3d_proj', 'score', 'unmatched_camera'):
            assert key in weight
        super().__init__(func, weight, **kwargs)

    def compute_loss(self, obj3d, obj_id, cam_list):
        loss = 0.

        if self.weight['bdb3d_proj'] is not None:
            bdb3d_proj_losses = [
                bdb3d_proj_loss(obj3d, cam['object'][obj_id], cam)
                for cam in cam_list
                if obj_id in cam['object']
            ]
            subloss = np.mean(bdb3d_proj_losses)
            self.metrics['bdb3d_proj_loss'].update(subloss)
        if self.weight['bdb3d_proj']:
            loss += self.weight['bdb3d_proj'] * subloss

        if self.weight['score'] is not None:
            subloss = 1. - obj3d['score']
            self.metrics['score_loss'].update(subloss)
        if self.weight['score']:
            loss += self.weight['score'] * subloss

        if self.weight['unmatched_camera'] is not None:
            subloss = sum(1. for cam in cam_list if obj_id not in cam['object'])
            self.metrics['unmatched_camera_loss'].update(subloss)
        if self.weight['unmatched_camera']:
            loss += self.weight['unmatched_camera'] * subloss

        if self.weight['bdb2d_size'] is not None:
            cam_diag = np.linalg.norm(np.array([cam_list[0]['width'], cam_list[0]['height']]))
            bdb2d_size_losses = [
                (cam_diag - np.linalg.norm(cam['object'][obj_id]['bdb2d'][2:])) / cam_diag
                for cam in cam_list
                if obj_id in cam['object']
            ]
            subloss = np.mean(bdb2d_size_losses)
            self.metrics['bdb2d_size_loss'].update(subloss)
        if self.weight['bdb2d_size']:
            loss += self.weight['bdb2d_size'] * subloss

        return loss


class SceneProposalLoss(LossModule):
    def __init__(self, func=None, weight=None, **kwargs):
        for key in ('affinity', 'bdb3d_iou', 'unmatched_object', 'bdb3d_proj', 'object_size', 'object_center', 'object_rotation'):
            assert key in weight
        super().__init__(func, weight, **kwargs)
        self.metrics['association_loss'] = torchmetrics.MeanMetric()

    def loss_between_object3d(self, obj3d1, obj_id1, cam_list1, obj3d2, obj_id2, cam_list2, affinity_loss=None):
        loss_dict = {
            'affinity_loss': affinity_loss,
            'bdb3d_iou_loss': 1. - bdb3d_iou(obj3d1, obj3d2),
            'object_size_loss': (np.abs(obj3d1['size'] - obj3d2['size']) / obj3d2['size']).mean(),
            'object_center_loss': np.linalg.norm(obj3d1['center'] - obj3d2['center']) / np.linalg.norm(obj3d2['size'] / 2),
            'object_rotation_loss': rotation_mat_dist(obj3d1['rotation_mat'], obj3d2['rotation_mat']),
        }

        bdb3d_proj_losses = []
        for cam in cam_list1:
            if obj_id1 in cam['object']:
                bdb3d_proj_losses.append(bdb3d_proj_loss(obj3d1, cam['object'][obj_id1], cam))
        for cam in cam_list2:
            if obj_id2 in cam['object']:
                bdb3d_proj_losses.append(bdb3d_proj_loss(obj3d1, cam['object'][obj_id2], cam))
        loss_dict['bdb3d_proj_loss'] = np.mean(bdb3d_proj_losses)

        loss_dict['association_loss'] = sum((self.weight.get(k[:-len('_loss')], 0) or 0.) * v for k, v in loss_dict.items())
        return loss_dict

    def compute_loss(self, association_graph, scene, new_cam, new_cam_id, topk_cam_ids=None, anchor_edge=None):
        loss = 0.

        # unmatched_object loss
        if self.weight['unmatched_object'] is not None:
            num_obj3d = sum(axis == 0 for axis, _ in association_graph.nodes)
            min_axis = min(num_obj3d, len(association_graph) - num_obj3d)
            unmatched_object_loss = (min_axis - len(association_graph.edges)) / min_axis
            self.metrics['unmatched_object_loss'].update(unmatched_object_loss)
        if self.weight['unmatched_object']:
            loss += self.weight['unmatched_object'] * unmatched_object_loss

        # relative_yaw loss
        if 'relative_yaw_prob' in scene and topk_cam_ids is not None:  # if no relative_yaw_prob, gt is used
            if self.weight['relative_yaw'] is not None:
                relative_yaw_loss = 0.
                for cam_id in topk_cam_ids:
                    relative_yaw_prob = np.interp(
                        new_cam['yaw'],
                        scene['camera_graph'][new_cam_id][cam_id]['relative_yaw_query'],
                        scene['camera_graph'][new_cam_id][cam_id]['relative_yaw_prob'],
                        period=2 * np.pi
                    )
                    relative_yaw_loss += 1. - relative_yaw_prob
                self.metrics['relative_yaw_loss'].update(relative_yaw_loss)
            if self.weight['relative_yaw']:
                loss += self.weight['relative_yaw'] * relative_yaw_loss

        # edge loss
        for key in next(iter(association_graph.edges(data=True)))[2].keys():
            key = key[:-len('_loss')]
            if key not in self.weight:
                continue
            if self.weight[key] is not None:
                loss_list = [
                    l for e, l in nx.get_edge_attributes(association_graph, f"{key}_loss").items()
                    if key == 'affinity' or e != anchor_edge  # ignore other loss for anchor edge except affinity loss
                ]
                if loss_list:
                    sub_loss = np.mean(loss_list)
                    self.metrics[f"{key}_loss"].update(sub_loss, weight=len(loss_list))
                    if self.weight[key]:
                        loss += self.weight[key] * sub_loss
        self.metrics['association_loss'].update(np.mean(list(nx.get_edge_attributes(association_graph, 'association_loss').values())))

        return loss


class SceneProposal(SceneEstimationModule):
    dataset_cls = SceneProposalDataset

    def __init__(self, camera_pose_proposal, object_proposal, loss, object3d_confidence, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # load proposal models
        if checkpoint_dir := camera_pose_proposal.pop('checkpoint'):
            self.camera_pose_proposal = CameraPoseProposal.load_from_checkpoint(checkpoint_dir, **camera_pose_proposal)
        if checkpoint_dir := object_proposal.pop('checkpoint'):
            self.object_proposal = ObjectProposal.load_from_checkpoint(checkpoint_dir, **object_proposal)
        self.loss = SceneProposalLoss(**loss)
        self.object3d_confidence_loss = Object3DConfidenceLoss(**object3d_confidence)
        self.scene_optimization = None

        # set input images for dataset to be the same as object proposal
        self.config['dataset']['input_images'] = {
            'obj_prop_input': self.object_proposal.hparams.input_images if hasattr(self, 'object_proposal') else None,
            'cam_prop_input': self.camera_pose_proposal.hparams.input_images if hasattr(self, 'camera_pose_proposal') else None,
        }

    def camera_object_proposal(self, est_scene, gt_scene, get_pitch_roll, get_camera_yaw, get_object_pose, get_embedding):
        if hasattr(self, 'camera_pose_proposal'):
            self.camera_pose_proposal(est_scene, get_pose=get_pitch_roll, get_relative_yaw=get_camera_yaw)
        if hasattr(self, 'object_proposal'):
            self.object_proposal(est_scene, get_pose=get_object_pose, get_embedding=get_embedding)

        if est_scene['if_log']:
            gt_scene_np, est_scene_np = gt_scene.numpy(), est_scene.numpy()
            for camera_id in est_scene_np['camera']:
                log_dict = {"global_step": est_scene['batch_idx'], "test/camera_id": camera_id}

                # visualize gt scene
                gt_subscene = gt_scene_np.subscene([camera_id], relabel_camera=True)
                gt_subscene.set_world_coordinate_frame(0)
                gt_subscene_vis = SceneVisualizer(gt_subscene)

                # convert object2d to object3d
                # est_subscene = gt_scene_np.subscene([camera_id], relabel_camera=True)  # visualize gt parameters
                # for obj2d in est_subscene['camera'][0]['object'].values():
                #     del obj2d['orientation']
                #     del obj2d['size']
                est_subscene = est_scene_np.subscene([camera_id], relabel_camera=True)  # visualize est parameters
                camera = est_subscene['camera'][0]
                set_camera_as_origin(camera)
                est_subscene['object'] = {obj_id: object3d_from_prediction(obj, camera) for obj_id, obj in camera['object'].items()}
                est_subscene_vis = SceneVisualizer(est_subscene)
                est_subscene_fig = est_subscene_vis.scene_vis(reference_scene_vis=gt_subscene_vis)

                # visualize bdb3d proj
                gt_img = gt_subscene['camera'][0]['image']['color']
                gt_bdb3d_img = gt_subscene_vis.bdb3d(image=gt_img, camera_id=0)
                est_bdb3d_img = est_subscene_vis.bdb3d(image=gt_img, camera_id=0)
                combined_img = image_grid([est_bdb3d_img, gt_bdb3d_img], rows=1, padding=2, background_color=(128, 128, 128))

                # log to wandb
                log_dict['test/single_view/bdb3d'] = wandb.Image(combined_img, caption=f"{gt_scene_np['uid']} camera-{camera_id}")
                with tempfile.NamedTemporaryFile(mode='r+', suffix='.html') as f:
                    est_subscene_fig.write_html(f, auto_play=False)
                    log_dict["test/single_view/camera_scene_vis"] = wandb.Html(f)
                self.logger.experiment.log(log_dict)

    def nms2d(self, obj2d_dict):
        old_obj2d_list = sorted(obj2d_dict.items(), key=lambda x: x[1]['score'], reverse=True)
        new_obj2d_dict = {}
        while old_obj2d_list:
            obj_id, obj2d = old_obj2d_list.pop(0)
            new_obj2d_dict[obj_id] = obj2d
            old_obj2d_list = [x for x in old_obj2d_list if seg_iou(obj2d, x[1]) <= self.hparams.nms2d_thres]
        return new_obj2d_dict

    def nms3d(self, scene, strict=False):
        matched_cam = defaultdict(set)
        for cam_id, cam in scene['camera'].items():
            for obj_id in cam['object'].keys():
                matched_cam[obj_id].add(cam_id)

        obj_loss = {
            obj_id: self.object3d_confidence_loss(obj3d, obj_id, list(scene['camera'].values()))
            for obj_id, obj3d in scene['object'].items()
        }
        remain_obj3d_list = sorted(scene['object'].items(), key=lambda x: obj_loss[x[0]])
        new_obj3d_dict = {}
        association = {}
        while remain_obj3d_list:
            obj_id, obj3d = remain_obj3d_list.pop(0)
            new_obj3d_dict[obj_id] = obj3d

            old_remain_obj3d_list = []
            for old_obj3d_id, old_obj3d in remain_obj3d_list:
                if strict or (matched_cam[old_obj3d_id] & matched_cam[obj_id]):  # duplicate camera
                    iou = bdb3d_iou(old_obj3d, obj3d)
                    if_remove = iou > self.hparams.nms3d_thres
                    association_loss = 1 - iou
                else:
                    loss_dict = self.loss.loss_between_object3d(
                        old_obj3d, old_obj3d_id, list(scene['camera'].values()),
                        obj3d, obj_id, list(scene['camera'].values()),
                        affinity_loss=1 - self.object_proposal.object_association_proposal.affinity_mat(
                            torch.from_numpy(old_obj3d['embedding']), torch.from_numpy(obj3d['embedding'])).item()
                    )
                    if_remove = all(loss_dict[key] <= thres for key, thres in self.config['loss_thres'].items() if thres is not None)
                    association_loss = loss_dict['association_loss']

                if if_remove:
                    association[(old_obj3d_id, obj_id)] = association_loss
                else:
                    old_remain_obj3d_list.append((old_obj3d_id, old_obj3d))

            remain_obj3d_list = old_remain_obj3d_list

        association = sorted(association.keys(), key=lambda x: association[x])
        return new_obj3d_dict, association

    def all_association_proposal(self, affinity_mat, obj1_ids, obj2_ids):
        assert affinity_mat.shape == (len(obj1_ids), len(obj2_ids))

        # initialize graph with all nodes
        full_graph = nx.Graph()
        for axis, obj_ids in enumerate([obj1_ids, obj2_ids]):
            full_graph.add_nodes_from(zip([axis] * len(obj_ids), obj_ids))

        # generate affinity_mask from affinity_mat
        if self.config['dataset']['gt_affinity']:
            affinity_mask = affinity_mat
        else:
            # ensure full_graph has at least two edge without conflict
            # meaning the second largest affinity should be on different row or column of the largest affinity
            largest_affinity_idx = np.unravel_index(np.argmax(affinity_mat), affinity_mat.shape)
            remain_affinity = affinity_mat.copy()
            remain_affinity[largest_affinity_idx[0], :] = 0
            remain_affinity[:, largest_affinity_idx[1]] = 0
            second_largest_affinity = np.max(remain_affinity)
            # add edges with affinity score > threshold
            affinity_mask = affinity_mat > self.config['affinity_thres']
            affinity_mask |= affinity_mat >= second_largest_affinity

        # add edges to full_graph
        obj1_idx, obj2_idx = np.where(affinity_mask)
        node1 = [(0, obj1_ids[i]) for i in obj1_idx]
        node2 = [(1, obj2_ids[i]) for i in obj2_idx]
        attr = [{'affinity_loss': 1 - float(a)} for a in affinity_mat[affinity_mask]]
        full_graph.add_edges_from(zip(node1, node2, attr))
        return full_graph

    def best_association_graph(self, full_graph, scene, new_cam, new_cam_id, topk_cam_ids=None, anchor_edge=None):
        # calculate object IoU and other metrics for each association from obj3d to obj2d
        new_obj2d_ids = [obj_id for axis, obj_id in full_graph.nodes if axis == 1]
        new_obj3ds = {obj_id: object3d_from_prediction(new_cam['object'][obj_id], new_cam) for obj_id in new_obj2d_ids}
        prop_graph = full_graph.copy()

        # remove edges conflicting with anchor edge from prop_graph
        if anchor_edge is not None:
            prop_graph_temp = copy.deepcopy(prop_graph)
            for edge in prop_graph_temp.edges:
                if edge != anchor_edge and any(n in edge for n in anchor_edge):
                    prop_graph.remove_edge(*edge)

        # calculate loss for each association and remove impossible association
        impossible_association = {}
        for association in prop_graph.edges:
            (_, obj3d_id), (_, obj2d_id) = association
            old_obj3d = scene['object'][obj3d_id]
            new_obj3d = new_obj3ds[obj2d_id]

            loss_dict = self.loss.loss_between_object3d(
                old_obj3d, obj3d_id, [cam for cam_id, cam in scene['camera'].items() if cam_id != new_cam_id],
                new_obj3d, obj2d_id, [new_cam],
                affinity_loss=prop_graph.edges[association]['affinity_loss']
            )
            prop_graph.edges[association].update(loss_dict)

            # remove impossible association
            if (not self.config['dataset']['gt_affinity']) and (association != anchor_edge) and any(
                    loss_dict[key] > thres for key, thres in self.config['loss_thres'].items() if thres is not None):
                impossible_association[association] = loss_dict['association_loss']

        # remove impossible association, keep one best guess if all edges are impossible
        if impossible_association and len(impossible_association) >= len(prop_graph.edges) - 1:
            best_edge = min(impossible_association, key=impossible_association.get)
            del impossible_association[best_edge]
        prop_graph.remove_edges_from(impossible_association)

        # enumerate all possible association proposals and calculate their score
        prop_edges = [e for e in prop_graph.edges(data=True) if e[:2] != anchor_edge]
        prop_edges = sorted(prop_edges, key=lambda x: x[2]['association_loss'], reverse=False)
        prop_edges = prop_edges[:self.hparams.max_association_proposal]

        def add_edge_or_not(g, edge_idx):
            # if GT affinity is provided
            if self.config['dataset']['gt_affinity']:
                yield prop_graph
                return

            if edge_idx >= len(prop_edges):
                if len(g.edges) > (0 if anchor_edge is None else 1):
                    yield g
                return

            if all(g.degree(n) == 0 for n in prop_edges[edge_idx][:2]):
                # add edge
                g_add = g.copy()
                g_add.add_edge(*prop_edges[edge_idx][:2], **prop_edges[edge_idx][2])
                yield from add_edge_or_not(g_add, edge_idx + 1)

            # skip edge
            yield from add_edge_or_not(g, edge_idx + 1)

        # initialize worst case with only anchor edge
        init_graph = nx.create_empty_copy(prop_graph)
        if anchor_edge is None:
            best_association_graph = None
            min_loss = 10000.
        else:
            init_graph.add_edge(*anchor_edge, **prop_graph.edges[anchor_edge])
            best_association_graph = init_graph.copy()
            min_loss = 10000. + self.loss(init_graph, scene, new_cam, new_cam_id, topk_cam_ids, anchor_edge, log=False)

        # recursively add edge to init_graph and calculate loss
        for association_graph in add_edge_or_not(init_graph, 0):
            loss = self.loss(association_graph, scene, new_cam, new_cam_id, topk_cam_ids, anchor_edge, log=False)
            if loss < min_loss:
                min_loss = loss
                best_association_graph = association_graph

        return best_association_graph, min_loss

    def topk_camera_proposal_from_object(self, object3d, object2d, camera, cam_yaw_list=None):
        # if cam_yaw_list is None:
        #     cam_yaw_list = [None]
        # for cam_yaw in cam_yaw_list:
        #     yield camera_proposal_from_object(object3d, object2d, camera, cam_yaw)
        # if 'orientation_score' not in object2d or self.config['dataset']['gt_camera_yaw']:
        #     # no orientation score, return the only one proposal
        #     return

        prop_orientation_cls = np.argsort(-object2d['orientation_score'])[:self.hparams.num_orientation_proposal]
        for orientation_cls in prop_orientation_cls:
            prop_object2d = object2d.copy()
            prop_object2d['orientation_cls'] = orientation_cls
            yield camera_proposal_from_object(object3d, prop_object2d, camera)

    def pre_nms(self, est_scene_np, gt_scene_np):
        # visualize 2D detection before NMS
        if est_scene_np['if_log']:
            SceneVisualizer(est_scene_np).bdb2d()
            est_bdb2d_img = [cam['image']['bdb2d'] for cam in est_scene_np['camera'].values()]
            SceneVisualizer(gt_scene_np).bdb2d()
            gt_bdb2d_img = [gt_scene_np['camera'][cam_id]['image']['bdb2d'] for cam_id in est_scene_np['camera'].keys()]

        # do NMS on 2D detection
        for cam in est_scene_np['camera'].values():
            cam['object'] = self.nms2d(cam['object'])

        # visualize 2D detection after 2D NMS
        if est_scene_np['if_log']:
            SceneVisualizer(est_scene_np).bdb2d()
            nms2d_bdb2d_img = [cam['image']['bdb2d'] for cam in est_scene_np['camera'].values()]

        # do NMS on 3D object
        for cam_id in est_scene_np['camera'].keys():
            est_subscene = est_scene_np.subscene([cam_id])
            set_camera_as_origin(est_subscene['camera'][cam_id])
            est_subscene['object'] = {}
            for obj2d_id, obj2d in est_subscene['camera'][cam_id]['object'].items():
                est_subscene['object'][obj2d_id] = object3d_from_prediction(obj2d, est_subscene['camera'][cam_id])

            # visualize 3D scene before 3D NMS
            if est_scene_np['if_log']:
                est_subscene_vis = SceneVisualizer(est_subscene.copy())

            est_subscene['object'], _ = self.nms3d(est_subscene, strict=True)

            # visualize 3D scene after 3D NMS
            if est_scene_np['if_log']:
                nms_subscene_fig = SceneVisualizer(est_subscene).scene_vis(reference_scene_vis=est_subscene_vis)
                with tempfile.NamedTemporaryFile(mode='r+', suffix='.html') as f:
                    nms_subscene_fig.write_html(f, auto_play=False)
                    self.logger.experiment.log({'global_step': est_scene_np['batch_idx'], 'test/proposal/nms3d_camera_scene_vis': wandb.Html(f)})

            # update 3D object in est_scene_np
            est_scene_np['camera'][cam_id]['object'] = {obj_id: est_scene_np['camera'][cam_id]['object'][obj_id] for obj_id in est_subscene['object'].keys()}

        # visualize 2D detection after 3D NMS
        if est_scene_np['if_log']:
            SceneVisualizer(est_scene_np).bdb2d()
            nms3d_bdb2d_img = [cam['image']['bdb2d'] for cam in est_scene_np['camera'].values()]
            combined_img = image_grid([est_bdb2d_img, nms2d_bdb2d_img, nms3d_bdb2d_img, gt_bdb2d_img], background_color=(128, 128, 128))
            self.logger.experiment.log({'global_step': est_scene_np['batch_idx'], 'test/proposal/nms2d_nms3d': wandb.Image(combined_img)})

    def initialize_scene(self, est_scene_np):
        origin_camera_id = est_scene_np['origin_camera_id']
        best_scene_np = est_scene_np.subscene([origin_camera_id])
        best_scene_np['object'] = {}
        if not self.config['dataset']['gt_affinity']:
            # relabel object2d proposals
            best_scene_np['camera'][origin_camera_id]['object'] = dict(enumerate(best_scene_np['camera'][origin_camera_id]['object'].values()))
        set_camera_as_origin(best_scene_np['camera'][origin_camera_id])
        return best_scene_np

    def choose_camera_to_add(self, est_scene_np, remain_cam_ids):
        # generate affinity matrix between 3D scene and remaining cameras
        if self.config['dataset']['gt_affinity']:
            # use gt affinity matrix
            obj3d_ids = np.array(list(est_scene_np['object'].keys()))
            affinity_mats = {}
            for cam_id in remain_cam_ids:
                obj2d_ids = np.array(list(est_scene_np['camera'][cam_id]['object'].keys()))
                affinity_mats[cam_id] = np.repeat(obj3d_ids[:, np.newaxis], len(obj2d_ids), axis=1) == \
                    np.repeat(obj2d_ids[np.newaxis, :], len(obj3d_ids), axis=0)
        else:
            # use object embedding to generate affinity matrix
            obj3d_embedding = np.stack([o['embedding'] for o in est_scene_np['object'].values()])
            remain_cams = {cam_id: est_scene_np['camera'][cam_id] for cam_id in remain_cam_ids}
            obj2d_embeddings = {cam_id: np.stack([o['embedding'] for o in cam['object'].values()])
                                for cam_id, cam in remain_cams.items()}
            affinity_mats = {cam_id: self.object_proposal.object_association_proposal.affinity_mat(
                torch.from_numpy(obj3d_embedding), torch.from_numpy(obj2d_embedding)).numpy()
                        for cam_id, obj2d_embedding in obj2d_embeddings.items()}

        # choose the the camera with the highest affinity to add to 3D scene
        cam_scene_affinity = {cam_id: camera_affinity_score(affinity_mats[cam_id]) for cam_id in remain_cam_ids}
        best_cam_id = max(cam_scene_affinity, key=cam_scene_affinity.get)
        affinity_mat = affinity_mats[best_cam_id]
        return best_cam_id, affinity_mat

    def initialize_hypothesis_generation(self, est_scene_np, existing_cam_ids, best_cam_id, affinity_mat):
        # keep the order of object3d by manually selecting existing camera
        init_scene_np = est_scene_np.copy()
        init_scene_np['camera'] = {k: init_scene_np['camera'][k] for k in existing_cam_ids}
        init_cam = init_scene_np['camera'][best_cam_id] = est_scene_np['camera'][best_cam_id].copy()

        # update relative_yaw matrices
        relative_yaw_idx = np.array([list(est_scene_np['camera'].keys()).index(cam_id) for cam_id in init_scene_np['camera'].keys()])
        for k, v in init_scene_np.items():
            if k.startswith('relative_yaw'):
                init_scene_np[k] = v[relative_yaw_idx][:, relative_yaw_idx]
        init_scene_np.pop('camera_graph', None)

        obj3d_ids = list(init_scene_np['object'].keys())
        obj2d_ids = list(init_cam['object'].keys())
        full_graph = self.all_association_proposal(affinity_mat, obj3d_ids, obj2d_ids)

        return init_scene_np, init_cam, full_graph

    def update_obj2d_id(self, curr_cam, est_scene_np, association_graph=None):
        if self.config['dataset']['gt_affinity']:
            return

        # relabel objects based on object association
        new_obj2d = {}
        obj3d_id_begin = max(est_scene_np['object'].keys())
        if association_graph is None:
            for obj3d_id_offset, (obj2d_id, obj2d) in enumerate(curr_cam['object'].items()):
                new_obj2d[obj3d_id_begin + obj3d_id_offset + 1] = obj2d
        else:
            # relabel object2d proposals with correspondence
            for (_, obj3d_id), (_, obj2d_id) in association_graph.edges:
                new_obj2d[obj3d_id] = curr_cam['object'][obj2d_id]
            # relabel object2d proposals without correspondence
            for obj3d_id_offset, obj2d_id in enumerate(obj_id for axis, obj_id in nx.isolates(association_graph) if axis == 1):
                new_obj2d[obj3d_id_begin + obj3d_id_offset + 1] = curr_cam['object'][obj2d_id]
        curr_cam['object'] = new_obj2d

    def skip_proposal_if_use_gt_camera_pose(self, est_scene_np, remain_cam_ids, best_cam_id, affinity_mat):
        existing_cam_ids = set(est_scene_np['camera'].keys()) - remain_cam_ids | {best_cam_id}
        best_scene_np = est_scene_np.copy()
        best_scene_np['camera'] = {k: est_scene_np['camera'][k] for k in existing_cam_ids}
        self.update_obj2d_id(best_scene_np['camera'][best_cam_id], est_scene_np, None)
        return best_scene_np

    def propose_pose_and_association_for_new_camera(self, est_scene_np, remain_cam_ids, best_cam_id, affinity_mat):
        # choose the camera with the top-k highest affinity score in camera graph
        # and calculate camera yaw proposal from relative camera yaw
        existing_cam_ids = set(est_scene_np['camera'].keys()) - remain_cam_ids
        cam_affinity = {cam_id: camera_affinity_score(est_scene_np['affinity'][(best_cam_id, cam_id)]) for cam_id in existing_cam_ids}
        topk_cam_ids = [k for k, _ in sorted(cam_affinity.items(), key=lambda item: item[1], reverse=True)[:self.config['max_relative_yaw_proposal']]]
        topk_cam_yaw = [
            (est_scene_np['camera'][cam_id]['yaw'] + est_scene_np['camera_graph'][best_cam_id][cam_id]['relative_yaw'] + np.pi) % (2 * np.pi) - np.pi
            for cam_id in topk_cam_ids
        ]

        # initialize hypothesis
        min_loss = np.inf
        best_scene_np = None
        best_association_graph = None
        best_anchor_association = None
        init_scene_np, init_cam, full_graph = self.initialize_hypothesis_generation(est_scene_np, existing_cam_ids, best_cam_id, affinity_mat)

        # enumerate all possible object association between 3D scene and the chosen camera
        # to generate camera pose proposal
        anchor_associations = nx.get_edge_attributes(full_graph, 'affinity_loss')
        anchor_associations = sorted(anchor_associations.items(), key=lambda x: x[1], reverse=False)
        anchor_associations = [a[0] for a in anchor_associations]
        anchor_associations = anchor_associations[:self.hparams.max_camera_pose_proposal]
        for anchor_association in tqdm(anchor_associations, desc='Camera pose proposal', leave=False):
            (_, anchor_obj_id), (_, obj2d_id) = anchor_association

            # propose camera pose from corresponding object
            obj3d = init_scene_np['object'][anchor_obj_id]
            obj2d = init_cam['object'][obj2d_id]
            for curr_cam in self.topk_camera_proposal_from_object(obj3d, obj2d, init_cam, topk_cam_yaw):
                curr_scene_np = init_scene_np.copy()
                curr_scene_np['camera'][best_cam_id] = curr_cam

                # propose best object association based on existing object correspondence
                association_graph, loss = self.best_association_graph(
                        full_graph, curr_scene_np, curr_cam, best_cam_id, topk_cam_ids, anchor_association)
                self.update_obj2d_id(curr_cam, est_scene_np, association_graph)

                # cache the best camera proposal
                if loss < min_loss:
                    min_loss = loss
                    best_scene_np = curr_scene_np.copy()
                    best_association_graph = association_graph.copy()
                    best_anchor_association = anchor_association

        # log loss for the best association graph
        self.loss(best_association_graph, best_scene_np, best_scene_np['camera'][best_cam_id],
                  best_cam_id, topk_cam_ids, best_anchor_association)

        return best_scene_np

    def after_nms(self, best_scene_np):
        # visualize 3D scene before 3D NMS
        if best_scene_np['if_log']:
            best_scene_vis = SceneVisualizer(best_scene_np.copy())

        best_scene_np['object'], association = self.nms3d(best_scene_np, strict=True)

        # merge 3D object1 to 3D object2 if they are merged in 3D NMS
        matched_cam = defaultdict(set)
        for cam_id, cam in best_scene_np['camera'].items():
            for obj2d_id in cam['object'].keys():
                matched_cam[obj2d_id].add(cam_id)
        for obj3d_id1, obj3d_id2 in association:
            new_matched_cam = matched_cam[obj3d_id1] - matched_cam[obj3d_id2]
            dup_matched_cam = matched_cam[obj3d_id1] & matched_cam[obj3d_id2]
            # add new obj2d-obj3d correspondence
            for cam_id in new_matched_cam:
                best_scene_np['camera'][cam_id]['object'][obj3d_id2] = best_scene_np['camera'][cam_id]['object'].pop(obj3d_id1)
                matched_cam[obj3d_id2].add(cam_id)
            # remove the worst duplicated obj2d-obj3d correspondence
            for cam_id in dup_matched_cam:
                obj3d1_loss = self.object3d_confidence_loss(best_scene_np['object'][obj3d_id2], obj3d_id1, [best_scene_np['camera'][cam_id]])
                obj3d2_loss = self.object3d_confidence_loss(best_scene_np['object'][obj3d_id2], obj3d_id2, [best_scene_np['camera'][cam_id]])
                if obj3d1_loss > obj3d2_loss:
                    best_scene_np['camera'][cam_id]['object'].pop(obj3d_id1)
                else:
                    best_scene_np['camera'][cam_id]['object'][obj3d_id2] = best_scene_np['camera'][cam_id]['object'].pop(obj3d_id1)

        # visualize 3D scene after 3D NMS
        if best_scene_np['if_log']:
            nms_scene_fig = SceneVisualizer(best_scene_np).scene_vis(reference_scene_vis=best_scene_vis)
            with tempfile.NamedTemporaryFile(mode='r+', suffix='.html') as f:
                nms_scene_fig.write_html(f, auto_play=False)
                self.logger.experiment.log({'global_step': best_scene_np['batch_idx'], 'test/proposal/nms3d_scene_vis': wandb.Html(f)})

    def calculate_affinity_for_scene(self, scene):
        if not self.config['dataset']['gt_affinity']:
            self.object_proposal.object_association_proposal.calculate_affinity_for_scene(scene)
        else:
            affinity_from_id_for_scene(scene)

    def generate_scene(self, est_scene, gt_scene):

        est_scene_np = est_scene.numpy()
        gt_scene_np = None if gt_scene is None else gt_scene.numpy()

        # do class-agnostic NMS if using 2D detection as input
        if not self.config['dataset']['gt_affinity']:
            self.pre_nms(est_scene_np, gt_scene_np)

        # add cameras to 3D scene one by one
        remain_cam_ids = set(est_scene_np['camera'].keys())
        total_cam = len(remain_cam_ids)
        for i_cam in range(total_cam):
            tqdm.write(f"Proposing camera {i_cam + 1}/{total_cam} for scene {est_scene_np['uid']}, "
                       f"exp: {self.logger.experiment.config['args']['id']}")
            if i_cam == 0:
                # use origin camera as the first camera
                best_scene_np = self.initialize_scene(est_scene_np)
                best_cam_id = est_scene_np['origin_camera_id']
            else:
                best_cam_id, affinity_mat = self.choose_camera_to_add(est_scene_np, remain_cam_ids)
                if self.config['dataset']['gt_camera_pose']:
                    best_scene_np = self.skip_proposal_if_use_gt_camera_pose(est_scene_np, remain_cam_ids, best_cam_id, affinity_mat)
                else:
                    best_scene_np = self.propose_pose_and_association_for_new_camera(est_scene_np, remain_cam_ids, best_cam_id, affinity_mat)

            # add the best camera proposal to 3D scene
            best_cam = best_scene_np['camera'][best_cam_id]
            remain_cam_ids.remove(best_cam_id)

            # merge object embeddings, category and score
            for obj2d_id, obj2d in best_cam['object'].items():
                if obj2d_id in best_scene_np['object']:
                    obj3d = best_scene_np['object'][obj2d_id]
                    # update object parameters if score is higher
                    if 'score' in obj2d and obj2d['score'] > obj3d['score']:
                        for key in ('embedding', 'category', 'score'):
                            obj3d[key] = obj2d[key]
                else:
                    # calculate pose for newly added object and add them to to object3d dict
                    best_scene_np['object'][obj2d_id] = object3d_from_prediction(obj2d, best_cam)

            # do class-agnostic NMS on 3D scene if using 2D detection as input
            if not self.config['dataset']['gt_affinity'] and i_cam > 0:
                self.after_nms(best_scene_np)

            # optimize scene with the best camera proposal
            if self.scene_optimization is not None and i_cam == total_cam - 1:
                best_scene = best_scene_np.tensor(device=self.device)
                self.scene_optimization.init_latent_code(best_scene, overwrite=False)
                gt_subscene = None if gt_scene is None else gt_scene.subscene(best_scene['camera'].keys())
                best_scene = self.scene_optimization.optimize_scene(best_scene, gt_subscene)
                best_scene_np = best_scene.numpy(now=True)

            # update est_scene from best_scene
            est_scene_np['object'] = best_scene_np['object']
            if 'latent_code' in best_scene_np:
                est_scene_np['latent_code'] = best_scene_np['latent_code']
            

            for cam_id, cam in best_scene_np['camera'].items():
                est_scene_np['camera'][cam_id] = best_scene_np['camera'][cam_id]

            # visualize scene after each camera proposal
            if est_scene['if_log']:
                # visualize object association graph
                self.calculate_affinity_for_scene(best_scene_np)
                best_scene_vis = SceneVisualizer(best_scene_np)
                est_association_graph_img = best_scene_vis.object_association_graph(get_image=True, draw_id=True)
                gt_subscene_np = gt_scene_np.subscene(best_scene_np['camera'].keys())
                gt_subscene_vis = SceneVisualizer(gt_subscene_np)
                gt_association_graph_img = gt_subscene_vis.object_association_graph(get_image=True, draw_id=True)
                combined_img = image_grid([est_association_graph_img, gt_association_graph_img], rows=1)
                log_dict = {'global_step': est_scene['batch_idx'], 'test/proposal/object_association_graph': wandb.Image(combined_img)}

                # visualize scene proposal
                best_scene_fig = best_scene_vis.scene_vis(reference_scene_vis=gt_subscene_vis)
                with tempfile.NamedTemporaryFile(mode='r+', suffix='.html') as f:
                    best_scene_fig.write_html(f, auto_play=False)
                    log_dict["test/proposal/est_scene_vis"] = wandb.Html(f)
                self.logger.experiment.log(log_dict)

        if est_scene['if_log']:
            # log loss
            log_dict = {'global_step': est_scene['batch_idx']}
            log_dict.update({f"test/proposal/{k}": v for k, v in self.loss.compute_metrics().items()})
            log_dict.update({f"test/nms3d/{k}": v for k, v in self.object3d_confidence_loss.compute_metrics().items()})
            self.logger.experiment.log(log_dict)

        # calculate new affinity
        self.calculate_affinity_for_scene(est_scene_np)

        return est_scene_np.tensor(device=self.device)

    def forward(self, est_scene, gt_scene):
        # camera/object proposal
        get_pitch_roll = not self.config['dataset']['gt_pitch_roll'] and not self.config['dataset']['gt_camera_pose']
        get_camera_yaw = not self.config['dataset']['gt_camera_yaw'] and not self.config['dataset']['gt_camera_pose']
        self.camera_object_proposal(
            est_scene, gt_scene, get_pitch_roll, get_camera_yaw,
            not self.config['dataset']['gt_object_pose'], not self.config['dataset']['gt_affinity'],
        )

        # hypothesis proposal and ranking
        est_scene = self.generate_scene(est_scene, gt_scene)

        # save estimated scene
        self.save_scene_result(est_scene)

        return est_scene

    def test_step(self, batch, batch_idx):
        input_data, gt_data = batch
        est_scene, gt_scene = Scene(input_data, backend=torch, device=self.device), Scene(gt_data, backend=torch, device=self.device)

        if not self.config['test']['skip_prediction']:
            est_scene = self(est_scene, gt_scene)

        # evaluate scene
        self.eval_scene(est_scene, gt_scene)

        # visualize scene
        self.visualize_scene(est_scene, gt_scene)

    def predict_step(self, batch, batch_idx):
        input_data, gt_data = batch
        est_scene, gt_scene = Scene(input_data, backend=torch, device=self.device), Scene(gt_data, backend=torch, device=self.device)

        if not self.config['test']['skip_prediction']:
            est_scene = self(est_scene, gt_scene)

        self.visualize_scene(est_scene, gt_scene)

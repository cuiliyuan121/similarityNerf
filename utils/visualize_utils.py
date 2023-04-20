from functools import wraps
from PIL import Image, ImageDraw
import numpy as np
from utils.transform import homotrans, cam2uv, uv2cam, is_in_cam, bdb3d_corners
from utils.dataset import Object3D
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import networkx as nx
import os
import plotly.graph_objects as go
from collections import defaultdict
from contextlib import suppress


def interpolate_line(p1, p2, num=30):
    t = np.expand_dims(np.linspace(0, 1, num=num, dtype=np.float32), 1)
    points = p1 * (1 - t) + t * p2
    return points


def line3d(image, camera, p1, p2, color, thickness, quality=30):
    color = (np.ones(3, dtype=np.uint8) * color).tolist()
    points = interpolate_line(p1, p2, quality)
    pixel = np.round(cam2uv(camera['K'], points) - 0.5).astype(np.int32)
    in_cam = is_in_cam(camera['K'], camera['width'], camera['height'], points)
    for t in range(quality - 1):
        p1, p2 = pixel[[t, t + 1]]
        if in_cam[[t, t + 1]].all():
            cv2.line(image, tuple(p1), tuple(p2), color, thickness, lineType=cv2.LINE_AA)


def _matplotlib_output(func):
    @wraps(func)
    def wrapper(self, dir=None, get_image=False, *args, **kwargs):
        func(self, *args, **kwargs)
        plt.tight_layout()
        if dir:
            output_image_dir = os.path.join(dir, f"{func.__name__}.png")
            plt.gcf().savefig(output_image_dir)
            plt.close()
        if get_image:
            canvas = FigureCanvas(plt.gcf())
            canvas.draw()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
    return wrapper


def _plotly_output(func):
    @wraps(func)
    def wrapper(self, dir=None, write_html=True, write_image=True, fig_dict=None, *args, **kwargs):
        new_fig = fig_dict is None
        fig_dict = {'data': [], 'layout': {}, 'frames': []} if new_fig or fig_dict is True else fig_dict
        fig_dict['layout']['margin'] = dict(t=0, r=0, l=0, b=0)
        fig_dict = func(self, fig_dict=fig_dict, *args, **kwargs)
        if new_fig:
            fig = go.Figure(fig_dict)
            fig.update_scenes(aspectmode='data')
            if dir:
                output_dir = os.path.join(dir, f"{func.__name__}")
                if write_html:
                    output_html_dir = output_dir + '.html'
                    fig.write_html(output_html_dir)
                if write_image:
                    output_image_dir = output_dir + '.png'
                    fig.write_image(output_image_dir, scale=2)
            return fig
        return fig_dict
    return wrapper


def camera_wireframe(camera, size=1.):
    camera_width, camera_height = camera['width'], camera['height']
    camera_corners = np.array([(0, 0), (0, camera_height), (camera_width, camera_height), (camera_width, 0)])
    camera_corners = uv2cam(camera['K'], camera_corners, size)
    camera_corners = homotrans(camera['cam2world_mat'], camera_corners)
    wireframe = np.concatenate([
        np.stack([camera_corners, np.roll(camera_corners, 1, axis=0)]),
        np.stack([np.repeat(camera['position'][np.newaxis, ...], 4, axis=0), camera_corners]),
    ], axis=1).transpose(1, 0, 2)
    return wireframe


def bdb3d_wireframe(object3d):
    lines = []
    corners = bdb3d_corners(object3d)
    corners_box = corners.reshape(2, 2, 2, 3)
    for k in [0, 1]:
        for l in [0, 1]:
            for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
                lines.append(np.stack([corners_box[idx1], corners_box[idx2]]))
    for idx1, idx2 in [(2, 7), (3, 6)]:
        lines.append(np.stack([corners[idx1], corners[idx2]]))
    wireframe = np.stack(lines)
    return wireframe


def image_grid(images, rows=None, padding=0, background_color=(255, 255, 255), short_height=False):
    if isinstance(images[0], (list, tuple)):
        n_row = len(images)
        n_col = max([len(row) for row in images])
        camera_height, camera_width = images[0][0].shape[:2]
    else:
        n_row = int(np.ceil(np.sqrt(len(images)))) if rows is None else rows
        n_col = int(np.ceil(len(images) / n_row))
        if rows is None and short_height:
            n_row, n_col = n_col, n_row
        camera_height, camera_width = images[0].shape[:2]
    
    # create a numpy array with initial value of background_color
    grid = np.ones(
        (n_row * (camera_height + padding) - padding, n_col * (camera_width + padding) - padding, 3),
        dtype=np.uint8) * np.array(background_color, dtype=np.uint8)
    
    def _add_image(image, row, col):
        if image.dtype == np.float32:
            image = image * 255
        grid[row * (camera_height + padding): row * (camera_height + padding) + camera_height,
             col * (camera_width + padding): col * (camera_width + padding) + camera_width] = image
    
    for i, image in enumerate(images):
        if isinstance(image, (list, tuple)):
            row = i
            col = 0
            for im in image:
                _add_image(im, row, col)
                col += 1
        else:
            row = i // n_col
            col = i % n_col
            _add_image(image, row, col)
    
    return grid


def image_float_to_uint8(img, auto_map=False):
    """
    Convert a float image (0.0-1.0) to uint8 (0-255)
    """
    if img.dtype == np.uint8:
        return img
    if auto_map:
        vmin = np.min(img)
        vmax = np.max(img)
        if vmax - vmin < 1e-10:
            vmax += 1e-10
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.clip(img, 0, 1)
    img *= 255.0
    return img.astype(np.uint8)


def visualize_instance_segmap(instance_segmap):
    obj_ids = list(np.unique(instance_segmap))
    color_box = np.array(sns.hls_palette(n_colors=len(obj_ids), l=.45, s=1.))
    instance_segmap_vis = np.zeros(list(instance_segmap.shape) + [3], dtype=np.uint8)
    for obj_id, color in zip(obj_ids, color_box):
        instance_segmap_vis[instance_segmap == obj_id] = color * 255
    return instance_segmap_vis


def _default_camera(func):
    @wraps(func)
    def wrapper(self, camera_id=None, *args, **kwargs):
        if camera_id is None:
            for camera_id, camera in self.data['camera'].items():
                camera['image'][func.__name__] = func(
                    self, camera_id=camera_id, *args, **kwargs)
        else:
            return func(self, camera_id=camera_id, *args, **kwargs)
    return wrapper


def _default_image(func):
    @wraps(func)
    def wrapper(self, image=None, *args, **kwargs):
        if image is None:
            image = self.data['camera'][kwargs['camera_id']]['image']['color'].copy()
        image = image_float_to_uint8(image)
        return func(self, image=image, *args, **kwargs)
    return wrapper


def _copy_image(func):
    @wraps(func)
    def wrapper(self, image, *args, **kwargs):
        image = image.copy()
        image = func(self, image=image, *args, **kwargs)
        return image
    return wrapper


def _default_object(object3d=True):
    def decorate(func):
        @wraps(func)
        def wrapper(self, image, camera_id, obj_id=None, *args, **kwargs):
            if obj_id is None:
                camera = self.data['camera'][camera_id]
                if object3d:
                    objs = {obj_id: self.data['object'][obj_id] for obj_id in camera['object'].keys()}
                    # calculate object distances from camera
                    distances = {obj_id: np.linalg.norm(objs[obj_id]['center'] - camera['position']) for obj_id in objs}
                    # sort objects ids by distance
                    obj_ids = sorted(distances, key=distances.get, reverse=True)
                else:
                    obj_ids = list(camera['object'].keys())
                for obj_id in obj_ids:
                    image = func(self, image=image, camera_id=camera_id, obj_id=obj_id, *args, **kwargs)
            else:
                image = func(self, image=image, camera_id=camera_id, obj_id=obj_id, *args, **kwargs)
            return image
        return wrapper
    return decorate


class ObjectVisualizer:
    def __init__(self, objnerf):
        self.data = objnerf
        self.color_box = np.array(sns.hls_palette(
            n_colors=len(objnerf.object_categories), l=.45, s=.8))

    @_plotly_output
    def camera(self, fig_dict, draw_camera_id=True, draw_camera_pose=True,
               pose_color='black', id_color='cornflowerblue', draw_object=True):
        # visualize object bdb3d
        if draw_object and isinstance(self.data, Object3D):
            with suppress(RuntimeError):
                self._plotly_bdb3d(fig_dict, self.data)

        # visualize camera poses
        if draw_camera_pose:
            for camera in self.data['camera'].values():
                for line in camera_wireframe(camera):
                    fig_dict['data'].append(go.Scatter3d(
                        x=line[:, 0], y=line[:, 1], z=line[:, 2],
                        mode='lines', line=dict(color=pose_color, width=3), showlegend=False
                    ))

        # visualize camera ids
        if draw_camera_id:
            camera_id = list(self.data['camera'].keys())
            camera_pos = np.stack([camera['position'] for camera in self.data['camera'].values()])
            fig_dict['data'].append(go.Scatter3d(
                x=camera_pos[:, 0], y=camera_pos[:, 1], z=camera_pos[:, 2],
                mode='markers+text', text=camera_id, textposition='middle center',
                marker=dict(size=10, color=id_color, opacity=0.8),
                textfont=dict(size=10, color='black'), showlegend=False
            ))

        return fig_dict

    def _draw_bdb3d(self, image, camera, obj_3d, thickness=2):
        color = 0 if obj_3d['category_id'] is None else self.color_box[obj_3d['category_id']] * 255
        lines = bdb3d_wireframe(obj_3d)
        lines = homotrans(camera['world2cam_mat'], lines)
        for line in lines:
            line3d(image, camera, line[0], line[1], color, thickness=thickness)
        return image

    @_default_camera
    @_default_image
    @_copy_image
    def bdb3d(self, image, camera_id, thickness=2):
        return self._draw_bdb3d(
            image,
            self.data['camera'][camera_id],
            self.data,
            thickness=thickness
        )

    def _plotly_bdb3d(self, fig_dict, obj_3d, color=None):
        if color is None:
            color = f"rgb{tuple((self.color_box[obj_3d['category_id']] * 255).astype(np.uint8))}"
        for line in bdb3d_wireframe(obj_3d):
            fig_dict['data'].append(go.Scatter3d(
                x=line[:, 0], y=line[:, 1], z=line[:, 2],
                mode='lines', line=dict(color=color, width=3), showlegend=False
            ))


class SceneVisualizer(ObjectVisualizer):
    def __init__(self, scene):
        super().__init__(scene)

    @_default_camera
    def instance_segmap_vis(self, camera_id):
        instance_segmap = self.data['camera'][camera_id]['image']['instance_segmap']
        instance_segmap_vis = visualize_instance_segmap(instance_segmap)
        return instance_segmap_vis
    
    @_default_camera
    def semantic_segmap_vis(self, camera_id):
        camera = self.data['camera'][camera_id]
        instance_segmap = camera['image']['instance_segmap']
        semantic_segmap_vis = np.zeros(list(instance_segmap.shape) + [3], dtype=np.uint8)
        seg_ids = list(np.unique(instance_segmap))
        for seg_id in seg_ids:
            if seg_id not in self.data['seg2obj_id']:
                continue
            obj_id = self.data['seg2obj_id'][seg_id]
            if obj_id not in camera['object']:
                continue
            color = self.color_box[camera['object'][obj_id]['category_id']]
            semantic_segmap_vis[instance_segmap == seg_id] = color * 255
        return semantic_segmap_vis
    
    @_default_camera
    @_default_image
    @_copy_image
    @_default_object(object3d=False)
    def bdb2d(self, image, camera_id, obj_id, thickness=2):
        obj = self.data['camera'][camera_id]['object'][obj_id]
        bdb2d = obj['bdb2d']
        cmin, rmin, w, h = bdb2d
        x1, y1, x2, y2 = cmin, rmin, cmin + w - 1, rmin + h - 1
        corners_uv = np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
        color = self.color_box[obj['category_id']] * 255
        for start, end in zip(corners_uv, np.roll(corners_uv, 1, axis=0)):
            cv2.line(image, tuple(start), tuple(end), color, thickness, lineType=cv2.LINE_AA)
        return image
    
    @_default_camera
    @_default_image
    @_copy_image
    @_default_object(object3d=False)
    def segmentation(self, image, camera_id, obj_id, thickness=2, fill=False, contour=True):
        obj = self.data['camera'][camera_id]['object'][obj_id]
        color = self.color_box[obj['category_id']] * 255
        
        # draw segmentation with alpha blending
        if fill:
            image = Image.fromarray(image)
            image.putalpha(255)
            item = obj['segmentation'].astype(np.uint8) * 128
            item = Image.fromarray(item, mode='L')
            overlay = Image.new('RGBA', image.size)
            draw_ov = ImageDraw.Draw(overlay)
            draw_ov.bitmap((0, 0), item, fill=(*color.astype(np.uint8), 128))
            image = Image.alpha_composite(image, overlay)
            image = np.array(image)
        
        # draw contours of segmentation
        if contour:
            contours = cv2.findContours(obj['segmentation'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(image, contours, -1, color, thickness, lineType=cv2.LINE_AA)
        
        return image

    @_default_camera
    @_default_image
    @_copy_image
    @_default_object()
    def axes(self, image, camera_id, obj_id, thickness=2):
        camera = self.data['camera'][camera_id]
        obj = self.data['object'][obj_id]
        origin = homotrans(camera['world2cam_mat'], obj['local2world_mat'][:3, 3])
        for axis, length in zip(np.eye(3, dtype=np.float32), obj['size'] / 2):
            endpoint = homotrans(camera['world2cam_mat'], homotrans(obj['local2world_mat'], axis * length))
            color = axis * 255
            line3d(image, camera, origin, endpoint, color, thickness)
        return image

    @_default_camera
    @_default_image
    @_copy_image
    @_default_object()
    def bdb3d(self, image, camera_id, obj_id, thickness=2):
        return self._draw_bdb3d(
            image,
            self.data['camera'][camera_id],
            self.data['object'][obj_id],
            thickness=thickness
        )

    @_plotly_output
    def scene_vis(self, fig_dict, draw_object=True, draw_camera_pose=True, draw_camera_id=True,
                  draw_camera_overlap=False, draw_object_association=False,
                  reference_scene_vis=None, force_color=None, view_direction=210):
        # draw reference scene if given
        if reference_scene_vis is not None:
            reference_scene_vis.scene_vis(
                fig_dict=fig_dict,
                draw_object=draw_object, draw_camera_pose=draw_camera_pose, draw_camera_id=False,
                draw_camera_overlap=False, draw_object_association=False,
                force_color='darkgray')

            # visualize est_camera and gt_camera correspondences
            for cam_id, est_cam in self.data['camera'].items():
                gt_cam = reference_scene_vis.data['camera'][cam_id]
                line = np.stack([gt_cam['position'], est_cam['position']], axis=-1)
                fig_dict['data'].append(go.Scatter3d(
                    x=line[0], y=line[1], z=line[2], mode='lines',
                    line=dict(color='darkgray', width=3, dash='dash'), showlegend=False))

        # visualize object bdb3d
        if draw_object and 'object' in self.data:
            for obj in self.data['object'].values():
                self._plotly_bdb3d(fig_dict, obj, color=force_color)

        # visualize camera
        camera_colors = {'pose_color': force_color, 'id_color': force_color} if force_color else {}
        self.camera(fig_dict=fig_dict, draw_camera_id=draw_camera_id, draw_camera_pose=draw_camera_pose,
                    draw_object=False, **camera_colors)
        
        # visualize camera_overlap_graph edges
        if draw_camera_overlap or draw_camera_id:
            pos = {id: tuple(c['position']) for id, c in self.data['camera'].items()}
            camera_overlap_graph = self.data.camera_overlap_graph
            if draw_camera_overlap and camera_overlap_graph.edges:
                for s, d in camera_overlap_graph.edges:
                    line = np.stack([pos[s], pos[d]], axis=-1)
                    fig_dict['data'].append(go.Scatter3d(
                        x=line[0], y=line[1], z=line[2], mode='lines',
                        line=dict(color=force_color if force_color else 'coral', width=2), showlegend=False))

        # visualize object association
        if draw_object_association:
            for camera in self.data['camera'].values():
                for obj_id in camera['object']:
                    obj = self.data['object'][obj_id]
                    line = np.stack([camera['position'], obj['center']], axis=-1)
                    fig_dict['data'].append(go.Scatter3d(
                        x=line[0], y=line[1], z=line[2], mode='lines',
                        line=dict(color=force_color if force_color else 'grey', width=2), showlegend=False))

        # set camera view
        dis = 1.7
        fig_dict['layout']['scene_camera'] = dict(
            eye=dict(
                x=-dis * np.cos(45.) / np.cos(np.deg2rad(view_direction)),
                y=-dis * np.cos(45.) / np.sin(np.deg2rad(view_direction)),
                z=np.sqrt(dis)
            )
        )

        return fig_dict
    
    @_matplotlib_output
    def camera_overlap_graph(self, layout='camera_position'):
        camera_overlap_graph = self.data.camera_overlap_graph
        if layout == 'camera_position':
            pos = {}
            for camera_id, camera in self.data['camera'].items():
                pos[camera_id] = tuple(camera['position'][:2])
            plt.axis('equal')
        elif layout == 'graphviz':
            pos = nx.nx_agraph.graphviz_layout(camera_overlap_graph)
        else:
            raise ValueError('Unknown layout: {}'.format(layout))
        
        node_weights = np.array([len(camera['camera']['object']) for camera in camera_overlap_graph.nodes.values()])
        edge_weights = np.array([len(e['overlapped_obj_ids']) for e in camera_overlap_graph.edges.values()])
        # edge_width = [c / max(edge_weights) * 3 for c in edge_weights]
        options = {
            "with_labels": True,
            "font_weight": "bold",
            "node_color": node_weights - node_weights.min(),
            "edge_color": edge_weights - edge_weights.min(),
            "cmap": plt.cm.Wistia,
            "edge_cmap": plt.cm.Wistia,
            "width": 3,
            "node_size": 300
        }
        nx.draw(camera_overlap_graph, pos, **options)

    @_matplotlib_output
    def object_association_graph(self, layout='object_position', draw_id=False, image_scene=None, padding=40):
        object_association_graph = self.data.object_association_graph

        if layout == 'graphviz':
            pos = nx.nx_agraph.graphviz_layout(object_association_graph)
        if layout == 'object_position':
            # sort self.data['camera'] by keys
            sorted_camera = sorted(self.data['camera'].items(), key=lambda x: x[0])
            if image_scene is not None:
                image_camera = sorted(image_scene['camera'].items(), key=lambda x: x[0])
            else:
                image_camera = sorted_camera
            combined_image = image_grid([image_float_to_uint8(c['image']['color']) for _, c in image_camera], padding=padding)
            n_row = int(np.ceil(np.sqrt(len(sorted_camera))))
            n_col = int(np.ceil(len(sorted_camera) / n_row))
            sample_camera = next(iter(self.data['camera'].values()))
            camera_width, camera_height = sample_camera['width'], sample_camera['height']
            pos = {}
            plt.figure(figsize=(camera_width / camera_height * n_col * 2, n_row * 2), dpi=300)
            plt.axis('off')
            for i, (camera_id, camera) in enumerate(sorted_camera):
                row = i // n_col
                col = i % n_col

                # draw image boarder
                camera_corners = np.array([
                    (-0.5, -0.5), (-0.5, camera_height - 0.5),
                    (camera_width - 0.5, camera_height - 0.5),
                    (camera_width - 0.5, -0.5), (-0.5, -0.5)
                ])
                camera_corners += np.array([(camera_width + padding) * col, (camera_height + padding) * row])
                plt.plot(*camera_corners.T, color='black', linewidth=1, zorder=1)

                # calculate node position
                for obj_id, obj in camera['object'].items():
                    cmin, rmin, w, h = obj['bdb2d']
                    center = np.array([cmin + w / 2, rmin + h / 2])
                    pos[(camera_id, obj_id)] = ((camera_width + padding) * col + center[0], (camera_height + padding) * row + center[1])
            plt.imshow(combined_image)
        else:
            raise ValueError('Unknown layout: {}'.format(layout))

        if object_association_graph.edges:
            edge_options = {
                "edge_color": [
                    self.color_box[self.data['object'][obj_id]['category_id']]
                    for (cam_id, obj_id), _ in object_association_graph.edges
                ],
                "width": 1.5,
            }
            if 'affinity' in next(iter(object_association_graph.edges(data=True)))[-1]:
                affinity = nx.get_edge_attributes(object_association_graph, 'affinity').values()
                edge_options['width'] = [c * 3 for c in affinity]
                edge_options['alpha'] = [a * 0.9 for a in affinity]
            nx.draw_networkx_edges(object_association_graph, pos, **edge_options)

        if object_association_graph.nodes:
            node_options = {
                "node_color": [
                    self.color_box[self.data['camera'][cam_id]['object'][obj_id]['category_id']]
                    for (cam_id, obj_id) in object_association_graph.nodes
                ],
                "node_size": 100,
                "alpha": 0.9
            }
            nx.draw_networkx_nodes(object_association_graph, pos, **node_options)
            if draw_id:
                node_options['node_size'] = 200
                labels = {n: n[1] for n in object_association_graph.nodes}
                nx.draw_networkx_labels(object_association_graph, pos, labels, font_size=10, font_color="whitesmoke")


class SceneOptimVisualizer:
    def __init__(self, gt_scene, vis_types):
        self.gt_scene = gt_scene
        self.vis_types = vis_types
        self.gt_scene_vis = SceneVisualizer(gt_scene)

        if 'wireframe' in vis_types:
            fig_dict = {'layout': {}, 'frames': []}

            fig_dict["layout"]["hovermode"] = "closest"
            fig_dict["layout"]["updatemenus"] = [
                {
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 1000/24, "redraw": True},
                                            "fromcurrent": True, "transition": {"duration": 0}}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }
            ]

            sliders_dict = {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Iter:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": []
            }

            fig_dict["layout"]["sliders"] = [sliders_dict]

            self.sliders_dict = sliders_dict
            self.fig_dict = fig_dict
            self.camera_pos_optim = defaultdict(list)

        if 'video' in vis_types:
            self.video_frames = []

    def add_frame(self, scene):
        # visualize wireframe with plotly
        if 'wireframe' in self.vis_types:
            iter = len(self.sliders_dict["steps"])
            # visualize scene
            scene_vis = SceneVisualizer(scene)
            fig_dict = scene_vis.scene_vis(fig_dict=True, reference_scene_vis=self.gt_scene_vis)
            if 'data' not in self.fig_dict:
                self.fig_dict['data'] = fig_dict['data']
            # visualize camera optimization trace
            for camera_id, camera in scene['camera'].items():
                self.camera_pos_optim[camera_id].append(camera['position'])
                line = np.stack(self.camera_pos_optim[camera_id])
                fig_dict['data'].append(go.Scatter3d(
                    x=line[:, 0], y=line[:, 1], z=line[:, 2],
                    mode='lines', line=dict(color='black', width=2), showlegend=False
                ))
            # add frame
            self.fig_dict['frames'].append({'data': fig_dict['data'], 'name': str(iter)})
            # add slider step
            slider_step = {"args": [
                [iter],
                {"frame": {"duration": 300, "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 300}}
            ],
                "label": iter,
                "method": "animate"}
            self.sliders_dict["steps"].append(slider_step)

        # visualize as video
        if 'video' in self.vis_types:
            scene_vis = SceneVisualizer(scene)
            img_rows = []
            for camera_id, camera in scene['camera'].items():
                generated_img = image_float_to_uint8(camera['image']['color'])
                generated_bdb3d_img = scene_vis.bdb3d(image=generated_img, camera_id=camera_id)
                generated_instance_segmap = visualize_instance_segmap(camera['image']['instance_segmap'])
                generated_instance_hitmap = visualize_instance_segmap(camera['image']['instance_hitmap'])

                # log gt instance segmentation map
                gt_instance_segmap = self.gt_scene['camera'][camera_id]['image']['instance_segmap']
                gt_instance_segmap_vis = visualize_instance_segmap(gt_instance_segmap)

                # log masked gt image
                masked_gt_img = self.gt_scene['camera'][camera_id]['image']['masked_color']
                masked_gt_img = image_float_to_uint8(masked_gt_img)

                # log gt image and object bounding boxes
                gt_img = image_float_to_uint8(self.gt_scene['camera'][camera_id]['image']['color'])
                gt_bdb3d_img = self.gt_scene_vis.bdb3d(image=gt_img, camera_id=camera_id)

                img_rows.append([
                    generated_instance_hitmap, generated_instance_segmap, gt_instance_segmap_vis,
                    generated_img, masked_gt_img, generated_bdb3d_img, gt_bdb3d_img, gt_img
                ])

            combined_img = image_grid(img_rows, padding=2, background_color=(128, 128, 128))
            self.video_frames.append(combined_img.transpose(2, 0, 1))

    @_plotly_output
    def scene_optim_vis(self, fig_dict=None):
        return self.fig_dict


@_matplotlib_output
def affinity_heatmap(affinity):
    sns.heatmap(affinity, vmin=0., vmax=1.)

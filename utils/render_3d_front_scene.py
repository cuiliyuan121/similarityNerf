import blenderproc as bproc
import numpy as np
import yaml
import os
import json
import random
import shutil
import debugpy
import sys
import tempfile
import networkx as nx
from tqdm import tqdm
sys.path.append(os.getcwd())
from utils.dataset import check_data, Camera, Scene, sample_connected_component_of_camera_overlap_graph
from utils.prepare_data_utils import shared_argparser, default_output_root, wait_for_gpu
from utils.transform import rotation_mat_dist, WORLD_FRAME_TO_TOTAL3D, CAMERA_FRAME_TO_TOTAL3D
from external import blenderproc_utils
from scipy.spatial.transform import Rotation
import bpy
from mathutils import Matrix
from blenderproc.python.types.MeshObjectUtility import MeshObject
from collections import defaultdict


class SkipException(Exception):
    pass


class Front3DPointInRoomSampler(bproc.sampler.Front3DPointInRoomSampler):
    def sample_near(self, camera_overlap_graph, height: float, max_tries: int = 1000) -> np.ndarray:
        for _ in range(max_tries):
            node = random.choice(list(camera_overlap_graph.nodes))
            node = camera_overlap_graph.nodes[node]

            # Sample room via floor objects
            floor_obj = node['floor_obj']

            # Get min/max along x/y-axis from bounding box of room
            bounding_box = floor_obj.get_bound_box()
            min_corner = np.min(bounding_box, axis=0)
            max_corner = np.max(bounding_box, axis=0)

            # Sample uniformly inside bounding box
            point = np.array([
                random.uniform(min_corner[0], max_corner[0]),
                random.uniform(min_corner[1], max_corner[1]),
                floor_obj.get_location()[2] + height
            ])

            # Check if sampled pose is above the floor to make sure its really inside the room
            if floor_obj.position_is_above_object(point):
                return point

        raise Exception("Cannot sample any point inside the loaded front3d rooms.")


def render_3d_front_scene(args):
    # set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # check data and output directory
    data_dirs = check_data(args.data_dir)

    # load config
    scene_data_config_dir = os.path.join(args.config_dir, 'scene_data.yaml')
    with open(scene_data_config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load 3D-FUTURE objects info
    future_model_info_dir = os.path.join(data_dirs['future_dir'], 'model_info.json')
    with open(future_model_info_dir, 'r') as f:
        future_model_info = json.load(f)
    future_model_info = {model['model_id']: model for model in future_model_info}

    # init blenderproc
    if args.gpu_ids != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    bproc.init()
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
    
    # set renderer
    bproc.renderer.set_cpu_threads(args.cpu_threads)
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                     transmission_bounces=200, transparent_max_bounces=200)
    samples = config['renderer']['samples']
    if args.debug:
        samples //= 4
    bproc.renderer.set_max_amount_of_samples(samples)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    # set camera intrinsics
    camera_config = config['camera']
    f = camera_config['focal_length']
    height = camera_config['height']
    width = camera_config['width']
    K = np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1.]])
    if args.debug:
        K[:2] /= 2
        height //= 2
        width //= 2
    bproc.camera.set_intrinsics_from_K_matrix(K, width, height)

    # load json file of the scene
    front_scene_json = os.path.join(data_dirs['front_scene_dir'], args.id + '.json')
    with open(front_scene_json, "r") as f:
        front_scene_data = json.load(f)

    # remove duplicated furnitures in front_scene_data
    furnitures_str = set()
    for furniture_dict in front_scene_data['furniture']:
        furnitures_str.add(json.dumps(furniture_dict, sort_keys=True))
    front_scene_data['furniture'] = [json.loads(s) for s in furnitures_str]

    # write deduplicated scene as json file into temp folder
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
        json.dump(front_scene_data, f)
        front_scene_json = f.name
        f.flush()

        # load the front 3D objects
        try:
            loaded_objects = bproc.loader.load_front3d(
                json_path=front_scene_json,
                future_model_path=data_dirs['future_dir'],
                front_3D_texture_path=data_dirs['front_texture_dir'],
                label_mapping=mapping,
                ceiling_light_strength=3.,
                lamp_light_strength=20.,
            )
        except Exception as e:
            raise SkipException(f"Error while loading front 3D scene: {e}")
        floor_objs = [obj for obj in loaded_objects if obj.get_name().lower().startswith("floor")]

    # randomly replace room textures
    if 'random_texture' in config:
        # load ccmaterials
        cc_materials = bproc.loader.load_ccmaterials(data_dirs['cc_material_dir'], ["Bricks", "Wood", "Carpet", "Tile", "Marble"])
        for texture in config['random_texture']:
            if texture == 'floor':
                floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
                for floor in floors:
                    # For each material of the object
                    for i in range(len(floor.get_materials())):
                        # In 95% of all cases
                        if np.random.uniform(0, 1) <= 0.95:
                            # Replace the material with a random one
                            floor.set_material(i, random.choice(cc_materials))
            elif texture == 'wood':
                baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*", regex=True)
                wood_floor_materials = bproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
                for obj in baseboards_and_doors:
                    # For each material of the object
                    for i in range(len(obj.get_materials())):
                        # Replace the material with a random one
                        obj.set_material(i, random.choice(wood_floor_materials))
            elif texture == 'wall':
                walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
                marble_materials = bproc.filter.by_cp(cc_materials, "asset_name", "Marble.*", regex=True)
                for wall in walls:
                    # For each material of the object
                    for i in range(len(wall.get_materials())):
                        # In 50% of all cases
                        if np.random.uniform(0, 1) <= 0.1:
                            # Replace the material with a random one
                            wall.set_material(i, random.choice(marble_materials))
            else:
                raise ValueError(f'Unknown texture type {texture}')

    # Init sampler for sampling locations inside the loaded front3D house
    amount_of_objects_needed_per_room = config['multi_view']['min_object'] - 1
    point_sampler = Front3DPointInRoomSampler(loaded_objects, amount_of_objects_needed_per_room=amount_of_objects_needed_per_room)
    if len(point_sampler.used_floors) == 0:
        raise SkipException(f'No floor found in scene {args.id}, there might be no enough objects in any room.')

    # Init bvh tree containing all mesh objects
    bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

    def check_name(name):
        for category_name in config['special_categories']:
            if category_name in name.lower():
                return True
        return False

    # filter some objects from the loaded objects, which are later used in calculating an interesting score
    obj_name2id = {obj.get_name(): i for i, obj in enumerate(loaded_objects) if check_name(obj.get_name())}

    def connected_complete_components(camera_overlap_graph):
        graph = camera_overlap_graph.copy()

        # ensure proposed connected component is a complete graph
        connected_components = []

        # break graph into fully connected components
        def break_graph_util_complete(g):
            if len(g.nodes) == 0:
                return

            # iterate over all connected components and break them down
            if not nx.is_connected(g):
                for sub_g in nx.connected_components(g):
                    break_graph_util_complete(graph.subgraph(sub_g).copy())
                return

            # skip isolated nodes
            if len(g.nodes) < 2:
                return

            # add connected component if it is complete
            if all([degree[1] == len(g.nodes) - 1 for degree in g.degree]):
                connected_components.append(set(g.nodes))
                return

            # remove minimum edge cut from g
            min_edge_cut = nx.minimum_edge_cut(g)
            g.remove_edges_from(min_edge_cut)
            break_graph_util_complete(g)

        break_graph_util_complete(graph)

        return connected_components

    # generate random camera poses for multi-view rendering
    mv_config = config['multi_view']
    camera_overlap_graph = nx.Graph()
    pose_id = 0
    max_try = mv_config['max_try']
    component_proposal_try = mv_config['component_proposal_try']
    if args.debug:
        max_try //= 10
        component_proposal_try //= 10
    target_num_camera = mv_config['target_num_camera']
    update_connected_component = True
    for i_try in tqdm(range(max_try), desc="Generating camera poses"):
        # Sample point inside house
        altitude = np.random.uniform(camera_config['altitude']['min'], camera_config['altitude']['max'])
        try:
            location = point_sampler.sample(altitude) if update_connected_component \
                else point_sampler.sample_near(camera_overlap_graph, altitude)
        except Exception as e:
            raise SkipException(f"Error while sampling point inside house: {e}")

        # Sample rotation
        euler = np.random.uniform(
            [np.deg2rad(camera_config['roll']['min']), np.deg2rad(camera_config['pitch']['min']), 0],
            [np.deg2rad(camera_config['roll']['max']), np.deg2rad(camera_config['pitch']['max']), np.pi * 2])
        rotation_mat = Rotation.from_euler('xzy', euler).as_matrix()
        rotation_mat = WORLD_FRAME_TO_TOTAL3D.T @ rotation_mat @ CAMERA_FRAME_TO_TOTAL3D
        cam2world_mat = bproc.math.build_transformation_mat(location, rotation_mat)

        # Check that obstacles are at least proximity_checks['min'] meter away from the camera
        # and have an average distance between proximity_checks['avg']['min'] and proximity_checks['avg']['max']
        # meters and make sure that no background is visible (optional), finally make sure the view is interesting enough
        if not bproc.camera.perform_obstacle_in_view_check(
                cam2world_mat, config['proximity_checks'], bvh_tree,  sqrt_number_of_rays=mv_config['sqrt_number_of_rays']):
            continue

        # get object ids in view
        def visible_objects(cam2world_matrix, sqrt_number_of_rays):
            cam2world_matrix = Matrix(cam2world_matrix)

            visible_object_info = defaultdict(lambda: {'area': 0., 'horizental_ray': set(), 'vertical_ray': set(), 'rays': []})
            cam_ob = bpy.context.scene.camera
            cam = cam_ob.data

            # Get position of the corners of the near plane
            frame = cam.view_frame(scene=bpy.context.scene)
            # Bring to world space
            frame = [cam2world_matrix @ v for v in frame]

            # Compute vectors along both sides of the plane
            vec_x = frame[1] - frame[0]
            vec_y = frame[3] - frame[0]

            # Go in discrete grid-like steps over plane
            position = cam2world_matrix.to_translation()
            for x in range(0, sqrt_number_of_rays):
                for y in range(0, sqrt_number_of_rays):
                    # Compute current point on plane
                    end = frame[0] + vec_x * x / float(sqrt_number_of_rays - 1) + vec_y * y / float(sqrt_number_of_rays - 1)
                    # Send ray from the camera position through the current point on the plane
                    _, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.evaluated_depsgraph_get(), position, end - position)
                    # Add hit object to set
                    obj = MeshObject(hit_object)
                    if hit_object and obj.has_cp('uid') and obj in loaded_objects and obj.get_name() in obj_name2id:
                        obj_id = obj_name2id[obj.get_name()]
                        visible_object_info[obj_id]['area'] += 100. / (sqrt_number_of_rays ** 2)
                        visible_object_info[obj_id]['horizental_ray'].add(y)
                        visible_object_info[obj_id]['vertical_ray'].add(x)
                        visible_object_info[obj_id]['rays'].append([x, y])

            return visible_object_info

        visible_object_info = visible_objects(cam2world_mat, sqrt_number_of_rays=mv_config['sqrt_number_of_rays'])
        visible_object_info = {
            obj_id: info
            for obj_id, info in visible_object_info.items()
            if info['area'] >= mv_config['min_object_area']
        }
        if len(visible_object_info) < mv_config['min_object']:
            continue  # ensures min_object visible with each covering at least min_object_area pixels
        horizontal_proj = set().union(*[info['horizental_ray'] for info in visible_object_info.values()])
        if (max(horizontal_proj) - min(horizontal_proj)) / mv_config['sqrt_number_of_rays'] < mv_config['min_horizontal_span']:
            continue  # ensures that the objects are not squeezed to the left or right of the image
        if len(horizontal_proj) / mv_config['sqrt_number_of_rays'] < mv_config['min_horizontal_coverage']:
            continue  # ensures that the objects covers at least min_horizontal_coverage of the horizontal field of view
        vertical_proj = set().union(*[info['vertical_ray'] for info in visible_object_info.values()])
        if (max(vertical_proj) - min(vertical_proj)) / mv_config['sqrt_number_of_rays'] < mv_config['min_vertical_span']:
            continue  # ensures that the objects are not squeezed to the top or bottom of the image
        if len(vertical_proj) / mv_config['sqrt_number_of_rays'] < mv_config['min_vertical_coverage']:
            continue  # ensures that the objects covers at least min_vertical_coverage of the vertical field of view
        visible_obj_ids = set(visible_object_info.keys())

        # get floor_obj of camera
        for floor_obj in floor_objs:
            if floor_obj.position_is_above_object(location):
                break

        # ensure that at least one object is in the center of the image
        obj_rays = sum([info['rays'] for info in visible_object_info.values()], [])
        obj_rays = np.array(obj_rays)
        ray_center_dis = np.linalg.norm(obj_rays - mv_config['sqrt_number_of_rays'] / 2, axis=1)
        hit_rays = ray_center_dis <= mv_config['focus_radius'] * mv_config['sqrt_number_of_rays']
        if not np.any(hit_rays):
            continue

        # temporarily add camera to graph
        camera_overlap_graph.add_node(pose_id, location=location, floor_obj=floor_obj, cam2world_mat=cam2world_mat, visible_obj_ids=visible_obj_ids)

        # check if the camera has overlapped objects with other cameras in the same room, if so, add an edge
        for node in camera_overlap_graph.nodes:
            if node == pose_id:
                continue
            overlapped_obj_ids = camera_overlap_graph.nodes[node]['visible_obj_ids'] & visible_obj_ids

            # ensure at least one overlapped object is in the center of the image
            overlapped_obj_rays = sum([info['rays'] for obj_id, info in visible_object_info.items() if obj_id in overlapped_obj_ids], [])
            overlapped_obj_rays = np.array(overlapped_obj_rays)
            if len(overlapped_obj_rays):
                ray_center_dis = np.linalg.norm(overlapped_obj_rays - mv_config['sqrt_number_of_rays'] / 2, axis=1)
                hit_rays = ray_center_dis <= mv_config['focus_radius'] * mv_config['sqrt_number_of_rays']
                if np.any(hit_rays) and \
                        len(overlapped_obj_ids) > mv_config['min_overlap_object'] and \
                        len(set().union(*[info['horizental_ray'] for obj_id, info in visible_object_info.items() if obj_id in overlapped_obj_ids])) / mv_config['sqrt_number_of_rays'] >= mv_config['min_horizontal_coverage'] and \
                        len(set().union(*[info['vertical_ray'] for obj_id, info in visible_object_info.items() if obj_id in overlapped_obj_ids])) / mv_config['sqrt_number_of_rays'] >= mv_config['min_vertical_coverage']:
                    camera_overlap_graph.add_edge(node, pose_id, overlapped_obj_ids=overlapped_obj_ids)

        if update_connected_component:
            # at first, update the connected components every time a new camera is added
            connected_components = connected_complete_components(camera_overlap_graph)
            connected_component = next(iter(c for c in connected_components if pose_id in c), [])
            connected_component = camera_overlap_graph.subgraph(connected_component).copy()
        if connected_components and i_try >= component_proposal_try:
            if update_connected_component:
                # select the largest connected component
                connected_component = max(connected_components, key=len)
                # starting from component_proposal_try, camera_overlap_graph only contains the selected connected component
                camera_overlap_graph = connected_component = camera_overlap_graph.subgraph(connected_component | {pose_id}).copy()
                update_connected_component = False
            # after component_proposal_try, only add camera to the selected connected component
            # if the camera has enough overlapped objects with all other cameras
            if camera_overlap_graph.degree(pose_id) != len(camera_overlap_graph) - 1:
                camera_overlap_graph.remove_node(pose_id)
                continue

        # check if the camera pose is different enough from other cameras in the same connected component
        existing_poses = [camera_overlap_graph.nodes[node]['cam2world_mat'] for node in connected_component.adj[pose_id]] if connected_component else []

        # check if the camera pose is different enough from other cameras with enough overlapped objects
        # different enough means that the camera poses are at least min_distance apart
        # or the relative rotation between the camera poses is at least min_rotation apart
        def check_novel_pose(cam2world_mat, existing_poses, min_rotation, min_distance):
            if min_rotation <= 0 or min_distance <= 0:
                return True

            for existing_pose in existing_poses:
                # calculate distances between cam2world_mat and existing_pose
                distance = np.linalg.norm(cam2world_mat[:3, 3] - existing_pose[:3, 3])
                # calculate absolute rotation difference between cam2world_mat and existing_pose
                rotation_error = rotation_mat_dist(cam2world_mat[:3, :3], existing_pose[:3, :3])
                if distance < min_distance and rotation_error < min_rotation:
                    return False
            return True

        if existing_poses and (
                not check_novel_pose(cam2world_mat, existing_poses, **config['my_check_novel_pose'])
                or not bproc.camera.check_novel_pose(cam2world_mat, existing_poses, **config['check_novel_pose'])):
            camera_overlap_graph.remove_node(pose_id)
            continue

        # break if enough poses are generated
        if update_connected_component:
            connected_components_size = [len(c) for c in connected_components if len(c) > 1]
            if connected_components_size and max(connected_components_size) >= target_num_camera:  # stop if any rooms have enough cameras
                break
        else:
            if len(camera_overlap_graph.nodes) >= target_num_camera:
                break

        pose_id += 1

    connected_components = connected_complete_components(camera_overlap_graph)
    if not connected_components:
        raise SkipException("No valid camera pose generated")
    connected_component = max(connected_components, key=len)
    connected_component = list(connected_component)
    random.shuffle(connected_component)
    connected_component = connected_component[:target_num_camera]
    camera_overlap_graph = camera_overlap_graph.subgraph(connected_component)

    # rename node labels
    camera_overlap_graph = nx.convert_node_labels_to_integers(camera_overlap_graph)

    # add camera poses to blenderproc
    if len(camera_overlap_graph.nodes) == 0:
        raise SkipException("No valid camera pose generated")
    for node in camera_overlap_graph.nodes.values():
        bproc.camera.add_camera_pose(node['cam2world_mat'])

    bpy.ops.wm.save_mainfile(filepath="/idas/users/cuiliyuan/NeRFSceneUnderstanding/data/gen_scene/scene.blend")
    
def main():
    parser = shared_argparser()
    args = parser.parse_args()

    # Enable debug mode
    if args.debugpy:
        debugpy.listen(5678)
        print('debugpy listening on port 5678')
        debugpy.wait_for_client()

    try:
        render_3d_front_scene(args)
    except SkipException as e:
        print('Skip scene due to:')
        print(e)
        output_scene_root = default_output_root(args.output_dir, 'scene', args.debug)
        skipped_scene_dir = os.path.join(output_scene_root, 'skipped_scenes.txt')
        with open(skipped_scene_dir, 'a') as f:
            f.write(args.id + '\n')


if __name__ == "__main__":
    main()

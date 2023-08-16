# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import argparse
import copy

import numpy as np

from renderer.online_object_renderer import OnlineObjectRenderer
from utils import utils


class DatasetGenerator(object):
    def __init__(
        self, num_points_to_sample_on_mesh, mesh_folder, grasp_folder, folder_for_storing, threshold, num_query_points, visualize_data=False
    ):
        self.renderer = OnlineObjectRenderer(caching=True)
        self.uniformly_spaces_orientations = utils.get_uniformly_spaced_orientations()
        self.num_point_to_sample_on_mesh = num_points_to_sample_on_mesh
        self.directory_to_store_results = folder_for_storing
        self.distance_threshold_for_neighbor = threshold
        self.num_query_points = num_query_points
        self.visualize_data = visualize_data
        self.mesh_files = utils.return_files_in_folder(mesh_folder, "obj")
        self.grasp_files = utils.return_files_in_folder(grasp_folder, "h5")
        self.setup_gripper()
        self.create_directories_to_store_results()

    def setup_gripper(self):
        """This function creates a mesh representation of the franka panda gripper in its canonical pose and the center grasp point"""
        self.gripper_visualizer, self.center_grasp_point = utils.setup_gripper()
        self.original_gripper = copy.deepcopy(self.gripper_visualizer)

    def render_object_point_cloud_from_all_viewpoints(self, cad_path, cad_scale):
        """Renders object point cloud from the qually spaced camera orientations toward the object

        Args:
            cad_path (string): path to the mesh file
            cad_scale (float): A scale telling us how much to increase or decrease the size of the mesh before rendering

        Returns:
            point_clouds (list): A list of rendered point clouds
            camera_poses (list): A list of the different camera poses for rendering the point clouds
        """
        point_clouds = []
        camera_poses = []
        for camera_pose in self.uniformly_spaces_orientations:
            try:
                rendered_point_cloud, camera_pose = self.render_object_point_cloud_from_one_viewpoint(camera_pose, cad_path, cad_scale)
                point_clouds.append(rendered_point_cloud)
                camera_poses.append(camera_pose)
            except ValueError:
                continue

        return point_clouds, camera_poses

    def render_object_point_cloud_from_one_viewpoint(self, camera_pose, cad_path, cad_scale):
        """This function resizes the object mesh to the correct scale and then renders a single point cloud of it from the given camera pose

        Args:
            camera_pose (np.array[4,4]): A 4x4 matrix representing the camera pose
            cad_path (string): path to the mesh file
            cad_scale (float): A scale telling us how much to increase or decrease the size of the mesh before rendering

        Raises:
            ValueError: If the object could not be rendered which could be the case for 2D objects

        Returns:
            point_cloud (numpy.array[1024,3]): A matrix containing the eucledian postition of each point in the rendered point cloud
            camerar_pose (numpy.array[4,4]): The tranformation matrix of the camera used for rendering the object point cloud
        """
        in_camera_pose = copy.deepcopy(camera_pose)
        pc, camera_pose = self.renderer.change_and_render(cad_path, cad_scale, in_camera_pose, canonical_pc=False)

        if pc.shape[0] == 0:
            raise ValueError("Could not render point cloud of object", cad_path)
        point_cloud = utils.regularize_pc_point_count(pc, self.num_point_to_sample_on_mesh)
        pc_mean = np.mean(pc, 0, keepdims=True)
        point_cloud = self.center_point_cloud(point_cloud, pc_mean)
        camera_pose = self.center_camera_pose(camera_pose, pc_mean)
        return point_cloud, camera_pose

    def center_point_cloud(self, point_cloud, center_value):
        """Translates the input point cloud by a specific value

        Args:
            point_cloud (numpy.array[K,3]): A K by 3 point cloud
            center_value (numpy.array[1,3]): The value for which we translate the input point cloud by

        Returns:
            point_cloud (numpy.array[K,3]): A K by 3 point cloud
        """
        point_cloud[:, :3] -= center_value[:, :3]
        return point_cloud

    def center_camera_pose(self, camera_pose, center_value):
        """Translates the input camera pose by a specific value

        Args:
            camera_pose (numpy.array[4,4]): The transformation matrix of the camera
            center_value (numpy.array[1,3]): The value for which we translate the camera pose by

        Returns:
            camera_pose (numpy.array[4,4]): The transformation matrix of the camera
        """

        camera_pose[:3, 3] -= center_value[0, :3]
        return camera_pose

    def create_directories_to_store_results(self):
        utils.create_folder(self.directory_to_store_results + "grasps/")

    def find_grasp_file_for_current_mesh(self, mesh_file):
        """Returns the file containing grasps for the provided mesh file

        Args:
            mesh_file (string): the file path to the mesh

        Raises:
            FileNotFoundError: Return an error if no grasp file is found for that mesh file

        Returns:
            String: the path to the grasp file
        """
        grasp_file = utils.find_match(self.grasp_files, utils.get_basename_from_file(mesh_file))

        if grasp_file == "":
            raise FileNotFoundError("Could not find any grasp file for mesh ", mesh_file)
        return grasp_file

    def load_grasp_information_from_file(self, file):
        """Load the information about grasps for an object

        Args:
            file (string): Path to a grasp file

        Returns:
            transformations (np.asarray[K,4,4]): An array of Kx4x4 transformation matrices
            success (np.asarray[K,1]): An array of Kx1 values representing the success of each grasp
            object scale (float): A value representing the scale applied to the object to make in possible to grasp it
        """
        grasps = utils.read_h5_file(file)
        transformations = np.array(grasps["grasps/transforms"], dtype=np.float32)
        success = np.array(grasps["grasps/qualities/flex/object_in_gripper"], dtype=np.int32)
        object_scale = grasps["object/scale"]
        return transformations, success, object_scale

    def generate_dataset(self):
        """Generates the dataset as explained in Section 5 in the paper"""
        for mesh in self.mesh_files:
            try:
                grasp_file = self.find_grasp_file_for_current_mesh(mesh)
            except FileNotFoundError:
                continue
            grasp_transformation_matrices, grasp_successes, object_scale = self.load_grasp_information_from_file(grasp_file)
            center_gripper_points_for_each_grasp = utils.transform_vector(self.center_grasp_point, grasp_transformation_matrices)

            rendered_point_clouds, camera_poses = self.render_object_point_cloud_from_all_viewpoints(mesh, object_scale)
            query_points_for_each_rendered_point_cloud = []
            grasp_indices_for_every_query_point_on_each_rendered_point_cloud = []
            o3d_mesh = utils.load_mesh_from_file(mesh, object_scale)
            max_radius = utils.radius_of_sphere_that_circumscribes_the_mesh(o3d_mesh)

            for rendered_point_cloud, camera_pose in zip(rendered_point_clouds, camera_poses):
                center_gripper_points_for_current_camera_pose = utils.transform_vector(
                    copy.deepcopy(center_gripper_points_for_each_grasp), camera_pose
                )
                grasp_to_point_proximity_matrix = self.find_all_successful_grasps_closer_to_points_than_threshold(
                    rendered_point_cloud,
                    center_gripper_points_for_current_camera_pose,
                    self.distance_threshold_for_neighbor,
                    grasp_successes,
                )

                query_points, grasps_per_query_points = self.cluster_points_on_mesh_and_grasps(
                    grasp_to_point_proximity_matrix, rendered_point_cloud, self.num_query_points, max_radius
                )
                query_points_for_each_rendered_point_cloud.append(query_points)
                grasp_indices_for_every_query_point_on_each_rendered_point_cloud.append(grasps_per_query_points)

                if self.visualize_data:
                    o3d_mesh_transformed = copy.deepcopy(o3d_mesh).transform(camera_pose)
                    self.visualize(
                        o3d_mesh_transformed,
                        query_points,
                        rendered_point_cloud,
                        grasp_transformation_matrices,
                        grasps_per_query_points,
                        camera_pose,
                    )

            self.save_data_to_file(
                mesh,
                object_scale,
                rendered_point_clouds,
                camera_poses,
                grasp_transformation_matrices,
                grasp_successes,
                query_points_for_each_rendered_point_cloud,
                grasp_indices_for_every_query_point_on_each_rendered_point_cloud,
                grasp_file,
            )

    def cluster_points_on_mesh_and_grasps(self, grasp_to_point_proximity_matrix, points_on_mesh, number_of_cluster, max_radius):
        threshold_for_neighbors = np.random.uniform(0, max_radius)
        grasp_indices = self.repack_grasps_per_point_on_mesh(grasp_to_point_proximity_matrix)
        cluster_of_points = utils.cluster_points(points_on_mesh, number_of_cluster, threshold_for_neighbors)

        cluster_of_points_with_grasps = utils.remove_clusters_with_no_grasps(cluster_of_points, grasp_indices)
        grasps_per_clusters = utils.get_grasps_per_cluster_parallel(cluster_of_points_with_grasps, grasp_indices)
        return cluster_of_points_with_grasps, grasps_per_clusters

    def save_data_to_file(
        self,
        mesh,
        object_scale,
        rendered_point_clouds,
        camera_poses,
        grasp_transformation_matrices,
        grasp_successes,
        query_points_for_each_rendered_point_cloud,
        grasp_indices_for_every_query_point_on_each_rendered_point_cloud,
        grasp_file,
    ):
        """Saves the generated data to a seperate file

        Args:
            mesh (string): The name of the object
            object_scale (float): The scale applied to the object
            rendered_point_clouds (list): A list of rendered point clouds
            camera_poses (list): A list of camera poses used to render the point clouds
            grasp_transformation_matrices (list): A list of 4x4 grasp transformation matrices
            grasp_successes (list): A list of the success of each grasp
            query_points_for_each_rendered_point_cloud (list): The indices of the points on the rendered point clouds
            grasp_indices_for_every_query_point_on_each_rendered_point_cloud (_type_): _description_
            grasp_file (string): The path to the grasp file used to create the new data
        """
        data_to_save = self.repack_data_for_storing(
            mesh,
            object_scale,
            rendered_point_clouds,
            camera_poses,
            grasp_transformation_matrices,
            grasp_successes,
            query_points_for_each_rendered_point_cloud,
            grasp_indices_for_every_query_point_on_each_rendered_point_cloud,
        )
        file_name_for_storing = utils.get_basename_from_file(grasp_file)
        utils.save_dataset(data_to_save, file_name_for_storing, self.directory_to_store_results + "grasps/")

    def repack_data_for_storing(
        self,
        mesh,
        object_scale,
        rendered_point_clouds,
        camera_poses,
        grasp_transformation_matrices,
        grasp_successes,
        query_points_for_each_rendered_point_cloud,
        grasp_indices_for_every_query_point_on_each_rendered_point_cloud,
    ):
        dictionary_for_storage = {}
        dictionary_for_storage["mesh/file"] = mesh
        dictionary_for_storage["mesh/scale"] = object_scale
        dictionary_for_storage["rendering/point_clouds"] = rendered_point_clouds
        dictionary_for_storage["rendering/camera_poses"] = camera_poses
        dictionary_for_storage["grasps/transformations"] = grasp_transformation_matrices
        dictionary_for_storage["grasps/successes"] = grasp_successes
        dictionary_for_storage["query_points/points_with_grasps_on_each_rendered_point_cloud"] = query_points_for_each_rendered_point_cloud
        dictionary_for_storage[
            "query_points/grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud"
        ] = grasp_indices_for_every_query_point_on_each_rendered_point_cloud
        return dictionary_for_storage

    def visualize(
        self, mesh, query_points_with_grasps_for_current_point_cloud, point_cloud, grasp_transformation_matrices, grasp_indices, camera_pose
    ):
        for index, (points_per_cluster, grasp_indices_per_clusters) in enumerate(
            zip(query_points_with_grasps_for_current_point_cloud, grasp_indices)
        ):
            self.gripper_visualizer.vertices = self.original_gripper.vertices
            utils.visualize_all_grasps_per_query_point(
                mesh,
                point_cloud,
                points_per_cluster,
                self.gripper_visualizer,
                grasp_transformation_matrices,
                grasp_indices_per_clusters,
                camera_pose,
            )
            if index == 5:
                break

    def find_all_successful_grasps_closer_to_points_than_threshold(
        self, points_on_mesh, transformed_gripper_points, threshold, grasp_successes
    ):
        distances = utils.euclid_dist(points_on_mesh, transformed_gripper_points)
        return (distances < threshold) * grasp_successes

    def repack_grasps_per_point_on_mesh(self, grasps_closer_than_threshold, max_num_grasps=5000):
        query_points_and_grasp_indices_grasps = np.argwhere(grasps_closer_than_threshold)
        query_points_and_grasp_indices_grasps = query_points_and_grasp_indices_grasps.astype(np.int32)
        num_grasps = query_points_and_grasp_indices_grasps.shape[0]

        if num_grasps > max_num_grasps:
            grasp_indices_to_keep = np.random.choice(range(num_grasps), size=max_num_grasps, replace=False)
            query_points_and_grasp_indices_grasps = query_points_and_grasp_indices_grasps[grasp_indices_to_keep]
        num_grasps = query_points_and_grasp_indices_grasps.shape[0]

        return query_points_and_grasp_indices_grasps


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--mesh_path", default="data/shapenetsem_example_meshes/", type=str, help="Path to meshes")
    parser.add_argument("-gp", "--grasp_path", default="data/acronym_example_grasps/", type=str, help="Path to acronym grasps")
    parser.add_argument(
        "-sp", "--folder_for_storing", type=str, default="/tmp/constrained_grasping_dataset/", help="Path where to save results"
    )
    parser.add_argument(
        "-np",
        "--number_of_points_to_sample_on_mesh",
        type=int,
        default=1024,
        help="Number of points to sample on the mesh, represented by O in the paper",
    )
    parser.add_argument(
        "-th",
        "--threshold",
        type=float,
        default=0.002,
        help="Threshold between center of grasps and query points, represented as d in the paper.",
    )
    parser.add_argument("-nq", "--num_query_points", type=int, default=50, help="Number of query points I to sample from O.")
    parser.add_argument("-v", "--visualize", action="store_true", help="Set to true if we want to visualize the results")

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = options()
    dataset_generator = DatasetGenerator(
        args.number_of_points_to_sample_on_mesh,
        args.mesh_path,
        args.grasp_path,
        args.folder_for_storing,
        args.threshold,
        args.num_query_points,
        args.visualize,
    )
    dataset_generator.generate_dataset()

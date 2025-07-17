import os
import time
import json
import csv
import glob
import trimesh
import warnings
import traceback
import numpy as np
import open3d as o3d
import multiprocessing
from scipy.stats import entropy
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional, Tuple, Dict, Any
from scipy.spatial.transform import Rotation as R
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_IN
from OCC.Core.TopoDS import topods, TopoDS_Shape, TopoDS_Solid, TopoDS_Iterator
from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

TEST_MODE = False
VISUALIZE = False
USE_VOXEL_IOU = False
USE_BREP_IOU = True
USE_MULTIPROCESSING = True  # 重新启用多进程处理
if not TEST_MODE or USE_MULTIPROCESSING:
    VISUALIZE = False


@dataclass
class SamplingConfig:
    """Sampling configuration class"""
    method: str = 'mesh'  # 'shape' or 'mesh'
    sampling_type: str = 'surface'  # 'surface' or 'volume'
    density: int = 30
    min_density: int = 20
    max_density: int = 50
    jitter_factor: float = 0.5
    poisson_radius: float = 0.1
    n_points: int = 1000


@dataclass
class MetricsResult:
    """Metrics calculation results
    Metric description:
    - Chamfer/Hausdorff/Earth_Mover/RMS distances: smaller is better, indicating distance between two point clouds
    - IoU: larger is better (0~1), indicating overlap ratio
    - Matching rate: larger is better (0~1), proportion of points matched within the threshold
    - Coverage rate: larger is better (0~1), contrary to the matching rate
    - JSD: smaller is better, Jensen-Shannon divergence
    - Normal Consistency: larger is better (0~1), indicating normal vector consistency
    """
    # Main distance metrics
    chamfer_distance: Optional[float] = None  # mean distance
    hausdorff_distance: Optional[float] = None  # max distance
    earth_mover_distance: Optional[float] = None  # earth mover's Distance
    rms_error: Optional[float] = None  # root mean square error

    # Similarity metrics (range 0~1)
    iou_voxel: Optional[float] = None  # Intersection over Union based on voxel
    iou_brep: Optional[float] = None  # Intersection over Union based on brep
    matching_rate: Optional[float] = None  # Matching rate
    coverage_rate: Optional[float] = None  # Coverage rate
    # jsd: Optional[float] = None  # Jensen-Shannon divergence
    normal_consistency: Optional[float] = None  # Normal consistency

    # Detailed distance metrics (shown only when needed)
    _chamfer_x_to_y: Optional[float] = None
    _chamfer_y_to_x: Optional[float] = None
    _hausdorff_x_to_y: Optional[float] = None
    _hausdorff_y_to_x: Optional[float] = None
    _matching_threshold: Optional[float] = None

    # Performance statistics
    computation_time: Optional[float] = None  # computation time (seconds)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary"""
        return {k: v for k, v in self.__dict__.items()
                if v is not None and not k.startswith('_')}

    def __str__(self) -> str:
        """String representation"""
        main_metrics = [
            ("Chamfer Distance", self.chamfer_distance, "↓"),
            ("Hausdorff Distance", self.hausdorff_distance, "↓"),
            ("Earth Mover's Distance", self.earth_mover_distance, "↓"),
            ("RMS Error", self.rms_error, "↓"),
            ("IoU Voxel", self.iou_voxel, "↑"),
            ("IoU Brep", self.iou_brep, "↑"),
            ("Matching Rate", self.matching_rate, "↑"),
            ("Coverage Rate", self.coverage_rate, "↑"),
            # ("JSD", self.jsd, "↓"),
            ("Normal Consistency", self.normal_consistency, "↑"),
        ]

        result = "=== Metrics Calculation Results ===\n"
        # Display main metrics
        for name, value, trend in main_metrics:
            if value is not None:
                result += f"{name}: {value:.6f} {trend}\n"

        # Display computation time
        if self.computation_time is not None:
            result += f"Computation time: {self.computation_time:.3f}s"

        return result

    def get_detailed_metrics(self) -> str:
        """Get detailed metrics info"""
        result = str(self) + "\n\n=== Detailed Metrics ===\n"
        if self._chamfer_x_to_y is not None:
            result += f"Chamfer (X→Y): {self._chamfer_x_to_y:.6f}\n"
            result += f"Chamfer (Y→X): {self._chamfer_y_to_x:.6f}\n"
        if self._hausdorff_x_to_y is not None:
            result += f"Hausdorff (X→Y): {self._hausdorff_x_to_y:.6f}\n"
            result += f"Hausdorff (Y→X): {self._hausdorff_y_to_x:.6f}\n"
        if self._matching_threshold is not None:
            result += f"Matching Threshold: {self._matching_threshold:.6f}\n"
        return result


def save_checkpoint(progress_file, results, batch_index):
    """Save checkpoint"""
    with open(progress_file, 'w') as f:
        json.dump({'results': results, 'last_batch_index': batch_index}, f)


def load_checkpoint(progress_file):
    """Load checkpoint"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            checkpoint = json.load(f)
            return checkpoint['results'], checkpoint['last_batch_index']
    return [], -1


def save_results_to_file(results: List[Dict[str, Any]], output_file: str):
    """Save results to file"""
    try:
        # Choose save format based on file extension
        if output_file.endswith('.json'):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        elif output_file.endswith('.csv'):
            if results:
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(
                        f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
        else:
            # Default save as text format
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results):
                    f.write(f"=== Result {i + 1} ===\n")
                    for k, v in result.items():
                        f.write(f"{k}: {v}\n")
                    f.write("\n")

    except Exception as e:
        print(f"Failed to save results: {e}")


class MeshHandler:
    def __init__(self, sampling_config: Optional[SamplingConfig] = None):
        self.config = sampling_config or SamplingConfig()
        self.n_points = self.config.n_points

    @staticmethod
    def load_shape_from_step(file_path: str) -> Optional[TopoDS_Shape]:
        """Load shape from STEP file"""
        try:
            reader = STEPControl_Reader()
            reader.ReadFile(file_path)
            reader.TransferRoot()
            return reader.Shape()
        except Exception as e:
            print(f"Shape loading failed: {e}")
            return None

    @staticmethod
    def load_mesh_from_stl(file_path: str) -> Optional[TopoDS_Shape]:
        """Load shape from STL file"""
        try:
            mesh = trimesh.load(file_path)
            if mesh is None or not isinstance(mesh, trimesh.Trimesh):
                print(f"Cannot load valid mesh from file {file_path}")
                return None

            return mesh
        except Exception as e:
            print(f"Mesh loading failed: {e}")
            return None

    @staticmethod
    def load_mesh_from_step(file_path: str) -> Optional[trimesh.Trimesh]:
        """Load mesh from STEP file"""
        try:
            mesh = trimesh.load(file_path)
            if isinstance(mesh, trimesh.Scene):
                # If scene, merge all geometries
                mesh = trimesh.util.concatenate(mesh.dump())

            if mesh is None or not isinstance(mesh, trimesh.Trimesh):
                print(f"Cannot load valid mesh from file {file_path}")
                return None

            mesh.vertices *= 1000.0

            return mesh
        except Exception as e:
            print(f"Mesh loading failed: {e}")
            return None

    @staticmethod
    def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        mesh = mesh.process(validate=True)
        if not mesh.is_watertight:
            mesh.fill_holes()
        return mesh

    @staticmethod
    def scale_to_sphere(mesh: trimesh.Trimesh, radius=1.0) -> tuple:
        """Scale model to fit inside a sphere of given radius"""
        if mesh is None:
            return None, np.eye(4)

        # Calculate the centroid of the mesh vertices
        centroid = mesh.vertices.mean(axis=0)

        # Calculate the maximum distance from the centroid to any vertex
        max_distance = np.max(np.linalg.norm(mesh.vertices - centroid, axis=1))

        # Compute scaling factor to fit the mesh inside the sphere of given radius
        if max_distance > 0:
            scale = radius / max_distance
        else:
            scale = 1.0  # Avoid division by zero if mesh is a single point

        # Construct translation matrix to move the centroid to the origin
        T = np.eye(4)
        T[:3, 3] = -centroid

        # Construct uniform scaling matrix
        S = np.eye(4) * scale
        S[3, 3] = 1.0  # Homogeneous coordinate remains 1

        # Combine translation and scaling: first translate, then scale
        transform = S @ T

        # Create a copy of the mesh and apply the transformation
        mesh_scaled = mesh.copy()
        mesh_scaled.apply_transform(transform)

        # Return the scaled mesh and the transformation matrix
        return mesh_scaled, transform

    @staticmethod
    def scale_to_cube1(mesh: trimesh.Trimesh, edge_length=1.0) -> tuple:
        # Calculate the centroid of the mesh vertices
        center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0

        # Determine the maximum extent (largest side length) of the bounding box
        max_extent = max(mesh.extents)
        if max_extent == 0:
            scale = 1.0
        else:
            scale = edge_length / max_extent

        # Construct translation matrix to move the mesh center to the origin
        T = np.eye(4)
        T[:3, 3] = -center

        # Construct uniform scaling matrix
        S = np.eye(4) * scale
        S[3, 3] = 1.0

        # Combine scaling and translation: scale after translation
        transform = S @ T

        # Create a copy of the mesh and apply the transformation
        mesh_scaled = mesh.copy()
        mesh_scaled.apply_transform(transform)

        return mesh_scaled, transform

    @staticmethod
    def scale_to_cube2(mesh: trimesh.Trimesh, edge_length=1.0) -> tuple:
        """Scale model to fit inside a cube with given edge length"""
        if mesh is None:
            return None, np.eye(4)

        # Calculate the centroid of the mesh vertices
        centroid = mesh.vertices.mean(axis=0)

        # Construct translation matrix to move centroid to origin
        T = np.eye(4)
        T[:3, 3] = -centroid

        # Calculate bounding box after translation (centered)
        vertices_centered = mesh.vertices - centroid
        min_bound = vertices_centered.min(axis=0)
        max_bound = vertices_centered.max(axis=0)
        sizes = max_bound - min_bound

        # Find largest dimension size
        max_size = np.max(sizes)

        # Compute scale factor to fit mesh inside cube of specified edge length
        if max_size > 0:
            scale = edge_length / max_size
        else:
            scale = 1.0  # Avoid division by zero if mesh is degenerate

        # Construct scaling matrix
        S = np.eye(4) * scale
        S[3, 3] = 1.0  # Homogeneous coordinate

        # Combine translation and scaling (scale after translation)
        transform = S @ T

        # Apply transformation to a copy of the mesh
        mesh_scaled = mesh.copy()
        mesh_scaled.apply_transform(transform)

        return mesh_scaled, transform

    @staticmethod
    def align_with_pca(mesh: trimesh.Trimesh) -> tuple:
        """Align model pose using PCA"""
        if mesh is None:
            return None, np.eye(4)

        vertices = mesh.vertices
        centroid = np.mean(vertices, axis=0)
        centered_vertices = vertices - centroid

        # Use SVD for eigen decomposition, more stable than covariance eigen decomposition
        U, S, Vt = np.linalg.svd(centered_vertices.T)
        eigen_vectors = U

        # Ensure right-handed coordinate system
        if np.linalg.det(eigen_vectors) < 0:
            eigen_vectors[:, 2] *= -1

        # Ensure consistency of principal axis directions
        for i in range(3):
            if eigen_vectors[i, i] < 0:
                eigen_vectors[:, i] *= -1

        # Construct transformation matrix:
        # 1. Translate vertices by -centroid (move centroid to origin)
        T = np.eye(4)
        T[:3, 3] = -centroid

        # 2. Rotate using eigen_vectors (principal axes)
        R = np.eye(4)
        R[:3, :3] = eigen_vectors

        # Combine transformations: rotation after translation
        transform = R @ T

        # Apply transform to a copy of the mesh
        mesh_aligned = mesh.copy()
        mesh_aligned.apply_transform(transform)

        return mesh_aligned, transform

    @staticmethod
    def normalize_volume(mesh: trimesh.Trimesh, target_volume: float = 1.0) -> tuple:
        """Normalize model volume to target (default unit volume)"""
        if mesh is None:
            return None, np.eye(4)

        if mesh.volume <= 0:
            # Cannot scale if volume is zero or negative, return identity transform
            return mesh.copy(), np.eye(4)

        # Calculate scale factor (volume scales with cube of linear size)
        scale_factor = (target_volume / mesh.volume) ** (1 / 3)

        # Construct scaling matrix
        transform = np.eye(4) * scale_factor
        transform[3, 3] = 1.0  # Homogeneous coordinate remains 1

        # Apply scale transformation to a copy of mesh
        mesh_scaled = mesh.copy()
        mesh_scaled.apply_transform(transform)

        return mesh_scaled, transform

    @staticmethod
    def process_mesh(mesh: trimesh.Trimesh) -> tuple:
        """Align mesh"""
        if mesh is None:
            return None
        mesh = MeshHandler.repair_mesh(mesh)
        # mesh, trans1 = MeshHandler.align_with_pca(mesh)
        # mesh, trans2 = MeshHandler.scale_to_sphere(mesh)
        mesh, trans2 = MeshHandler.scale_to_cube1(mesh)
        # transform = trans2 @ trans1
        transform = trans2
        return mesh, transform

    def sample_surface_points(self, mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
        """Sample points from mesh surface"""
        try:
            if not mesh.is_watertight:
                print(
                    "Warning: mesh is not watertight, sampling quality may be affected")

            # Try even surface sampling
            try:
                points, _ = trimesh.sample.sample_surface_even(
                    mesh, self.n_points)
                if len(points) == 0:
                    raise ValueError("Even sampling returned empty results")
            except Exception as e:
                print(
                    f"Even surface sampling failed, using random sampling: {str(e)}")
                points, _ = trimesh.sample.sample_surface(mesh, self.n_points)

            if len(points) == 0:
                print("Surface sampling failed, trying vertices")
                points = mesh.vertices
                if len(points) > self.n_points:
                    # Randomly choose n_points vertices
                    indices = np.random.choice(
                        len(points), self.n_points, replace=False)
                    points = points[indices]

            return points

        except Exception as e:
            print(f"Surface sampling failed: {e}")
            return None

    def sample_surface_points_with_normals(self, mesh: trimesh.Trimesh) -> Tuple[
            Optional[np.ndarray], Optional[np.ndarray]]:
        """Sample points and normals from mesh surface"""
        try:
            # Ensure mesh has normal info
            if not hasattr(mesh, 'face_normals') or mesh.face_normals is None:
                mesh.generate_face_normals()  # generate face normals

            # Sample points and face indices
            points, face_indices = trimesh.sample.sample_surface(
                mesh, self.n_points)

            # Get normals
            normals = mesh.face_normals[face_indices]

            return points, normals

        except Exception as e:
            print(f"Surface sampling with normals failed: {e}")
            return None, None

    @staticmethod
    def extract_solids_from_shape(shape: TopoDS_Shape) -> List[TopoDS_Solid]:
        """Extract all entities from a composite shape"""
        solids = []
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            solids.append(topods.Solid(explorer.Current()))
            explorer.Next()

        if not solids:
            iterator = TopoDS_Iterator(shape)
            while iterator.More():
                sub_solids = MeshHandler.extract_solids_from_shape(
                    iterator.Value())
                solids.extend(sub_solids)
                iterator.Next()

        return solids

    def sample_volume_points(self, shape):
        volume_points = []
        solids = MeshHandler.extract_solids_from_shape(shape)

        if len(solids) == 0:
            solids = [shape]

        for solid in solids:
            solid_classifier = BRepClass3d_SolidClassifier(solid)
            bbox = Bnd_Box()
            brepbndlib.Add(solid, bbox)

            if bbox.IsVoid():
                continue

            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

            # Estimate the ratio of total volume to solid volume
            total_volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)

            # Adjust the sampling times according to the volume
            adjusted_samples = int(self.n_points * (1 + total_volume / 1000))
            inside_points = 0

            # Random sampling
            for _ in range(adjusted_samples):
                x = xmin + (xmax - xmin) * np.random.random()
                y = ymin + (ymax - ymin) * np.random.random()
                z = zmin + (zmax - zmin) * np.random.random()

                pt = gp_Pnt(x, y, z)
                solid_classifier.Perform(pt, 1e-6)
                if solid_classifier.State() == TopAbs_IN:
                    volume_points.append([x, y, z])
                    inside_points += 1

                # If enough points have been sampled, end early
                if inside_points >= self.n_points:
                    break

        return volume_points

    def mesh_to_pointcloud(self, mesh: trimesh.Trimesh) -> o3d.geometry.PointCloud:
        points, _ = trimesh.sample.sample_surface(mesh, self.n_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Calculate average distance between points to dynamically set normal estimation radius
        avg_dist = self.compute_average_point_distance(pcd)
        # Use 2~3 times average distance
        radius_normals = avg_dist * 2.5

        # Estimate normals using KDTree search within the radius
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normals, max_nn=30)
        )
        return pcd

    def preprocess_pointcloud(self, pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)

        # Recalculate average point distance after downsampling for normal estimation radius
        avg_dist = self.compute_average_point_distance(pcd_down)
        radius_normals = avg_dist * 2.5

        # Estimate normals for downsampled point cloud
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normals, max_nn=30))

        # Compute FPFH (Fast Point Feature Histogram) features for registration
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=avg_dist * 5, max_nn=100)
        )
        return pcd_down, fpfh

    def compute_average_point_distance(self, pcd: o3d.geometry.PointCloud) -> float:
        # Build a KDTree for fast nearest neighbor search
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        distances = []
        for i in range(len(pcd.points)):
            # Find the 2 nearest neighbors (including itself)
            [_, _, d] = pcd_tree.search_knn_vector_3d(pcd.points[i], 2)
            if len(d) == 2:
                # Append the distance to the nearest neighbor (excluding self)
                distances.append(np.sqrt(d[1]))
        if len(distances) == 0:
            return 0.1  # Return default small value to avoid division by zero or zero radius
        return float(np.mean(distances))

    def align_with_icp(self, source_mesh: trimesh.Trimesh, target_mesh: trimesh.Trimesh,
                       max_iterations=100, tolerance=1e-6) -> tuple:
        """Align two mesh models using ICP algorithm"""
        if source_mesh is None or target_mesh is None:
            return source_mesh, np.eye(4)

        # Helper function to calculate bounding box diagonal length of a mesh
        def get_bbox_diag(m):
            return np.linalg.norm(m.bounds[1] - m.bounds[0])

        source_diag = get_bbox_diag(source_mesh)
        target_diag = get_bbox_diag(target_mesh)
        # Determine voxel size as 2% of smaller bounding box diagonal for consistent scale
        voxel_size = min(source_diag, target_diag) * 0.02
        # print(f"Calculated voxel_size: {voxel_size}")

        # Convert meshes to point clouds with estimated normals
        source_pcd = self.mesh_to_pointcloud(source_mesh)
        target_pcd = self.mesh_to_pointcloud(target_mesh)

        # Downsample and compute FPFH features for source and target point clouds
        source_down, source_fpfh = self.preprocess_pointcloud(
            source_pcd, voxel_size)
        target_down, target_fpfh = self.preprocess_pointcloud(
            target_pcd, voxel_size)

        # Orient normals consistently for better registration stability
        source_down.orient_normals_consistent_tangent_plane(30)
        target_down.orient_normals_consistent_tangent_plane(30)

        # Set distance threshold for RANSAC correspondence based on voxel size
        distance_threshold = voxel_size * 1.5

        # Perform global registration using RANSAC based on feature matching
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down,
            source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                4000000, 500)
        )
        # print("RANSAC fitness:", result_ransac.fitness)
        # print("RANSAC inlier RMSE:", result_ransac.inlier_rmse)
        # print("RANSAC transformation:\n", result_ransac.transformation)

        # Check RANSAC result fitness, warn if poor and return identity transform
        if result_ransac.fitness < 0.1:
            print("Warning: Poor RANSAC result, ICP may fail.")
            return source_mesh, np.eye(4)

        # Use ICP for fine alignment, point-to-plane metric is generally more precise
        distance_threshold_icp = voxel_size * 1.5
        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down,
            max_correspondence_distance=distance_threshold_icp,
            init=result_ransac.transformation,
            # init=np.identity(4),  # initial transform matrix is identity
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations,
                relative_rmse=tolerance
            )
        )
        # print("ICP fitness:", result_icp.fitness)
        # print("ICP inlier RMSE:", result_icp.inlier_rmse)
        # print("ICP transformation:\n", result_icp.transformation)

        # Apply the ICP transformation to a copy of the source mesh
        aligned_mesh = source_mesh.copy()
        aligned_mesh.apply_transform(result_icp.transformation)

        return aligned_mesh, result_icp.transformation

    @staticmethod
    def apply_scale_to_shape(shape: TopoDS_Shape, transformation: np.ndarray) -> TopoDS_Shape:
        if shape is None or shape.IsNull() or transformation is None:
            return shape

        if transformation.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4.")

        # Calculate uniform scale factor from the first column vector
        scale_factor = np.linalg.norm(transformation[:3, 0])
        if not np.allclose(
            [np.linalg.norm(transformation[:3, i]) for i in range(3)],
            scale_factor,
            atol=1e-6,
        ):
            raise ValueError("Non-uniform scaling is not supported")

        trans_vec = transformation[:3, 3]

        trsf = gp_Trsf()
        # Set uniform scale about origin
        trsf.SetScale(gp_Pnt(0, 0, 0), scale_factor)
        trsf.SetTranslationPart(gp_Vec(*trans_vec))

        brep_trsf = BRepBuilderAPI_Transform(shape, trsf, True)
        brep_trsf.Build()
        if not brep_trsf.IsDone():
            raise RuntimeError("Failed to apply scale transform to shape.")

        return brep_trsf.Shape()

    @staticmethod
    def apply_rigid_transform_to_shape(shape: TopoDS_Shape, transformation: np.ndarray) -> TopoDS_Shape:
        if shape is None or shape.IsNull() or transformation is None:
            return shape

        if transformation.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4.")

        rot_mat = transformation[:3, :3]
        trans_vec = transformation[:3, 3]

        # Check that rotation matrix is orthogonal and has determinant 1
        if not np.allclose(rot_mat @ rot_mat.T, np.eye(3), atol=1e-6):
            raise ValueError("Rotation matrix is not orthogonal")
        if not np.isclose(np.linalg.det(rot_mat), 1.0, atol=1e-6):
            raise ValueError("Rotation matrix determinant is not 1")

        trsf = gp_Trsf()
        trsf.SetValues(
            rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], trans_vec[0],
            rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], trans_vec[1],
            rot_mat[2, 0], rot_mat[2, 1], rot_mat[2, 2], trans_vec[2]
        )

        brep_trsf = BRepBuilderAPI_Transform(shape, trsf, True)
        brep_trsf.Build()
        if not brep_trsf.IsDone():
            raise RuntimeError("Failed to apply rigid transform to shape.")

        return brep_trsf.Shape()

    @staticmethod
    def visualize_point_clouds(points1: np.ndarray,
                               points2: np.ndarray,
                               normals1: Optional[np.ndarray] = None,
                               normals2: Optional[np.ndarray] = None,
                               show_normals: bool = False):
        """Visualize two surface point clouds"""
        # Create point cloud objects
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        pcd1.paint_uniform_color([1, 0, 0])  # Red

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.paint_uniform_color([0, 0, 1])  # Blue

        # Add normals
        if show_normals and normals1 is not None and normals2 is not None:
            pcd1.normals = o3d.utility.Vector3dVector(normals1)
            pcd2.normals = o3d.utility.Vector3dVector(normals2)

        # Visualization
        geometries = [pcd1, pcd2]
        o3d.visualization.draw_geometries(geometries,
                                          window_name='Surface Point Clouds Comparison',
                                          width=1200, height=800,
                                          point_show_normal=show_normals)


class MetricsCalculator:
    """Metrics calculator"""

    def __init__(self, sampling_config: Optional[SamplingConfig] = None):
        self.config = sampling_config or SamplingConfig()
        self.mesh_handler = MeshHandler(self.config)

    @staticmethod
    def compute_chamfer_distance(sample_pcs: np.ndarray, ref_pcs: np.ndarray) -> Tuple[float, float, float]:
        """Compute Chamfer distance"""
        if len(sample_pcs) == 0 or len(ref_pcs) == 0:
            raise ValueError("Input point cloud cannot be empty.")

        # Fast nearest neighbor search
        # Distances from sample_pcs to ref_pcs
        nbrs1 = NearestNeighbors(
            n_neighbors=1, algorithm='kd_tree').fit(ref_pcs)
        distances1, _ = nbrs1.kneighbors(sample_pcs)
        distances1 = distances1.flatten()
        mean_dist_1_to_2 = np.mean(distances1)

        # Distances from ref_pcs to sample_pcs
        nbrs2 = NearestNeighbors(
            n_neighbors=1, algorithm='kd_tree').fit(sample_pcs)
        distances2, _ = nbrs2.kneighbors(ref_pcs)
        distances2 = distances2.flatten()
        mean_dist_2_to_1 = np.mean(distances2)

        # Chamfer distance is average of two directional distances
        chamfer_dist = (mean_dist_1_to_2 + mean_dist_2_to_1) / 2.0

        return chamfer_dist, mean_dist_1_to_2, mean_dist_2_to_1

    @staticmethod
    def compute_hausdorff_distance(sample_pcs: np.ndarray, ref_pcs: np.ndarray) -> Tuple[float, float, float]:
        """Compute Hausdorff distance"""
        if len(sample_pcs) == 0 or len(ref_pcs) == 0:
            raise ValueError("Input point cloud cannot be empty.")

        # Nearest neighbor distances
        nbrs1 = NearestNeighbors(
            n_neighbors=1, algorithm='kd_tree').fit(ref_pcs)
        distances1, _ = nbrs1.kneighbors(sample_pcs)
        distances1 = distances1.flatten()
        max_dist_1_to_2 = np.max(distances1)

        nbrs2 = NearestNeighbors(
            n_neighbors=1, algorithm='kd_tree').fit(sample_pcs)
        distances2, _ = nbrs2.kneighbors(ref_pcs)
        distances2 = distances2.flatten()
        max_dist_2_to_1 = np.max(distances2)

        # Hausdorff distance is max of max distances in both directions
        hausdorff_dist = max(max_dist_1_to_2, max_dist_2_to_1)

        return hausdorff_dist, max_dist_1_to_2, max_dist_2_to_1

    @staticmethod
    def compute_emd(sample_pcs: np.ndarray, ref_pcs: np.ndarray) -> float:
        """Compute Earth Mover's Distance (EMD)"""
        try:
            import ot
        except ImportError:
            print("Warning: POT library not installed, skipping EMD calculation. Run: pip install POT")
            return np.nan

        try:
            n_points_a = sample_pcs.shape[0]
            n_points_b = ref_pcs.shape[0]

            # Compute cost matrix of shape (n_points, n_points)
            # Each cost matrix corresponds to pairwise distances between points in a and b
            costs = np.linalg.norm(
                sample_pcs[:, None, :] - ref_pcs[None, :, :], axis=-1)

            # Uniform weights for points in each point cloud
            weights_a = np.ones(n_points_a) / n_points_a
            weights_b = np.ones(n_points_b) / n_points_b

            # Compute squared Earth Mover's Distance between distributions
            emd = ot.emd2(weights_a, weights_b, costs)

            return float(emd)
        except Exception as e:
            print(f"Warning: EMD calculation failed: {e}")
            return np.nan

    @staticmethod
    def compute_rms_error(sample_pcs: np.ndarray, ref_pcs: np.ndarray) -> float:
        """Compute Root Mean Square Error (RMS)"""
        if len(sample_pcs) == 0 or len(ref_pcs) == 0:
            raise ValueError("Input point cloud cannot be empty.")

        # For each point find nearest neighbor
        nbrs = NearestNeighbors(
            n_neighbors=1, algorithm='kd_tree').fit(ref_pcs)
        distances, _ = nbrs.kneighbors(sample_pcs)
        distances = distances.flatten()

        # Compute RMS error
        rms = np.sqrt(np.mean(np.square(distances)))

        return rms

    @staticmethod
    def compute_volume(shape) -> float:
        if shape is None or shape.IsNull():
            return np.nan

        vol = 0.0
        props = GProp_GProps()
        try:
            if shape.ShapeType() == TopAbs_SOLID:
                brepgprop.VolumeProperties(shape, props)
                vol += abs(props.Mass())
            else:
                exp = TopExp_Explorer(shape, TopAbs_SOLID)
                solids_found = False
                while exp.More():
                    solids_found = True
                    solid = topods.Solid(exp.Current())
                    brepgprop.VolumeProperties(solid, props)
                    vol += abs(props.Mass())
                    exp.Next()
                if not solids_found:
                    return np.nan
        except Exception as e:
            return np.nan

        if vol < 0:
            return np.nan

        return vol

    @staticmethod
    def compute_iou_from_shape(shape1, shape2) -> float:
        vol1 = MetricsCalculator.compute_volume(shape1)
        vol2 = MetricsCalculator.compute_volume(shape2)

        if np.isnan(vol1) or np.isnan(vol2):
            return np.nan

        if vol1 <= 1e-9 or vol2 <= 1e-9:  # Use a small tolerance for volume
            if vol1 <= 1e-9 and vol2 <= 1e-9:
                return 1.0
            return 0.0

        # Calculate intersection
        common_op = BRepAlgoAPI_Common(shape1, shape2)
        common_op.Build()
        if not common_op.IsDone():
            # Fallback to union-based calculation if common fails
            union_op = BRepAlgoAPI_Fuse(shape1, shape2)
            union_op.Build()
            union_shape = union_op.Shape() if union_op.IsDone() else None
            if union_shape is None or union_shape.IsNull():
                return np.nan
            vol_union = MetricsCalculator.compute_volume(
                union_shape) if union_shape and not union_shape.IsNull() else (vol1 + vol2)
            if np.isnan(vol_union):
                return np.nan
            vol_inter = vol1 + vol2 - vol_union
        else:
            common_shape = common_op.Shape()
            vol_inter = MetricsCalculator.compute_volume(common_shape)
            if np.isnan(vol_inter):
                return np.nan

        # Compute union
        vol_union = vol1 + vol2 - vol_inter

        # Compute Iou
        if vol_union > 1e-9:
            iou_value = vol_inter / vol_union
            # Clamp value between 0 and 1
            return max(0.0, min(1.0, iou_value))
        else:
            # If union is zero, IoU is 1 if intersection is also (near) zero, else 0.
            return 1.0 if vol_inter < 1e-9 else 0.0

    @staticmethod
    def compute_iou_from_mesh1(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, voxel_size: float = 0.02) -> float:
        """Compute Intersection over Union (IoU)"""
        try:
            # Split connected parts to avoid Boolean operation failure
            meshes1 = mesh1.split()
            meshes2 = mesh2.split()

            # Calculate the intersection volume
            intersection_volume = 0.0
            for m1 in meshes1:
                for m2 in meshes2:
                    try:
                        inter = m1.intersection(m2)
                        if inter is not None:
                            intersection_volume += inter.volume
                    except Exception as e:
                        return np.nan

            # Calculate their respective volumes
            volume1 = sum(m.volume for m in meshes1)
            volume2 = sum(m.volume for m in meshes2)

            union_volume = volume1 + volume2 - intersection_volume
            if union_volume == 0:
                return np.nan

            iou = intersection_volume / union_volume
            return iou

        except Exception as e:
            print(f"Error calculating IoU: {str(e)}")
            return np.nan

    @staticmethod
    def compute_iou_from_mesh2(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, voxel_size: float = 0.02) -> float:
        """Compute Intersection over Union (IoU)"""
        try:
            # Compute bounds
            bounds1 = mesh1.bounds
            bounds2 = mesh2.bounds

            min_bound = np.minimum(bounds1[0], bounds2[0])
            max_bound = np.maximum(bounds1[1], bounds2[1])

            # Add margin
            margin = voxel_size * 2
            min_bound = min_bound - margin
            max_bound = max_bound + margin

            # Compute sampling dimensions
            dims = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
            total_points = np.prod(dims)

            # Dynamic adjustment strategy
            max_memory_points = 5000000  # 5 million points memory limit

            if total_points <= max_memory_points:
                # Small scale: direct computation
                return MetricsCalculator._compute_iou_direct_from_mesh(mesh1, mesh2, min_bound, max_bound, voxel_size)
            else:
                # Large scale: batch computation
                return MetricsCalculator._compute_iou_batched_from_mesh(mesh1, mesh2, min_bound, max_bound, dims)

        except Exception as e:
            print(f"Error calculating IoU: {str(e)}")
            return np.nan

    @staticmethod
    def _compute_iou_direct_from_mesh(mesh1, mesh2, min_bound, max_bound, voxel_size):
        """Directly compute IoU (for small scale)"""
        # Generate all sample points
        x = np.arange(min_bound[0], max_bound[0], voxel_size)
        y = np.arange(min_bound[1], max_bound[1], voxel_size)
        z = np.arange(min_bound[2], max_bound[2], voxel_size)

        # Use more efficient meshgrid method
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        # Check all points at once
        inside1 = mesh1.contains(points)
        inside2 = mesh2.contains(points)

        # Compute IoU
        intersection = np.sum(inside1 & inside2)
        union = np.sum(inside1 | inside2)

        return float(intersection) / float(union) if union > 0 else np.nan

    @staticmethod
    def _compute_iou_batched_from_mesh(mesh1, mesh2, min_bound, max_bound, dims):
        """Batch compute IoU (for large scale)"""
        # Optimized batching: slice along Z axis to reduce memory fragmentation
        z_coords = np.linspace(min_bound[2], max_bound[2], dims[2])
        x_coords = np.linspace(min_bound[0], max_bound[0], dims[0])
        y_coords = np.linspace(min_bound[1], max_bound[1], dims[1])

        # Number of points per Z slice
        points_per_z = dims[0] * dims[1]
        # max 3 million points per batch
        max_z_slices = max(1, 3000000 // points_per_z)

        intersection_total = 0
        union_total = 0

        for z_start in range(0, dims[2], max_z_slices):
            z_end = min(z_start + max_z_slices, dims[2])
            z_batch = z_coords[z_start:z_end]

            # Create meshgrid for current batch
            xx, yy, zz = np.meshgrid(
                x_coords, y_coords, z_batch, indexing='ij')
            points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

            # Check points inside meshes
            inside1 = mesh1.contains(points)
            inside2 = mesh2.contains(points)

            # Accumulate results
            intersection_total += np.sum(inside1 & inside2)
            union_total += np.sum(inside1 | inside2)

            # Clear memory
            del points, inside1, inside2, xx, yy, zz

        return float(intersection_total) / float(union_total) if union_total > 0 else np.nan

    @staticmethod
    def compute_adaptive_threshold(pcs: np.ndarray, factor: float = 0.02) -> float:
        bbox_min = np.min(pcs, axis=0)
        bbox_max = np.max(pcs, axis=0)
        scale = np.linalg.norm(bbox_max - bbox_min)
        return scale * factor

    @staticmethod
    def compute_matching_rate(sample_pcs: np.ndarray, ref_pcs: np.ndarray, threshold: float = None,
                              threshold_factor: float = 0.02) -> float:
        """Compute point cloud matching rate within a given threshold"""
        if len(sample_pcs) == 0 or len(ref_pcs) == 0:
            raise ValueError("Input point cloud cannot be empty.")

        if threshold is None:
            threshold = MetricsCalculator.compute_adaptive_threshold(
                ref_pcs, factor=threshold_factor)

        # For each point find nearest neighbor
        nbrs = NearestNeighbors(
            n_neighbors=1, algorithm='kd_tree').fit(ref_pcs)
        distances, _ = nbrs.kneighbors(sample_pcs)
        distances = distances.flatten()

        # Compute proportion of points within threshold
        matching_rate = np.mean(distances < threshold)

        return matching_rate

    @staticmethod
    def entropy_of_occupancy_grid(pcloud: np.ndarray, grid_resolution: int, in_sphere: bool = False) -> Tuple[
            float, np.ndarray]:
        """Compute entropy of occupancy grid"""
        epsilon = 1e-5
        bound = 1 + epsilon

        # Check if point clouds are in unit cube
        if abs(np.max(pcloud)) > bound or abs(np.min(pcloud)) > bound:
            warnings.warn('Point-clouds are not in unit cube.')

        # Check if point clouds are in unit sphere
        if in_sphere and np.max(np.sqrt(np.sum(pcloud ** 2, axis=1))) > bound:
            warnings.warn('Point-clouds are not in unit sphere.')

        # Generate grid coordinates
        grid_coordinates, _ = MetricsCalculator.unit_cube_grid_point_cloud(
            grid_resolution, in_sphere)
        grid_coordinates = grid_coordinates.reshape(-1, 3)

        # Initialize counters
        grid_counters = np.zeros(len(grid_coordinates))
        grid_bernoulli_rvars = np.zeros(len(grid_coordinates))

        # Fit nearest neighbor model
        nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

        # Find nearest grid cell indices for each point in the point cloud
        _, indices = nn.kneighbors(pcloud)
        indices = np.squeeze(indices)

        # Increment grid counters for all points
        for i in indices:
            grid_counters[i] += 1

        # Increment Bernoulli random variable (count unique grid cells occupied)
        unique_indices = np.unique(indices)
        for i in unique_indices:
            grid_bernoulli_rvars[i] += 1

        # Compute entropy over Bernoulli random variables (grid cells)
        acc_entropy = 0.0
        n = float(len(pcloud))  # number of points in the single point cloud

        for g in grid_bernoulli_rvars:
            if g > 0:
                p = float(g) / n
                acc_entropy += entropy([p, 1.0 - p])

        return acc_entropy / len(grid_counters), grid_counters

    @staticmethod
    def unit_cube_grid_point_cloud(resolution: int, clip_sphere: bool = False) -> Tuple[np.ndarray, float]:
        """Generate unit cube grid point cloud"""
        grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
        spacing = 1.0 / float(resolution - 1) * 2

        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    grid[i, j, k, 0] = i * spacing - 1.0
                    grid[i, j, k, 1] = j * spacing - 1.0
                    grid[i, j, k, 2] = k * spacing - 1.0

        if clip_sphere:
            grid = grid.reshape(-1, 3)
            grid = grid[np.linalg.norm(grid, axis=1) <= 1.0]

        return grid, spacing

    @staticmethod
    def compute_jsd(sample_pcs: np.ndarray, ref_pcs: np.ndarray, in_unit_sphere: bool = True,
                    resolution: int = 28) -> float:
        """Compute Jensen-Shannon divergence"""
        P = MetricsCalculator.entropy_of_occupancy_grid(
            sample_pcs, resolution, in_unit_sphere
        )[1]

        Q = MetricsCalculator.entropy_of_occupancy_grid(
            ref_pcs, resolution, in_unit_sphere
        )[1]
        if np.any(P < 0) or np.any(Q < 0):
            raise ValueError('Negative values.')
        if len(P) != len(Q):
            raise ValueError('Non equal size.')

        P_ = P / np.sum(P)
        Q_ = Q / np.sum(Q)

        e1 = entropy(P_, base=2)
        e2 = entropy(Q_, base=2)
        e_sum = entropy((P_ + Q_) / 2.0, base=2)
        res = e_sum - ((e1 + e2) / 2.0)
        return res

    @staticmethod
    def compute_nc(sample_pcs: np.ndarray, ref_pcs: np.ndarray,
                   sample_normals: np.ndarray, ref_normals: np.ndarray) -> float:
        """Compute Normal Consistency"""
        # Compute distance matrix between all sample and reference points: shape (M, N)
        # Efficient broadcasting:
        # sample_pcs_expanded shape: (M, 1, 3)
        # ref_pcs_expanded shape: (1, N, 3)
        sample_pcs_exp = sample_pcs[:, np.newaxis, :]  # (M, 1, 3)
        ref_pcs_exp = ref_pcs[np.newaxis, :, :]  # (1, N, 3)
        diff = sample_pcs_exp - ref_pcs_exp  # (M, N, 3)
        distances = np.linalg.norm(diff, axis=2)  # (M, N)

        # For each sample point, find closest reference point index
        matched_ref_indices = np.argmin(distances, axis=1)  # (M,)

        # For each reference point, find closest sample point index
        matched_sample_indices = np.argmin(distances, axis=0)  # (N,)

        # Compute normal consistency from sample to reference
        # dot product between sample normals and matched reference normals
        nl_dot = np.sum(sample_normals *
                        ref_normals[matched_ref_indices], axis=1)  # (M,)
        nl = np.mean(np.abs(nl_dot))

        # Compute normal consistency from reference to sample
        nr_dot = np.sum(
            # (N,)
            ref_normals * sample_normals[matched_sample_indices], axis=1)
        nr = np.mean(np.abs(nr_dot))

        # Average the two consistency scores
        normal_consistency = (nl + nr) / 2

        return float(normal_consistency)

    @staticmethod
    def compute_metrics(ext: str,
                        mesh1: trimesh.Trimesh,
                        mesh2: trimesh.Trimesh,
                        points1: np.ndarray,
                        points2: np.ndarray,
                        normals1: Optional[np.ndarray] = None,
                        normals2: Optional[np.ndarray] = None,
                        matching_threshold: float = None,
                        resolution: int = 28,
                        shape1: Optional[TopoDS_Shape] = None,
                        shape2: Optional[TopoDS_Shape] = None) -> MetricsResult:
        """Compute various metrics between two point clouds"""
        start_time = time.time()
        result = MetricsResult()
        result._matching_threshold = matching_threshold

        try:
            # Compute Chamfer distance
            chamfer_dist, dist_1_to_2, dist_2_to_1 = MetricsCalculator.compute_chamfer_distance(
                points1, points2)
            result.chamfer_distance = chamfer_dist
            result._chamfer_x_to_y = dist_1_to_2
            result._chamfer_y_to_x = dist_2_to_1

            # Compute Hausdorff distance
            hausdorff_dist, max_dist_1_to_2, max_dist_2_to_1 = MetricsCalculator.compute_hausdorff_distance(
                points1, points2)
            result.hausdorff_distance = hausdorff_dist
            result._hausdorff_x_to_y = max_dist_1_to_2
            result._hausdorff_y_to_x = max_dist_2_to_1

            # Compute Earth Mover's distance
            result.earth_mover_distance = MetricsCalculator.compute_emd(
                points1, points2)

            # Compute RMS error
            result.rms_error = MetricsCalculator.compute_rms_error(
                points1, points2)

            # Compute IOU
            if USE_VOXEL_IOU:
                result.iou_voxel = MetricsCalculator.compute_iou_from_mesh1(
                    mesh1, mesh2)
                # result.iou_voxel = MetricsCalculator.compute_iou_from_mesh2(
                #     mesh1, mesh2)
            if USE_BREP_IOU and ext == "step":
                result.iou_brep = MetricsCalculator.compute_iou_from_shape(
                    shape1, shape2)
            else:
                result.iou_brep = 0.0

            # Compute matching rate
            result.matching_rate = MetricsCalculator.compute_matching_rate(
                points1, points2, matching_threshold)

            # Compute coverage rate
            result.coverage_rate = MetricsCalculator.compute_matching_rate(
                points2, points1, matching_threshold)

            # Compute JSD
            # result.jsd = MetricsCalculator.compute_jsd(points1, points2,
            #                                            in_unit_sphere=True, resolution=resolution)

            # Compute normal consistency (if normals are given)
            if normals1 is not None and normals2 is not None:
                result.normal_consistency = MetricsCalculator.compute_nc(
                    points1, points2, normals1, normals2)

        except Exception as e:
            print(f"Error computing metrics: {e}")
            traceback.print_exc()

        # Record computation time
        result.computation_time = time.time() - start_time

        return result

    def compute_files_metrics(self,
                              ext: str,
                              file1: str,
                              file2: str,
                              matching_threshold: float = None,
                              include_normals: bool = False,
                              use_icp: bool = True) -> Optional[MetricsResult]:
        """Compute various metrics on surface point clouds from two STEP or STL files"""
        try:
            if ext == "stl":
                mesh1 = MeshHandler.load_mesh_from_stl(file1)
                mesh2 = MeshHandler.load_mesh_from_stl(file2)
            elif ext == "step":
                mesh1 = MeshHandler.load_mesh_from_step(file1)
                mesh2 = MeshHandler.load_mesh_from_step(file2)
            else:
                return None
            
            if mesh1 is None or mesh2 is None:
                return None

            mesh1, trans1 = MeshHandler.process_mesh(mesh1)
            if mesh1 is None:
                return None

            mesh2, trans2 = MeshHandler.process_mesh(mesh2)
            if mesh2 is None:
                return None

            if use_icp:
                # print("Aligning point clouds with ICP...")
                mesh2, trans3 = self.mesh_handler.align_with_icp(mesh2, mesh1)
                # mesh2, trans4 = MeshHandler.scale_to_cube1(mesh2)

            if USE_BREP_IOU and ext == "step":
                shape1 = MeshHandler.load_shape_from_step(file1)
                shape2 = MeshHandler.load_shape_from_step(file2)
                shape1 = MeshHandler.apply_scale_to_shape(shape1, trans1)
                shape2 = MeshHandler.apply_scale_to_shape(shape2, trans2)
                if use_icp:
                    shape2 = MeshHandler.apply_rigid_transform_to_shape(
                        shape2, trans3)
                    # shape2 = MeshHandler.apply_scale_to_shape(shape2, trans4)
                # points1 = self.mesh_handler.sample_volume_points(shape1)
                # points2 = self.mesh_handler.sample_volume_points(shape2)
                # if points1 is not None and points2 is not None:
                #     MeshHandler.visualize_point_clouds(points1, points2)
                # else:
                #     print(f"Error visualizing point clouds: {e}")
            else:
                shape1 = None
                shape2 = None

            # Sample point clouds
            if include_normals:
                points1, normals1 = self.mesh_handler.sample_surface_points_with_normals(
                    mesh1)
                if points1 is None:
                    print("Sampling first mesh surface failed")
                    return None

                points2, normals2 = self.mesh_handler.sample_surface_points_with_normals(
                    mesh2)
                if points2 is None:
                    print("Sampling second mesh surface failed")
                    return None
            else:
                points1 = self.mesh_handler.sample_surface_points(mesh1)
                normals1 = None
                if points1 is None or len(points1) == 0:
                    print("Sampling first mesh surface failed")
                    return None

                points2 = self.mesh_handler.sample_surface_points(mesh2)
                normals2 = None
                if points2 is None or len(points2) == 0:
                    print("Sampling second mesh surface failed")
                    return None

            # print(f"Point cloud 1 size: {len(points1)}, Point cloud 2 size: {len(points2)}")
            result = MetricsCalculator.compute_metrics(
                ext,
                mesh1, mesh2,
                points1, points2,
                normals1, normals2,
                matching_threshold,
                shape1=shape1, shape2=shape2
            )

            # Visualize point clouds
            if VISUALIZE:
                try:
                    MeshHandler.visualize_point_clouds(
                        points1, points2, normals1, normals2, show_normals=False)
                except Exception as e:
                    print(f"Error visualizing point clouds: {e}")
            return result

        except Exception as e:
            print(
                f"Error computing metrics for files {file1} and {file2}: {e}")
            return None

    def batch_compute_metrics(self,
                              ext: str,
                              file_pairs: List[Tuple[str, str]],
                              progress_file: str,
                              matching_threshold: float = None,
                              include_normals: bool = False,
                              use_icp: bool = True,
                              batch_size: int = 100) -> List[Dict[str, Any]]:
        """Batch compute metrics for multiple pairs of STEP or STL files"""
        all_results, last_batch_index = load_checkpoint(progress_file)
        num_batches = (len(file_pairs) + batch_size - 1) // batch_size

        print(f"Completed batch indices: {last_batch_index + 1}.")
        for batch_index in range(last_batch_index + 1, num_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, len(file_pairs))
            batch_pairs = file_pairs[start:end]
            print(
                f"Starting processing batch {batch_index + 1}/{num_batches}, total {len(batch_pairs)} pairs.")

            if USE_MULTIPROCESSING:
                cpu_count = multiprocessing.cpu_count() // 2
                num_processes = min(cpu_count, len(batch_pairs))
                with multiprocessing.Pool(num_processes) as pool:
                    results = pool.starmap(self.compute_files_metrics,
                                           [(ext, file1, file2, matching_threshold, include_normals, use_icp)
                                            for file1, file2 in batch_pairs])
            else:
                results = [self.compute_files_metrics(ext, file1, file2, matching_threshold, include_normals,
                                                      use_icp)
                           for file1, file2 in batch_pairs]

            # Merge results
            for i, ((file1, file2), result) in enumerate(zip(batch_pairs, results)):
                if result is not None:
                    result_dict = result.to_dict()
                    result_dict['file1'] = file1.replace("\\", "/")
                    result_dict['file2'] = file2.replace("\\", "/")
                    result_dict['pair_index'] = start + i
                    all_results.append(result_dict)

            save_checkpoint(progress_file, all_results, batch_index)
            print(f"Finished batch {batch_index + 1}, checkpoint saved.")

        print('All batches processed.')

        grouped_results = {}
        for res in all_results:
            # 使用 get() 而不是 pop() 来避免修改原始数据
            f1 = res.get('file1', None)
            f2 = res.get('file2', None)
            if f1 is None or f2 is None:
                continue
            if f1 not in grouped_results:
                grouped_results[f1] = {}
            # 创建结果副本并移除文件路径信息
            res_copy = {k: v for k, v in res.items() if k not in ['file1', 'file2', 'pair_index']}
            grouped_results[f1][f2] = res_copy

        return all_results, grouped_results


def main():
    batch_size = 30
    n_points = 2000

    # Create calculator
    config = SamplingConfig(n_points=n_points)
    calculator = MetricsCalculator(sampling_config=config)

    # File paths
    ext = 'step'
    dataset = "test_samples"
    gt_folder = f"./{dataset}/gt"
    predict_folder = f"./{dataset}/predict"
    output_dir = f"./result/{dataset}"
    output_file = os.path.join(output_dir, "metrics_results.json")
    progress_file = os.path.join(output_dir, "progress.json")
    summary_file = os.path.join(output_dir, "metrics_summary.json")
    os.makedirs(output_dir, exist_ok=True)

    # Collect all file pairs
    file_pairs = []
    gt_files = glob.glob(os.path.join(gt_folder, f"*.{ext}"))
    predict_files = glob.glob(os.path.join(predict_folder, f"*.{ext}"))
    for gt_file in gt_files:
        # Create pairs for each predict file
        gt_base_name = os.path.basename(gt_file).split('.')[0]
        for predict_file in predict_files:
            predict_base_name = os.path.basename(
                predict_file).split('.')[0].split('_')[0]
            if predict_base_name == gt_base_name:
                file_pairs.append((gt_file, predict_file))
    if TEST_MODE:
        file_pairs = file_pairs[:10]
        # file_pairs = [file_pairs[-1]]
    print(f"Total number of pairs: {len(file_pairs)}")
    if len(file_pairs) == 0:
        print("No file pairs found. Exiting.")
        return

    # Batch compute metrics
    batch_results, grouped_results = calculator.batch_compute_metrics(
        ext,
        file_pairs,
        progress_file,
        include_normals=True,
        use_icp=True,
        batch_size=batch_size,
    )
    batch_results = [res for res in batch_results if not any(
        np.isnan(v) for v in res.values())]

    def get_values(key):
        values = [res.get(key, np.nan) for res in batch_results]
        return np.nan_to_num(values, nan=0.0)

    metrics = [
        'chamfer_distance', 'hausdorff_distance', 'earth_mover_distance', 'rms_error',
        'matching_rate', 'coverage_rate', 'normal_consistency', 'computation_time'
    ]
    average_metrics = {}
    std_metrics = {}
    for metric in metrics:
        vals = get_values(metric)
        average_metrics[metric] = np.mean(vals)
        std_metrics[f"{metric}_std"] = np.std(vals)

    if USE_VOXEL_IOU:
        average_metrics["iou_voxel"] = np.mean(get_values('iou_voxel'))
        std_metrics["iou_voxel_std"] = np.std(get_values('iou_voxel'))

    if USE_BREP_IOU and ext == "step":
        average_metrics["iou_brep"] = np.mean(get_values('iou_brep'))
        std_metrics["iou_brep_std"] = np.std(get_values('iou_brep'))

    # Merge averages and std deviations
    metrics_summary = {
        **average_metrics,
        **std_metrics,
        'total_samples': len(batch_results)
    }

    # Format output
    print("\n=== Metrics Statistics ===")
    print(f"Total samples: {metrics_summary['total_samples']}")
    print("\nBasic Metrics:")
    print(
        f"Chamfer Distance: {metrics_summary['chamfer_distance']:.6f} ± {metrics_summary['chamfer_distance_std']:.6f} ↓")
    print(
        f"Hausdorff Distance: {metrics_summary['hausdorff_distance']:.6f} ± {metrics_summary['hausdorff_distance_std']:.6f} ↓")
    print(
        f"Earth Mover's Distance: {metrics_summary['earth_mover_distance']:.6f} ± {metrics_summary['earth_mover_distance_std']:.6f} ↓")
    print(
        f"RMS Error: {metrics_summary['rms_error']:.6f} ± {metrics_summary['rms_error_std']:.6f} ↓")
    if USE_VOXEL_IOU:
        print(
            f"IOU Voxel: {metrics_summary['iou_voxel']:.6f} ± {metrics_summary['iou_voxel_std']:.6f} ↑")
    if USE_BREP_IOU and ext == "step":
        print(
            f"IOU Brep: {metrics_summary['iou_brep']:.6f} ± {metrics_summary['iou_brep_std']:.6f} ↑")
    print(
        f"Matching Rate: {metrics_summary['matching_rate']:.6f} ± {metrics_summary['matching_rate_std']:.6f} ↑")
    print(
        f"Coverage Rate: {metrics_summary['coverage_rate']:.6f} ± {metrics_summary['coverage_rate_std']:.6f} ↑")
    # print(
    #     f"JSD: {metrics_summary['jsd']:.6f} ± {metrics_summary['jsd_std']:.6f} ↓")
    print(
        f"Normal Consistency: {metrics_summary['normal_consistency']:.6f} ± {metrics_summary['normal_consistency_std']:.6f} ↑")
    print(
        f"\nAverage Computation Time: {metrics_summary['computation_time']:.2f} ± {metrics_summary['computation_time_std']:.2f} seconds")

    # Save results
    with open(summary_file, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nSummary results saved to {summary_file}")

    save_results_to_file(grouped_results, output_file)
    print(f"All pairs' statistics saved to {output_file}")


if __name__ == "__main__":
    main()

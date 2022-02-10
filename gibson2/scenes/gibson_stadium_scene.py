import pybullet_data
import pybullet as p
import os
from gibson2.scenes.scene_base import Scene
import logging
import numpy as np
from gibson2.utils.utils import l2_distance
from gibson2.utils.assets_utils import get_scene_path
from gibson2.objects.articulated_object import ArticulatedObject
import gibson2
from gibson2.scenes.indoor_scene import IndoorScene


class StaticStadiumScene(IndoorScene):
    """
    A empty scene for debugging
    """

    def __init__(
        self,
        scene_id,
        height_range,
        trav_map_resolution=0.1,
        trav_map_erosion=2,
        trav_map_type="with_obj",
        build_graph=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
        pybullet_load_texture=False,
    ):
        super(StaticStadiumScene, self).__init__(
            scene_id=scene_id,
            trav_map_resolution=trav_map_resolution,
            trav_map_erosion=trav_map_erosion,
            trav_map_type=trav_map_type,
            build_graph=build_graph,
            num_waypoints=num_waypoints,
            waypoint_resolution=waypoint_resolution,
            pybullet_load_texture=pybullet_load_texture,
        )
        self.scene_id = scene_id
        self.height_range = height_range
        logging.info("GibsonStadium Scene: {}".format(scene_id))

    def load(self):
        """
        Load the scene into pybullet
        """
        filename = os.path.join(
            pybullet_data.getDataPath(), "stadium_no_collision.sdf")
        self.stadium = p.loadSDF(filename)
        plane_file = os.path.join(
            pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.floor_body_ids += [p.loadMJCF(plane_file)[0]]
        pos, orn = p.getBasePositionAndOrientation(self.floor_body_ids[0])
        p.resetBasePositionAndOrientation(
            self.floor_body_ids[0], [pos[0], pos[1], pos[2] - 0.005], orn)
        p.changeVisualShape(
            self.floor_body_ids[0], -1, rgbaColor=[1, 1, 1, 0.5])
        
        self.load_trav_map(get_scene_path(self.scene_id))
        return list(self.stadium) + self.floor_body_ids

    def get_random_floor(self):
        return 0


    def get_random_point(self, floor=None):
        """
        """
        return floor, np.array([
            np.random.uniform(0, 5),
            np.random.uniform(0, 5),
            np.random.uniform(self.height_range[0], self.height_range[1]),
        ])

    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Get a trivial shortest path because the scene is empty
        """
        logging.warning(
            "WARNING: trying to compute the shortest path in EmptyScene (assuming empty space)"
        )
        shortest_path = np.stack((source_world, target_world))
        geodesic_distance = l2_distance(source_world, target_world)
        return shortest_path, geodesic_distance

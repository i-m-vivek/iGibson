import pybullet_data
import pybullet as p
import os
from gibson2.scenes.scene_base import Scene
import logging
import numpy as np
from gibson2.utils.utils import l2_distance
from gibson2.objects.articulated_object import ArticulatedObject


class EmptyTableScene(Scene):
    """
    A empty scene for debugging
    """

    def __init__(
        self,
        table_urdf_path,
        min_change=2,
        max_change=2,
        table_height=0.5,
        table_scale=2,
        table_pos=[2, 0, 0.3],
    ):
        super(EmptyTableScene, self).__init__()
        self.table_urdf_path = table_urdf_path
        self.table_scale = table_scale
        self.table_pos = table_pos
        self.min_change = min_change
        self.max_change = max_change
        self.table_height = table_height

    def load(self):
        """
        Load the scene into pybullet
        """
        plane_file = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.floor_body_ids += [p.loadMJCF(plane_file)[0]]
        p.changeDynamics(self.floor_body_ids[0], -1, lateralFriction=1)
        # white floor plane for visualization purpose if needed
        p.changeVisualShape(self.floor_body_ids[0], -1, rgbaColor=[1, 1, 1, 1])
        self.table = p.loadURDF(
            self.table_urdf_path,
            basePosition=self.table_pos,
            globalScaling=self.table_scale,
            useFixedBase=1,
        )
        return [self.table] + self.floor_body_ids

    def get_random_point(self, floor=None):
        """
        Get a random point in the region of table top
        The table top boundary will depend on the table itself.

        min_change: the minimum distance to go from the center of the table
        max_change: the maximum distance to go from the center of the table
        table_height: Hieght of the table
        """
        random_pt = np.array(
            [
                self.table_pos[0]
                + 0.3+ np.random.uniform(-self.min_change, self.max_change),
                self.table_pos[1]
                + 0.3+ np.random.uniform(-self.min_change, self.max_change),
                self.table_height
            ]
        )
        # print(random_pt)
        return floor, random_pt

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

from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task
from panda3d.core import *
import numpy as np


SIZE = 600


def turn_pose(pose):
    """
    First switches y and z components and the invert the z component to make the pose stand upright,
    where the head is in positive z direction.
    :param pose:
    :return:
    """
    if len(pose.shape) == 2:
        pose = pose[:, [0, 2, 1]] * np.array([1, 1, -1])
    elif len(pose.shape) == 3:
        pose = pose[:, :, [0, 2, 1]] * np.array([1, 1, -1])
    elif len(pose.shape) == 4:
        pose = pose[:, :, :, [0, 2, 1]] * np.array([1, 1, -1])
    else:
        raise Exception("Not implemented {}".format(len(pose.shape)))
    return pose


class PoseRenderer(ShowBase):

    def __init__(self, poses, skeleton=get_skeleton_15(), colored_skeleton=get_colored_skeleton_15(), colored=True,
                 max_display=8, image_paths=None):
        ShowBase.__init__(self)
        poses = np.asarray(poses)
        if len(poses.shape) == 2:
            poses = [poses]
        if len(poses.shape) == 3:
            poses = [poses]
        poses = poses[:max_display]
        self.poses = poses

        self.skeleton = skeleton
        self.colored_skeleton = colored_skeleton
        self.colored = colored
        self.image_paths = image_paths
        self.fps = 30
        self.rotation_speed = 10
        self.current_pose_index = 0
        self.vertex_data = None
        self.configure()

        self.render_poses()

    def configure(self):
        self.disableMouse()
        globalClock.setMode(ClockObject.MLimited)
        globalClock.setFrameRate(self.fps)
        self.create_windows()
        self.configure_tasks()

    def create_windows(self):
        for i in range(len(self.poses)):
            wp = WindowProperties()
            wp.setTitle('Pose {}'.format(i))
            wp.setSize(SIZE, SIZE)
            wp.setOrigin(50 + i * (SIZE + 20), 200 + i // 4 * (SIZE + 20))
            if i != 0:
                self.openWindow(props=wp)
            else:
                self.winList[i].requestProperties(wp)
            self.camList[i].setPos(6, 6, 4)
            self.camList[i].lookAt(0, 0, 0)
            mk = self.dataRoot.attachNewNode(MouseAndKeyboard(self.winList[i], 0, 'w2mouse'))
            mk.attachNewNode(ButtonThrower('w2mouse'))

    def configure_tasks(self):
        self.taskMgr.add(self.auto_rotate, 'autoRotate')
        self.accept('+', self.next_pose)
        self.accept('+-repeat', self.next_pose)
        self.accept('-', self.previous_pose)
        self.accept('--repeat', self.previous_pose)

    def next_pose(self):
        self.current_pose_index += 1
        self.render_poses()

    def previous_pose(self):
        self.current_pose_index -= 1
        self.render_poses()

    def auto_rotate(self, task):
        """
        Lets the camera rotate around the pose with speed self.rotation_speed
        :param task:
        :return:
        """
        time = task.time
        angle = time / self.fps * self.rotation_speed % 360
        for cam in self.camList:
            cam_position = np.asarray(cam.getPos())
            radius = np.linalg.norm(cam_position[:2])
            cam.setPos(np.cos(angle) * radius,
                       np.sin(angle) * radius,
                       cam_position[2])
            cam.lookAt(0, 0, 0)
        return Task.cont

    def render_poses(self):
        for i, poses in enumerate(self.poses):
            current_pose = poses[self.current_pose_index % len(poses)]
            root = NodePath('Pose {}'.format(i))
            if self.colored:
                pose_node = self.render_colored_pose(current_pose)
            else:
                pose_node = self.render_pose(current_pose)

            axes_node = self.create_axes(2)
            axes_node.reparentTo(root)
            pose_node.reparentTo(root)
            self.camList[i].reparentTo(root)

            if self.image_paths:
                print('Showing {}'.format(path.basename(self.image_paths[self.current_pose_index % len(poses)])))

    def render_colored_pose(self, pose):
        skeleton = self.colored_skeleton
        ls = LineSegs()
        ls.setThickness(2)

        for i in range(skeleton.shape[0]):
            for j in range(skeleton.shape[1]):
                if np.any(skeleton[i][j]):
                    ls.setColor(*skeleton[i][j])
                    ls.move_to(*pose[i])
                    ls.draw_to(*pose[j])

        node = ls.create()
        return NodePath(node)

    def render_pose(self, pose):
        ls = LineSegs()
        ls.setThickness(2)

        for i in range(self.skeleton.shape[0]):
            for j in range(self.skeleton.shape[1]):
                if self.skeleton[i][j]:
                    ls.move_to(*pose[i])
                    ls.draw_to(*pose[j])

        node = ls.create()
        return NodePath(node)

    def create_axes(self, length=1):
        ls = LineSegs()
        ls.setThickness(1)

        # X axis
        ls.setColor(1.0, 0.0, 0.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(1.0 * length, 0.0, 0.0)

        # Y axis
        ls.setColor(0.0, 1.0, 0.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(0.0, 1.0 * length, 0.0)

        # Z axis
        ls.setColor(0.0, 0.0, 1.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(0.0, 0.0, 1.0 * length)

        node = ls.create()
        return NodePath(node)


if __name__ == '__main__':
    from os import path
    import pickle as pkl
    import data_processing.human36m.process_custom as proc
    import projections
    import data_processing.human36m.preprocessing as pre

    data_dir = '/home/nikolas/Projects/2dto3dpose/data/'
    file_path = path.join('/home/nikolas/Projects/2dto3dpose/data/', 'processed', 'by_camera.pkl')
    if not path.isfile(file_path):
        p2, raw_poses, p3_univ, p3_orig = proc.get_all_poses_by_camera(data_dir, 'test')
        poses = []
        for cam in raw_poses:
            poses.append(read_data.convert_to_n_joints(np.asarray(raw_poses[cam], dtype=np.float32), 15))

        poses = np.asarray(poses)
        with open(file_path, 'wb') as file:
            pkl.dump(poses, file)
    else:
        with open(file_path, 'rb') as file:
            poses = pkl.load(file)

    for i in range(len(poses)):
        poses[i] = np.asarray(pre.rectify_all(poses[i]))
        poses[i] = poses[i][:1111] - poses[i][0, 0]
        poses[i] = projections.normalize_3d_pose(poses[i], scale_joints=[0, 7])

    # poses = read_data.get_original_3d_poses(data_dir, joints=15, phase='test')
    # poses = np.array(poses, dtype=np.float32)
    # poses = projections.normalize_3d_pose(poses, scale_joints=[0, 7])
    # poses -= poses[0][0]
    poses = turn_pose(poses)

    app = PoseRenderer(poses, skeleton=get_skeleton_15(),
                       colored_skeleton=get_colored_skeleton_15())
    app.run()

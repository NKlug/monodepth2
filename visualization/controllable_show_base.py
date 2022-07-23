from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task
from direct.showbase.Loader import Loader
from panda3d.core import *
import numpy as np

FORWARD = 'forward'
BACKWARD = 'backward'
LEFT = 'left'
RIGHT = 'right'
UP = 'up'
DOWN = 'down'

ON = 1
OFF = 0


class ControllableShowBase(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        self.camera_position = np.asarray([0, 0, 2])
        self.yaw = 135
        self.pitch = -10
        self.control_map = {FORWARD: 0, BACKWARD: 0, RIGHT: 0, LEFT: 0, UP: 0, DOWN: 0}

        self.root = NodePath('Root Node')

        self.global_grid_scale = None

        self.fps = 30

        self.rotation_speed = 10
        self.xy_move_speed = 0.1
        self.z_move_speed = 0.05
        self.cam_speed = 2

        self.configure()
        self.create_hints()

    def addInstructions(self, pos, msg):
        text = OnscreenText(text=msg, style=1, fg=(0, 0, 0, 1),
                            pos=(0.03, pos), align=TextNode.ALeft, scale=.04)
        text.reparentTo(self.a2dLeftCenter)
        return text

    def create_hints(self):
        self.addInstructions(0.95, "Q: Quit")
        self.addInstructions(0.90, "W A S D: forward, left, back, and right movement.")
        self.addInstructions(0.85, "X C: down and up movement.")

    def create_light(self):
        ambientLight = AmbientLight("ambientLight")
        # existing lighting is effectively darkened so boost ambient a bit
        ambientLight.setColor(Vec4(.4, .4, .4, 1))

        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection(Vec3(-5, -5, -5))

        directionalLight.setColor(Vec4(2.0, 2.0, 2.0, 1.0))
        directionalLight.setSpecularColor(Vec4(2.0, 2.0, 2.0, 1))
        self.root.setLight(self.root.attachNewNode(ambientLight))
        self.root.setLight(self.root.attachNewNode(directionalLight))

    def configure(self):
        self.disableMouse()
        globalClock.setMode(ClockObject.MLimited)
        globalClock.setFrameRate(self.fps)
        self.create_window()

        self.accept('q', self.finalizeExit)

        self.setFrameRateMeter(True)

        self.configure_camera_controls()
        # self.create_light()

    def create_window(self):
        wp = WindowProperties()
        wp.setTitle('3D depths')
        height = self.pipe.getDisplayHeight()
        width = self.pipe.getDisplayWidth()
        wp.setSize(width, height)
        wp.setOrigin(0, 0)
        self.disableMouse()
        wp.setCursorHidden(True)
        wp.setMouseMode(WindowProperties.MRelative)

        self.winList[0].requestProperties(wp)
        self.camera.setPos(*self.camera_position)
        self.camera.setHpr(self.yaw, self.pitch, 0)
        self.camLens.setNear(0.2)
        # mk = self.dataRoot.attachNewNode(MouseAndKeyboard(self.winList[0], 0, 'w2mouse'))
        # mk.attachNewNode(ButtonThrower('w2mouse'))

    def configure_camera_controls(self):
        self.accept('w', self.set_control, [FORWARD, ON])
        self.accept('w-up', self.set_control, [FORWARD, OFF])
        self.accept('s', self.set_control, [BACKWARD, ON])
        self.accept('s-up', self.set_control, [BACKWARD, OFF])
        self.accept('d', self.set_control, [RIGHT, ON])
        self.accept('d-up', self.set_control, [RIGHT, OFF])
        self.accept('a', self.set_control, [LEFT, ON])
        self.accept('a-up', self.set_control, [LEFT, OFF])
        self.accept('x', self.set_control, [DOWN, ON])
        self.accept('x-up', self.set_control, [DOWN, OFF])
        self.accept('c', self.set_control, [UP, ON])
        self.accept('c-up', self.set_control, [UP, OFF])

        self.taskMgr.add(self.move, 'moveTask')
        # self.taskMgr.add(self.move_camera, 'moveCameraTask')

    def set_control(self, direction, mode):
        self.control_map[direction] = mode

    def move(self, task):
        # first, move camera
        self.move_camera()

        # get look-at direction in x-y-plane
        yaw = self.yaw / 180 * np.pi
        look_at_direction = np.asarray([-np.sin(yaw), np.cos(yaw), 0])
        look_at_direction = look_at_direction / np.linalg.norm(look_at_direction)

        # get vector perpendicular to look-at direction, pointing towards the right
        right_direction = look_at_direction[[1, 0, 2]]
        right_direction[1] *= -1

        up_direction = np.asarray([0, 0, 1])

        if self.control_map[FORWARD] == ON:
            direction = look_at_direction
        elif self.control_map[BACKWARD] == ON:
            direction = -look_at_direction
        elif self.control_map[RIGHT] == ON:
            direction = right_direction
        elif self.control_map[LEFT] == ON:
            direction = -right_direction
        elif self.control_map[UP] == ON:
            direction = up_direction
        elif self.control_map[DOWN] == ON:
            direction = -up_direction
        else:
            return Task.cont

        speed = self.z_move_speed if self.control_map[UP] == ON or self.control_map[DOWN] == ON else self.xy_move_speed

        self.camera_position = self.camera_position + speed * direction
        self.camera.setPos(*self.camera_position)
        self.camera.setHpr(self.yaw, self.pitch, 0)

        return Task.cont

    def move_camera(self):
        if self.mouseWatcherNode.hasMouse():
            # get changes in mouse position
            md = self.win.getPointer(0)
            deltaX = md.getX()
            deltaY = md.getY()
            self.win.movePointer(0, 0, 0)

            yaw, pitch, roll = self.camera.getHpr()
            self.yaw = ((yaw + 180 + (-1) * self.cam_speed * 1e-2 * deltaX) % 360) - 180
            self.pitch = ((pitch + 180 + (-1) * self.cam_speed * 1e-2 * deltaY) % 360) - 180
            self.pitch = np.minimum(np.maximum(self.pitch, -90), 90)
            self.camera.setHpr(self.yaw, self.pitch, roll)

    def auto_rotate(self, task):
        """
        Lets the camera rotate around the pose with speed self.rotation_speed
        @param task:
        @return:
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

    def create_axes_and_grid(self, length=1):
        grid_scale = self.global_grid_scale
        if grid_scale is None:
            grid_scale = 1.0
        ls = LineSegs()
        ls.setThickness(3)

        # X axis
        ls.setColor(1.0, 0.0, 0.0, 1.0)
        ls.moveTo(-1.0 * length, 0.0, 0.0)
        ls.drawTo(1.0 * length, 0.0, 0.0)

        # Y axis
        ls.setColor(0.0, 1.0, 0.0, 1.0)
        ls.moveTo(0.0, -1.0 * length, 0.0)
        ls.drawTo(0.0, 1.0 * length, 0.0)

        # Z axis
        ls.setColor(0.0, 0.0, 1.0, 1.0)
        ls.moveTo(0.0, 0.0, -1.0 * length)
        ls.drawTo(0.0, 0.0, 1.0 * length)
        node = ls.create()
        node = NodePath(node)

        grid = LineSegs()
        grid.setThickness(1)

        # grid
        for i in range(-length, length):
            for j in range(-length, length):
                if i != 0 and j != 0:
                    grid.setColor(0.0, 0.0, 0.0, 0.7)
                    grid.moveTo(- i * 1.0, -j * 1.0, 0.0)
                    grid.drawTo(i * 1.0, -j * 1.0, 0.0)
                    grid.moveTo(- i * 1.0, -j * 1.0, 0.0)
                    grid.drawTo(-i * 1.0, j * 1.0, 0.0)

        grid = grid.create()
        grid = NodePath(grid)
        grid.reparentTo(node)

        return node

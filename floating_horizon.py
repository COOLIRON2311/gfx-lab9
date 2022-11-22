from dataclasses import dataclass, field
from math import cos, pi, radians, sin, sqrt, tan
from tkinter import simpledialog as sd
from typing import Callable

import numpy as np
import pygame as pg

# pylint: disable=no-member


@dataclass
class Point:
    x: float
    y: float
    z: float
    vis: bool = field(default=True, init=False)

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Point):
            return self.x == __o.x and self.y == __o.y and self.z == __o.z
        return False

    def __array__(self):
        return np.array([self.x, self.y, self.z])

    def _look_mat(self, camTarget, upVec) -> np.ndarray:
        camDir = np.array([Camera.position.x, Camera.position.y, Camera.position.z]) - camTarget
        camDir = camDir / np.linalg.norm(camDir)
        camRight = np.cross(upVec, camDir)
        camRight = camRight / np.linalg.norm(camRight)
        camUp = np.cross(camDir, camRight)
        mat1 = np.array([
            [camRight[0], camRight[1], camRight[2], 0],
            [camUp[0], camUp[1], camUp[2], 0],
            [camDir[0], camDir[1], camDir[2], 0],
            [0, 0, 0, 1]
        ])
        mat2 = np.array([
            [1, 0, 0, -Camera.position.x],
            [0, 1, 0, -Camera.position.y],
            [0, 0, 1, -Camera.position.z],
            [0, 0, 0, 1]
        ])
        lookMat = np.matmul(mat1, mat2)
        return lookMat

    def screen_coords(self) -> 'Point':
        lookMat = self._look_mat(np.array([Camera.position.x, Camera.position.y, Camera.position.z]) +
                                 Camera.camFront, np.array([0.0, 1.0, 0.0]))
        lookMat = lookMat.T
        angle = 45.0
        ratio = 1
        near = 0.001
        far = 1000.0
        tan_half = np.tan(np.radians(angle)/2)
        perMat = np.array([
            [1/(tan_half*ratio), 0, 0, 0],
            [0, 1/(tan_half), 0, 0],
            [0, 0, -(far+near)/(far-near), -(2*far*near)/(far-near)],
            [0, 0, -1, 0]])

        coor = np.array([self.x, self.y, self.z, 1])
        res = np.matmul(coor, lookMat)
        res = np.matmul(res, perMat)

        if res[3] == 0:
            x = res[0] + App.W / 2
            y = res[1] + App.H / 2
            z = res[2]
        else:
            x = res[0]/(res[3]) + App.W/2
            y = res[1]/(res[3]) + App.H/2
            z = res[2]/res[3]
        return Point(x, y, z)

    def draw(self, surf: pg.Surface, color: str = 'white', draw_points: bool = True):
        # coord = self.screen_coords()
        if self.vis and draw_points and 0 <= self.x < App.W and 0 <= self.y < App.H:
            # canvas.set_at((int(coord.x), int(coord.y)), pg.Color(color))
            pg.draw.circle(surf, pg.Color(color), (int(self.x), int(self.y)), 2)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def not_on_screen(self) -> bool:
        return not (0 <= self.x < App.W and 0 <= self.y < App.H)


@dataclass
class Line:
    p1: Point
    p2: Point

    def draw(self, canvas: pg.Surface, color: str = 'white', draw_points: bool = False):
        # p1.draw(canvas, color, draw_points)
        # p2.draw(canvas, color, draw_points)
        # self.__wu(canvas, self.p1, self.p2, pg.Color(color))
        pg.draw.line(canvas, pg.Color(color), (self.p1.x, self.p1.y), (self.p2.x, self.p2.y))
        return self.p1, self.p2

class Camera:
    is_moving = False

    class moving:
        Forward = False
        Backward = False
        Left = False
        Right = False
        RotLeft = False
        RotRight = False
        RotUp = False
        RotDown = False
    position = Point(0, 0, 1000)
    horRot = -90.0  # yaw
    verRot = 0.0  # pitch
    camFront = np.array([0.0, 0.0, -1.0])
    camSpeed = 10
    camRotSpeed = 1

    @staticmethod
    def dist_to(x, y, z) -> float:
        return sqrt((x - Camera.position.x) ** 2 + (y - Camera.position.y) ** 2 + (z - Camera.position.z) ** 2)

    @staticmethod
    def move_forward():
        Camera.position.x += Camera.camFront[0] * Camera.camSpeed
        Camera.position.y += Camera.camFront[1] * Camera.camSpeed
        Camera.position.z += Camera.camFront[2] * Camera.camSpeed

    @staticmethod
    def move_backward():
        Camera.position.x -= Camera.camFront[0] * Camera.camSpeed
        Camera.position.y -= Camera.camFront[1] * Camera.camSpeed
        Camera.position.z -= Camera.camFront[2] * Camera.camSpeed

    @staticmethod
    def move_left():
        r = np.cross(Camera.camFront, [0.0, 1.0, 0.0])
        r = r / np.linalg.norm(r)
        Camera.position.x -= r[0] * Camera.camSpeed
        Camera.position.y -= r[1] * Camera.camSpeed
        Camera.position.z -= r[2] * Camera.camSpeed

    @staticmethod
    def move_right():
        r = np.cross(Camera.camFront, [0.0, 1.0, 0.0])
        r = r / np.linalg.norm(r)
        Camera.position.x += r[0] * Camera.camSpeed
        Camera.position.y += r[1] * Camera.camSpeed
        Camera.position.z += r[2] * Camera.camSpeed

    @staticmethod
    def rotateLeft():
        Camera.horRot -= Camera.camRotSpeed
        Camera.rotate()

    @staticmethod
    def rotateRight():
        Camera.horRot += Camera.camRotSpeed
        Camera.rotate()

    @staticmethod
    def rotateUp():
        Camera.verRot -= Camera.camRotSpeed
        if Camera.verRot > 89.0:
            Camera.verRot = 89.0
        Camera.rotate()

    @staticmethod
    def rotateDown():
        Camera.verRot += Camera.camRotSpeed
        if Camera.verRot < -89.0:
            Camera.verRot = -89.0
        Camera.rotate()

    @staticmethod
    def rotate():
        f = np.array([0.0, 0.0, 0.0])
        f[0] = np.cos(np.radians(Camera.horRot))*np.cos(np.radians(Camera.verRot))
        f[1] = np.sin(np.radians(Camera.verRot))
        f[2] = np.sin(np.radians(Camera.horRot))*np.cos(np.radians(Camera.verRot))
        n = np.linalg.norm(f)
        Camera.camFront = f/n


class App:
    SCALE = 50
    W = 800
    H = 600
    uph: np.ndarray
    downh: np.ndarray
    surf: pg.Surface
    points: list[Point]
    n: int
    func: Callable[[float, float], float]

    def __init__(self):
        n = sd.askinteger("Number of points", "Enter number of points", minvalue=3, initialvalue=20)
        if n is None:
            return
        func = sd.askstring("Function", "Enter function", initialvalue="sin(x) + cos(y)")
        if func is None:
            return
        func = eval(f"lambda x, y: {func}")
        self.func = func
        self.n = n

        self.surf = pg.display.set_mode((App.W, App.H))
        pg.display.set_caption("Floating horizon")
        pg.display.init()
        self.reset()
        self.draw()
        pg.display.flip()

    def process_point(self, p: Point) -> None:
        if p.not_on_screen():
            p.vis = False
            return

        if p.y >= self.uph[int(p.x) - 1]:
            p.vis = True
            return

        if p.y <= self.downh[int(p.x) - 1]:
            p.vis = True
            return

        p.vis = False

    def update_horizon(self, p: Point, prev: Point) -> None:
        if p.not_on_screen():
            return
        if prev.x < 0 or p.x >= App.W:
            return
        elif int(prev.x) == int(p.x):
            self.uph[int(p.x)] = max(self.uph[int(p.x)], p.y)
            self.downh[int(p.x)] = min(self.downh[int(p.x)], p.y)
        else:
            gradient = (p.y - prev.y) / (p.x - prev.x)
            for i in range(int(prev.x), int(p.x) + 1):
                y = prev.y + gradient * (i - prev.x)
                self.uph[i] = max(self.uph[i], y)
                self.downh[i] = min(self.downh[i], y)

    def intersect(self, p1: Point, p2: Point, reverse: bool = False) -> Point:
        d = 20
        xstep = (p2.x - p1.x) / d
        ystep = (p2.y - p1.y) / d

        for i in range(d):
            p = Point(p1.x + xstep * i, p1.y + ystep * i, 0)
            self.process_point(p)
            if reverse and p.vis:
                return p
            elif not reverse and not p.vis:
                return p
        return p2

    def calc_points(self):
        n = self.n
        func = self.func
        self.points = []
        for i in range(n, 0, -1):
            z = i * self.SCALE
            prev = Point(0, func(0, z), z).screen_coords()
            self.process_point(prev)

            for j in range(n):
                x = j * self.SCALE
                curr = Point(x, self.SCALE*func(x, z), z).screen_coords()
                self.process_point(curr)
                if curr.vis:
                    if prev.vis:
                        Line(prev, curr).draw(self.surf)
                        self.update_horizon(curr, prev)
                    else:
                        r = self.intersect(prev, curr, reverse=False)
                        Line(r, curr).draw(self.surf)
                        self.update_horizon(curr, r)
                else:
                    if prev.vis:
                        r = self.intersect(prev, curr, reverse=True)
                        Line(prev, r).draw(self.surf)
                        self.update_horizon(r, prev)
                prev = curr

    def reset(self):
        self.uph = np.full((App.W), -np.inf)
        self.downh = np.full((App.W), np.inf)
        self.surf.fill('#393939')

    def draw(self):
        self.reset()
        self.calc_points()
        # ln = 100
        # Line(Point(0, 0, 0), Point(ln, 0, 0)).draw(self.surf, color='red')  # x axis
        # Line(Point(0, 0, 0), Point(0, ln, 0)).draw(self.surf, color='green')  # y axis
        # Line(Point(0, 0, 0), Point(0, 0, ln)).draw(self.surf, color='blue')  # z axis
        pg.display.update()

    def run(self):
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    pg.quit()
                    return

                elif e.type == pg.KEYDOWN:
                    if e.key == pg.K_w:
                        Camera.is_moving = True
                        Camera.moving.Forward = True
                    elif e.key == pg.K_s:
                        Camera.is_moving = True
                        Camera.moving.Backward = True
                    elif e.key == pg.K_a:
                        Camera.is_moving = True
                        Camera.moving.Left = True
                    elif e.key == pg.K_d:
                        Camera.is_moving = True
                        Camera.moving.Right = True
                    elif e.key == pg.K_LEFT:
                        Camera.is_moving = True
                        Camera.moving.RotLeft = True
                    elif e.key == pg.K_RIGHT:
                        Camera.is_moving = True
                        Camera.moving.RotRight = True
                    elif e.key == pg.K_UP:
                        Camera.is_moving = True
                        Camera.moving.RotUp = True
                    elif e.key == pg.K_DOWN:
                        Camera.is_moving = True
                        Camera.moving.RotDown = True

                elif e.type == pg.KEYUP:
                    if e.key == pg.K_w:
                        Camera.is_moving = False
                        Camera.moving.Forward = False
                    if e.key == pg.K_s:
                        Camera.is_moving = False
                        Camera.moving.Backward = False
                    if e.key == pg.K_a:
                        Camera.is_moving = False
                        Camera.moving.Left = False
                    if e.key == pg.K_d:
                        Camera.is_moving = False
                        Camera.moving.Right = False
                    elif e.key == pg.K_LEFT:
                        Camera.is_moving = False
                        Camera.moving.RotLeft = False
                    elif e.key == pg.K_RIGHT:
                        Camera.is_moving = False
                        Camera.moving.RotRight = False
                    elif e.key == pg.K_UP:
                        Camera.is_moving = False
                        Camera.moving.RotUp = False
                    elif e.key == pg.K_DOWN:
                        Camera.is_moving = False
                        Camera.moving.RotDown = False

            if Camera.is_moving:
                if Camera.moving.Forward:
                    Camera.move_forward()
                if Camera.moving.Backward:
                    Camera.move_backward()
                if Camera.moving.Left:
                    Camera.move_left()
                if Camera.moving.Right:
                    Camera.move_right()
                if Camera.moving.RotLeft:
                    Camera.rotateLeft()
                if Camera.moving.RotRight:
                    Camera.rotateRight()
                if Camera.moving.RotUp:
                    Camera.rotateUp()
                if Camera.moving.RotDown:
                    Camera.rotateDown()
                self.reset()
                self.draw()
                pg.display.update()


if __name__ == "__main__":
    App().run()

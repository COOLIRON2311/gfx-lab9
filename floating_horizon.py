from dataclasses import dataclass
from math import cos, pi, radians, sin, sqrt, tan
from tkinter import simpledialog as sd

import numpy as np
import pygame as pg

# pylint: disable=no-member

@dataclass
class Point:
    x: float
    y: float
    z: float

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

    def draw(self, surf: pg.Surface, color: str = 'white', draw_points: bool = True, dry_run=False):
        lookMat = self._look_mat(np.array([Camera.position.x, Camera.position.y, Camera.position.z]) +
                                 Camera.camFront, np.array([0.0, 1.0, 0.0]))
        lookMat = lookMat.T
        angle = 45.0
        ratio = 1
        near = 0.001
        far = 1000.0
        tan_half = tan(radians(angle)/2)
        perMat = np.array([
            [1/(tan_half*ratio), 0, 0, 0],
            [0, 1/(tan_half), 0, 0],
            [0, 0, -(far+near)/(far-near), -(2*far*near)/(far-near)],
            [0, 0, -1, 0]])

        coor = np.array([self.x, self.y, self.z, 1])
        res = np.matmul(coor, lookMat)
        res = np.matmul(res, perMat)

        x = res[0]/(res[3]) + App.W/2
        y = res[1]/(res[3]) + App.H/2
        z = res[2]/res[3]
        if draw_points and not dry_run:
            surf.set_at((int(x), int(y)), pg.Color(color))
        return x, y, z

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def transform(self, matrix: np.ndarray):
        p = np.array([self.x, self.y, self.z, 1])
        p = np.dot(matrix, p)
        self.x = p[0]
        self.y = p[1]
        self.z = p[2]


@dataclass
class Line:
    p1: Point
    p2: Point

    def draw(self, canvas: pg.Surface, color: str = 'white', draw_points: bool = False):
        p1X, p1Y, p1Z = self.p1.draw(canvas, color, draw_points)
        p2X, p2Y, p2Z = self.p2.draw(canvas, color, draw_points=draw_points)
        self.__wu(canvas, Point(p1X, p1Y, p1Z), Point(p2X, p2Y, p2Z), pg.Color(color))
        return Point(p1X, p1Y, p1Z), Point(p2X, p2Y, p2Z)

    def transform(self, matrix: np.ndarray):
        self.p1.transform(matrix)
        self.p2.transform(matrix)


    def __wu(self, canvas: pg.Surface, a: Point, b: Point, color: pg.Color) -> None:
        if a.x > b.x:
            a, b = b, a

        dx = b.x - a.x
        dy = b.y - a.y

        if dx == 0:
            if a.y > b.y:
                a, b = b, a
            for y in range(int(a.y), int(b.y)):
                canvas.set_at((int(a.x), y), color)
            return

        gradient = dy/dx

        y = a.y+gradient

        if abs(gradient) < 1:
            if a.x > b.x:
                a, b = b, a

            for i in range(int(a.x), int(b.x)):
                canvas.set_at((i, int(y)), color)
                canvas.set_at((i, int(y+1)), color)
                y += gradient
        else:
            if a.y > b.y:
                a, b = b, a
            gradient2 = dx/dy
            x = a.x + gradient2
            for i in range(int(a.y), int(b.y)):
                canvas.set_at((int(x), i), color)
                canvas.set_at((int(x+1), i), color)
                x += gradient2


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
    W = 800
    H = 600
    surf: pg.Surface
    points: list[Point]

    def __init__(self):
        n = sd.askinteger("Number of points", "Enter number of points", minvalue=3)
        if n is None:
            return
        func = sd.askstring("Function", "Enter function", initialvalue="sin(x)")
        if func is None:
            return
        func = eval(f"lambda x, z: {func}")
        self.init_points(n, func)

        self.surf = pg.display.set_mode((App.W, App.H))
        self.surf.fill('#393939')
        pg.display.set_caption("Floating horizon")
        pg.display.init()

    def init_points(self, n: int, func):
        self.points = []
        for i in range(n):
            for j in range(n):
                ... # TODO: add points
                # x = i*100 - (n-1)*50
                # z = j*100 - (n-1)*50
                # self.points.append(Point(x, func(x, z), z))

    def reset(self):
        self.surf.fill('#393939')

    def draw(self):
        self.reset()
        for point in self.points:
            point.draw(self.surf)
        pg.display.update()

    def run(self):
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT:
                    pg.quit()

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

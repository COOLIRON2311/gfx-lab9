import tkinter as tk
from dataclasses import dataclass, field
from enum import Enum
from math import cos, pi, radians, sin, sqrt
from threading import Thread
from tkinter import filedialog as fd
from tkinter import messagebox as mb
from tkinter import simpledialog as sd
from typing import Callable

import numpy as np
import pygame as pg

# pylint: disable=no-member
# pylint: disable=eval-used


class Projection(Enum):
    Perspective = 0
    Axonometric = 1
    FreeCamera = 2

    def __str__(self) -> str:
        match self:
            case Projection.Perspective:
                return "Перспективная"
            case Projection.Axonometric:
                return "Аксонометрическая"
            case Projection.FreeCamera:
                return "Свободная камера"
        return "Неизвестная проекция"


class Mode(Enum):
    Translate = 0  # перемещение
    Rotate = 1  # вращение
    Scale = 2  # масштабирование

    def __str__(self) -> str:
        return super().__str__().split(".")[-1]


class Function(Enum):
    None_ = 0
    ReflectOverPlane = 1
    ScaleAboutCenter = 2
    RotateAroundAxis = 3
    RotateAroundLine = 4

    def __str__(self) -> str:
        match self:
            case Function.None_:
                return "Не выбрано"
            case Function.ReflectOverPlane:
                return "Отражение относительно плоскости"
            case Function.ScaleAboutCenter:
                return "Масштабирование относ. центра"
            case Function.RotateAroundAxis:
                return "Вращение относительно оси"
            case Function.RotateAroundLine:
                return "Вращение вокруг прямой"
            case _:
                pass
        return "Неизвестная функция"


class ShapeType(Enum):
    Tetrahedron = 0
    Hexahedron = 1
    Octahedron = 2
    Icosahedron = 3
    Dodecahedron = 4
    RotationBody = 5
    FuncPlot = 6

    def __str__(self) -> str:
        match self:
            case ShapeType.Tetrahedron:
                return "Тетраэдр"
            case ShapeType.Hexahedron:
                return "Гексаэдр"
            case ShapeType.Octahedron:
                return "Октаэдр"
            case ShapeType.Icosahedron:
                return "Икосаэдр"
            case ShapeType.Dodecahedron:
                return "Додекаэдр"
            case ShapeType.RotationBody:
                return "Тело вращения"
            case ShapeType.FuncPlot:
                return "График функции"
            case _:
                pass
        return "Неизвестная фигура"


class Shape:
    """Base class for all shapes"""

    def draw(self, canvas: pg.Surface, projection: Projection, color: str = 'white', draw_points: bool = True) -> None:
        pass

    def transform(self, matrix: np.ndarray) -> None:
        pass

    def fix_points(self):
        pass

    @staticmethod
    def load(path: str) -> 'Shape':
        with open(path, "r", encoding='utf8') as file:
            s = eval(file.read())
            if isinstance(s, (Polyhedron, FuncPlot)):
                s.fix_points()
            return s

    def save(self, path: str):
        if not path.endswith(".shape"):
            path += ".shape"
        with open(path, "w", encoding='utf8') as file:
            file.write(str(self))

    @property
    def center(self):
        pass


@dataclass
class Point(Shape):
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

    def createLookMat(self, camTarget, upVec):
        # print("LookMat")
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
        #mat1 = mat1.T
        mat2 = np.array([
            [1, 0, 0, -Camera.position.x],
            [0, 1, 0, -Camera.position.y],
            [0, 0, 1, -Camera.position.z],
            [0, 0, 0, 1]
        ])
        #mat2 = mat2.T
        lookMat = np.matmul(mat1, mat2)
        return lookMat

    def draw(self, canvas: pg.Surface, projection: Projection, color: str = 'white', draw_points: bool = True, dry_run=False):
        if projection == Projection.Perspective:
            # print(App.dist)
            per = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1 / App.dist],
                [0, 0, 0, 1]])
            coor = np.array([self.x, self.y, self.z, 1])
            res = np.matmul(coor, per)
            x = res[0]/res[3] + 450
            y = res[1]/res[3] + 250
            z = res[2]/res[3]

        elif projection == Projection.Axonometric:
            #print(App.phi, App.theta)
            phi = App.phi*(pi/180)
            theta = App.theta*(pi/180)
            iso = np.array([
                [cos(phi), cos(theta)*sin(phi), 0, 0],
                [0, cos(theta), 0, 0],
                [sin(phi), -sin(theta)*cos(phi), 0, 0],
                [0, 0, 0, 1]])
            coor = np.array([self.x, self.y, self.z, 1])
            res = np.matmul(coor, iso)
            x = res[0] + 600
            y = res[1] + 250
            z = res[2]

        elif projection == Projection.FreeCamera:
            # print("camera")
            # print(Camera.position)
            lookMat = self.createLookMat(np.array([Camera.position.x, Camera.position.y, Camera.position.z]) +
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

            x = res[0]/(res[3]) + App.W/2
            y = res[1]/(res[3]) + App.H/2
            z = res[2]/res[3]
        else:
            x = self.x
            y = self.y
            z = self.z
        if draw_points and x < 1000 and y < 1000 and not dry_run:
            canvas.set_at((int(x), int(y)), pg.Color(color))
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

    def copy(self):
        return Point(self.x, self.y, self.z)

    @property
    def center(self) -> 'Point':
        return Point(self.x, self.y, self.z)

    def normalized(self) -> 'Point':
        norm = np.linalg.norm([self.x, self.y, self.z])
        if norm == 0:
            return self
        return Point(self.x/norm, self.y/norm, self.z/norm)


@dataclass
class Line(Shape):
    p1: Point
    p2: Point

    def draw(self, canvas: pg.Surface, projection: Projection, color: str = 'white', draw_points: bool = False):
        p1X, p1Y, p1Z = self.p1.draw(canvas, projection, color, draw_points)
        p2X, p2Y, p2Z = self.p2.draw(canvas, projection, color, draw_points=draw_points)
        self.__wu(canvas, Point(p1X, p1Y, p1Z), Point(p2X, p2Y, p2Z), pg.Color(color))
        return Point(p1X, p1Y, p1Z), Point(p2X, p2Y, p2Z)
        # pg.draw.line(canvas, pg.Color(color), (p1X, p1Y), (p2X, p2Y))
        # canvas.create_line(p1X, p1Y, p2X, p2Y, fill=color)

    def transform(self, matrix: np.ndarray):
        self.p1.transform(matrix)
        self.p2.transform(matrix)

    def get_x(self, y):
        if self.p1.y == self.p2.y:
            return self.p1.x
        return self.p1.x + (self.p2.x - self.p1.x) * (y - self.p1.y) / (self.p2.y - self.p1.y)

    def get_y(self, x):
        if self.p1.x == self.p2.x:
            return self.p1.y
        return self.p1.y + (self.p2.y - self.p1.y) * (x - self.p1.x) / (self.p2.x - self.p1.x)

    def get_z(self, y):
        if self.p1.y == self.p2.y:
            return self.p1.z
        return self.p1.z + (self.p2.z - self.p1.z) * (y - self.p1.y) / (self.p2.y-self.p1.y)

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
                # ZBuffer.draw_point(canvas, int(a.x), y, a.z, color)
            return

        gradient = dy/dx

        y = a.y+gradient

        if abs(gradient) < 1:
            if a.x > b.x:
                a, b = b, a

            for i in range(int(a.x), int(b.x)):
                canvas.set_at((i, int(y)), color)
                canvas.set_at((i, int(y+1)), color)  # single color lines
                # ZBuffer.draw_point(canvas, i, int(y), a.z, color)
                # ZBuffer.draw_point(canvas, i, int(y+1), a.z, color)
                y += gradient
        else:
            if a.y > b.y:
                a, b = b, a
            gradient2 = dx/dy
            x = a.x + gradient2
            for i in range(int(a.y), int(b.y)):
                canvas.set_at((int(x), i), color)
                canvas.set_at((int(x+1), i), color)  # single color lines
                # ZBuffer.draw_point(canvas, int(x), i, a.z, color)
                # ZBuffer.draw_point(canvas, int(x+1), i, a.z, color)
                x += gradient2

    @property
    def center(self) -> 'Point':
        return Point((self.p1.x + self.p2.x) / 2, (self.p1.y + self.p2.y) / 2,
                     (self.p1.z + self.p2.z) / 2)


@dataclass
class Polygon(Shape):
    points: list[Point]
    normal: Point = field(init=False)

    def __post_init__(self):
        self.normal = self.calculate_normal()

    def draw(self, canvas: pg.Surface, projection: Projection, color: str = 'white', draw_points: bool = False):
        ln = len(self.points)
        lines = [Line(self.points[i], self.points[(i + 1) % ln])
                 for i in range(ln)]
        for line in lines:
            line.draw(canvas, projection, color, draw_points)
        # self.normal.draw(canvas, projection, 'red', draw_points=True)

    def interpolate(self, x1, z1, x2, z2):
        res = []

        d = abs(int(x2)-int(x1))
        if d < 0.001:
            res.append(z1)
            return res

        step = (z2-z1)/(x2-x1)

        for x in np.arange(x1, x2):
            res.append(z1)
            z1 += step

        return res

    def fill(self, canvas: pg.Surface, color: pg.Color):
        ln = len(self.points)
        tlines = [Line(self.points[i], self.points[(i + 1) % ln])
                  for i in range(ln)]
        lines: list[Line] = []
        points: set[Point] = set()
        for l in tlines:
            p1, p2 = l.draw(canvas, Projection.FreeCamera)
            lines.append(Line(p1, p2))
            points.add(p1)
            points.add(p2)
        ymax = max(p.y for p in points)
        ymin = min(p.y for p in points)

        far = max(p.z for p in points)
        near = min(p.z for p in points)

        for y in range(int(ymin), int(ymax)):
            intersections: list[Point] = []
            for line in lines:
                if line.p1.y <= y < line.p2.y or line.p2.y <= y < line.p1.y:
                    t = self.interpolate(line.p1.x, line.p1.z, line.p2.x, line.p2.z)
                    intersections.append(Point(line.get_x(y), y, line.get_z(y)))
            intersections.sort(key=lambda p: p.x)
            for i in range(0, len(intersections), 2):
                z = self.interpolate(intersections[i].x, intersections[i].z, intersections[i+1].x, intersections[i+1].z)
                for x in range(int(intersections[i].x), int(intersections[i+1].x)):
                    # z = self.points[0].z  # TODO: calculate z
                    #z = far/(far - near) + 1/intersections[i].z * ((-2*far*near)/(far-near))
                    cz = z[x-int(intersections[i].x)]
                    ZBuffer.draw_point(canvas, x, y, cz, color)

    def transform(self, matrix: np.ndarray):
        for point in self.points:
            point.transform(matrix)
        self.normal.transform(matrix)

    def copy(self):
        return Polygon([p.copy() for p in self.points])

    @property
    def center(self) -> 'Point':
        return Point(sum(point.x for point in self.points) / len(self.points),
                     sum(point.y for point in self.points) / len(self.points),
                     sum(point.z for point in self.points) / len(self.points))

    def calculate_normal(self) -> Point:
        # normal = Point(0, 0, 0)
        # ln = len(self.points)
        # for i in range(ln):
        #     currentv = self.points[i]
        #     nextv = self.points[(i + 1) % ln]
        #     normal.x += (currentv.y - nextv.y) * (currentv.z + nextv.z)
        #     normal.y += (currentv.z - nextv.z) * (currentv.x + nextv.x)
        #     normal.z += (currentv.x - nextv.x) * (currentv.y + nextv.y)
        # return normal.normalized()
        p1 = np.array(self.points[0])
        p2 = np.array(self.points[1])
        p3 = np.array(self.points[2])
        v1 = p1 - p2
        v2 = p3 - p2
        normal = np.cross(v2, v1)
        return Point(normal[0], normal[1], normal[2])


@dataclass
class Polyhedron(Shape):
    polygons: list[Polygon]

    def draw(self, canvas: pg.Surface, projection: Projection, color: str = 'white', draw_points: bool = False):
        bfc: bool = App.bfc.get()
        p = Camera.camFront + np.array(Camera.position)
        for poly in self.polygons:
            if bfc:
                v0 = np.array(poly.points[0])
                n = np.array(poly.normal)  # /np.linalg.norm(np.array(poly.normal))d
                if np.dot(v0 - p, n) < 0:
                    poly.draw(canvas, projection, color, draw_points)
                # else:
                #     poly.draw(canvas, projection, 'red', draw_points)
            else:
                poly.draw(canvas, projection, color, draw_points)

    def transform(self, matrix: np.ndarray):
        points = {point for poly in self.polygons for point in poly.points}
        for point in points:
            point.transform(matrix)

    def fix_points(self):
        points: dict[tuple[float, float, float], Point] = {}
        for poly in self.polygons:
            for i, point in enumerate(poly.points):
                k = (point.x, point.y, point.z)
                if k not in points:
                    points[k] = point
                else:
                    poly.points[i] = points[k]

    @property
    def center(self) -> 'Point':
        return Point(sum(polygon.center.x for polygon in self.polygons) /
                     len(self.polygons),
                     sum(polygon.center.y for polygon in self.polygons) /
                     len(self.polygons),
                     sum(polygon.center.z for polygon in self.polygons) /
                     len(self.polygons))

    def fill(self, canvas: pg.Surface, color: pg.Color):
        count = 0
        colors = [pg.Color("red"), pg.Color("green"), pg.Color("blue"), pg.Color('yellow')]
        for poly in self.polygons:
            poly.fill(canvas, colors[count % 4])
            count += 1


@dataclass
class RotationBody(Shape):
    polygon: Polygon
    axis: str
    partitions: int
    _mesh: Polyhedron = field(init=False, default=None)

    def draw(self, canvas: pg.Surface, projection: Projection, color: str = 'white', draw_points: bool = False):
        if self._mesh:
            self._mesh.draw(canvas, projection, color, draw_points)
            return
        angle = radians(360 / self.partitions)
        poly = self.polygon.copy()
        surface = []
        # Cheese:
        # 0, 0, 0, 0, 100, 0, 100, 100, 0, 100, 0, 0, Y, 10
        # Vase:
        # 0,0,0, 100,0,0, 100,50,0, 50,50,0, 150,250,0, 100,250,0, 100,300,0, 150,300,0, 150,350,0, 0,350,0, Y, 10
        for _ in range(self.partitions):
            surface.append(poly.copy())
            self.rotate(poly, angle)

        mesh = []
        # pylint: disable=consider-using-enumerate
        for i in range(self.partitions):
            poly1: Polygon = surface[i]
            poly2: Polygon = surface[(i + 1) % self.partitions]
            for j in range(len(poly1.points)):
                mesh.append(Polygon([poly1.points[j], poly1.points[(j + 1) % len(poly1.points)],
                                     poly2.points[(j + 1) % len(poly2.points)], poly2.points[j]]))
        self._mesh = Polyhedron(mesh)
        self._mesh.fix_points()
        self._mesh.draw(canvas, projection, color, draw_points)

    def rotate(self, poly: Polygon, phi: float):
        match self.axis:
            case 'X':
                mat = np.array([
                    [1, 0, 0, 0],
                    [0, cos(phi), -sin(phi), 0],
                    [0, sin(phi), cos(phi), 0],
                    [0, 0, 0, 1]])
            case 'Z':
                mat = np.array([
                    [cos(phi), -sin(phi), 0, 0],
                    [sin(phi), cos(phi), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
            case 'Y':
                mat = np.array([
                    [cos(phi), 0, sin(phi), 0],
                    [0, 1, 0, 0],
                    [-sin(phi), 0, cos(phi), 0],
                    [0, 0, 0, 1]])
            case _:
                raise ValueError("Invalid axis")
        poly.transform(mat)

    def transform(self, matrix: np.ndarray):
        self._mesh.transform(matrix)

    @property
    def center(self) -> 'Point':
        return self.polygon.center

    def save(self, path: str):
        self._mesh.save(path)


@dataclass
class FuncPlot(Shape):
    func: Callable[[float, float], float]
    x0: float
    x1: float
    y0: float
    y1: float
    nx: int
    ny: int
    _polyhedron: Polyhedron = field(init=False, default=None, repr=False)

    def __init__(self, func: str, x0: float, x1: float, y0: float, y1: float, nx: int, ny: int):
        self.func = eval(f"lambda x, y: {func}")
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.nx = nx
        self.ny = ny
        self._polyhedron = self._build_polyhedron()
        self.fix_points()

    def draw(self, canvas: pg.Surface, projection: Projection, color: str = 'white', draw_points: bool = False):
        self._polyhedron.draw(canvas, projection, color, draw_points)

    def save(self, path: str):
        self._polyhedron.save(path)

    def transform(self, matrix: np.ndarray) -> None:
        self._polyhedron.transform(matrix)

    def fix_points(self):
        points: dict[tuple[float, float, float], Point] = {}
        for poly in self._polyhedron.polygons:
            for i, point in enumerate(poly.points):
                k = (point.x, point.y, point.z)
                if k not in points:
                    points[k] = point
                else:
                    poly.points[i] = points[k]

    def _build_polyhedron(self) -> Polyhedron:
        polygons = []
        dx = (self.x1 - self.x0) / self.nx
        dy = (self.y1 - self.y0) / self.ny
        for i in range(self.nx):
            for j in range(self.ny):
                x0 = self.x0 + i * dx
                y0 = self.y0 + j * dy
                x1 = x0 + dx
                y1 = y0 + dy
                #z0 = self.func(x0, y0)
                #z1 = self.func(x1, y1)
                polygons.append(Polygon([
                    Point(x0, y0, self.func(x0, y0)),
                    Point(x1, y0, self.func(x1, y0)),
                    Point(x1, y1, self.func(x1, y1)),
                    Point(x0, y1, self.func(x0, y1))
                ]))
        return Polyhedron(polygons)

    @property
    def center(self) -> Point:
        return self._polyhedron.center


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


class Models:
    """
    Tetrahedron = 0
    Hexahedron = 1
    Octahedron = 2
    Icosahedron = 3
    Dodecahedron = 4
    """
    class Tetrahedron(Polyhedron):
        def __init__(self, size=100):
            t = Models.Hexahedron(size)
            p1 = t.polygons[0].points[1]
            p2 = t.polygons[0].points[3]
            p3 = t.polygons[2].points[2]
            p4 = t.polygons[1].points[3]
            polygons = [
                Polygon([p1, p2, p3]),
                Polygon([p1, p2, p4]),
                Polygon([p1, p3, p4]),
                Polygon([p2, p3, p4])
            ]
            super().__init__(polygons)

        def __repr__(self) -> str:
            return super().__repr__().replace('Models.Tetrahedron', 'Polyhedron')

    class Hexahedron(Polyhedron):
        def __init__(self, size=100):
            p1 = Point(0, 0, 0)
            p2 = Point(size, 0, 0)
            p3 = Point(size, size, 0)
            p4 = Point(0, size, 0)
            p5 = Point(0, 0, size)
            p6 = Point(size, 0, size)
            p7 = Point(size, size, size)
            p8 = Point(0, size, size)
            polygons = [
                Polygon([p4, p3, p2, p1]),
                Polygon([p1, p2, p6, p5]),
                Polygon([p3, p7, p6, p2]),
                Polygon([p3, p4, p8, p7]),
                Polygon([p4, p1, p5, p8]),
                Polygon([p5, p6, p7, p8])
            ]
            super().__init__(polygons)

        def __repr__(self) -> str:
            return super().__repr__().replace('Models.Hexahedron', 'Polyhedron')

    class Octahedron(Polyhedron):
        def __init__(self, size=100):
            t = Models.Hexahedron(size)
            p1 = t.polygons[0].center
            p2 = t.polygons[1].center
            p3 = t.polygons[2].center
            p4 = t.polygons[3].center
            p5 = t.polygons[4].center
            p6 = t.polygons[5].center
            polygons = [
                Polygon([p1, p2, p3]),
                Polygon([p1, p3, p4]),
                Polygon([p1, p5, p4]),
                Polygon([p1, p2, p5]),
                Polygon([p2, p3, p6]),
                Polygon([p5, p4, p6]),
                Polygon([p3, p4, p6]),
                Polygon([p2, p5, p6])
            ]
            super().__init__(polygons)

        def __repr__(self) -> str:
            return super().__repr__().replace('Models.Octahedron', 'Polyhedron')

    class Icosahedron(Polyhedron):
        def __init__(self, size=100):
            r = size
            _bottom = []
            for i in range(5):
                angle = 2 * pi * i / 5
                _bottom.append(Point(r * cos(angle), r * sin(angle), -r/2))

            _top = []
            for i in range(5):
                angle = 2 * pi * i / 5 + pi / 5
                _top.append(Point(r * cos(angle), r * sin(angle), r/2))

            top = Polygon(_top)
            bottom = Polygon(_bottom)

            polygons = []

            bottom_p = bottom.center
            top_p = top.center

            bottom_p.z -= r / 2
            top_p.z += r / 2

            for i in range(5):
                polygons.append(
                    Polygon([_bottom[i], bottom_p, _bottom[(i + 1) % 5]]))

            for i in range(5):
                polygons.append(
                    Polygon([_bottom[i], _top[i], _bottom[(i + 1) % 5]]))

            for i in range(5):
                polygons.append(
                    Polygon([_top[i], _top[(i + 1) % 5], _bottom[(i + 1) % 5]]))

            for i in range(5):
                polygons.append(Polygon([_top[i], top_p, _top[(i + 1) % 5]]))

            super().__init__(polygons)

        def __repr__(self) -> str:
            return super().__repr__().replace('Models.Icosahedron', 'Polyhedron')

    class Dodecahedron(Polyhedron):
        def __init__(self, size=100):
            t = Models.Icosahedron(size)
            points = []
            for polygon in t.polygons:
                points.append(polygon.center)
            p = points
            polygons = [
                Polygon([p[0], p[1], p[2], p[3], p[4]]),
                Polygon([p[0], p[4], p[9], p[14], p[5]]),
                Polygon([p[0], p[5], p[10], p[6], p[1]]),
                Polygon([p[1], p[2], p[7], p[11], p[6]]),
                Polygon([p[2], p[3], p[8], p[12], p[7]]),
                Polygon([p[3], p[8], p[13], p[9], p[4]]),
                Polygon([p[5], p[14], p[19], p[15], p[10]]),
                Polygon([p[6], p[11], p[16], p[15], p[10]]),
                Polygon([p[7], p[12], p[17], p[16], p[11]]),
                Polygon([p[8], p[13], p[18], p[17], p[12]]),
                Polygon([p[9], p[14], p[19], p[18], p[13]]),
                Polygon([p[15], p[16], p[17], p[18], p[19]])
            ]
            super().__init__(polygons)

        def __repr__(self) -> str:
            return super().__repr__().replace('Models.Dodecahedron', 'Polyhedron')


class App(tk.Tk):
    W: int = 1200
    H: int = 600
    shape: Shape = None
    shape_type_idx: int
    shape_type: ShapeType
    func_idx: int
    func: Function
    projection: Projection
    projection_idx: int
    phi: int = 60
    theta: int = 45
    dist: int = 1000
    bfc: tk.BooleanVar
    zbuf: tk.BooleanVar

    def __init__(self):
        super().__init__()
        self.title("ManualCAD 4D")
        self.resizable(0, 0)
        self.geometry(f"{self.W+200}x{70}+0+0")
        self.shape_type_idx = 0
        self.shape_type = ShapeType(self.shape_type_idx)
        self.func_idx = 0
        self.func = Function(self.func_idx)
        self.projection_idx = 0
        self.projection = Projection(self.projection_idx)
        self.create_widgets()
        pg.display.set_caption("Viewport")
        pg.init()

    def create_widgets(self):
        self.canvas = pg.display.set_mode((self.W, self.H))
        self.canvas.fill('#393939')
        pg.display.update()
        self.buttons = tk.Frame(self)
        self.translateb = tk.Button(
            self.buttons, text="Смещение", command=self.translate)
        self.rotateb = tk.Button(
            self.buttons, text="Поворот", command=self.rotate)
        self.scaleb = tk.Button(
            self.buttons, text="Масштаб", command=self.scale)
        self.phis = tk.Scale(self.buttons, from_=0, to=360,
                             orient=tk.HORIZONTAL, label="φ", command=self._phi_changed)
        self.thetas = tk.Scale(self.buttons, from_=0, to=360, orient=tk.HORIZONTAL,
                               label="θ", command=self._theta_changed)
        self.dists = tk.Scale(self.buttons, from_=1, to=self.W, orient=tk.HORIZONTAL,
                              label="Расстояние", command=self._dist_changed)

        self._axis = tk.BooleanVar()
        self.axis = tk.Checkbutton(self.buttons, text="Оси", var=self._axis, command=self.reset)

        self._grid = tk.BooleanVar()
        self.grid = tk.Checkbutton(self.buttons, text="Сетка", var=self._grid, command=self.reset)

        App.bfc = tk.BooleanVar()
        self._bfc = tk.Checkbutton(self.buttons, text="Back-face culling", var=App.bfc, command=self.reset)

        App.zbuf = tk.BooleanVar()
        self._zbuf = tk.Checkbutton(self.buttons, text="Z-buffer", var=App.zbuf, command=self.reset)

        self.shapesbox = tk.Listbox(
            self.buttons, selectmode=tk.SINGLE, height=1, width=16)
        self.scroll1 = tk.Scrollbar(
            self.buttons, orient=tk.VERTICAL, command=self._scroll1)
        self.funcsbox = tk.Listbox(
            self.buttons, selectmode=tk.SINGLE, height=1, width=40)
        self.scroll2 = tk.Scrollbar(
            self.buttons, orient=tk.VERTICAL, command=self._scroll2)
        self.projectionsbox = tk.Listbox(
            self.buttons, selectmode=tk.SINGLE, height=1, width=20)
        self.scroll3 = tk.Scrollbar(
            self.buttons, orient=tk.VERTICAL, command=self._scroll3)

        # self.canvas.pack()
        # self.canvas.config(cursor="cross")
        self.buttons.pack(fill=tk.X)
        self.translateb.pack(side=tk.LEFT, padx=5)
        self.rotateb.pack(side=tk.LEFT, padx=5)
        self.scaleb.pack(side=tk.LEFT, padx=5)
        self.phis.pack(side=tk.LEFT, padx=5)
        self.thetas.pack(side=tk.LEFT, padx=5)
        self.dists.pack(side=tk.LEFT, padx=5)
        self.axis.pack(side=tk.LEFT, padx=5)
        self.grid.pack(side=tk.LEFT, padx=5)
        self._bfc.pack(side=tk.LEFT, padx=5)
        self._zbuf.pack(side=tk.LEFT, padx=5)

        self.phis.set(self.phi)
        self.thetas.set(self.theta)
        self.dists.set(self.dist)
        App.bfc.set(False)
        App.zbuf.set(False)

        self.scroll1.pack(side=tk.RIGHT, fill=tk.Y)
        self.shapesbox.pack(side=tk.RIGHT, padx=1)
        self.shapesbox.config(yscrollcommand=self.scroll1.set)

        self.scroll3.pack(side=tk.RIGHT, fill=tk.Y)
        self.projectionsbox.pack(side=tk.RIGHT, padx=1)
        self.projectionsbox.config(yscrollcommand=self.scroll3.set)

        self.scroll2.pack(side=tk.RIGHT, fill=tk.Y)
        self.funcsbox.pack(side=tk.RIGHT, padx=1)
        self.funcsbox.config(yscrollcommand=self.scroll2.set)

        self.shapesbox.delete(0, tk.END)
        self.shapesbox.insert(tk.END, *ShapeType)
        self.shapesbox.selection_set(0)

        self.funcsbox.delete(0, tk.END)
        self.funcsbox.insert(tk.END, *Function)
        self.funcsbox.selection_set(0)

        self.projectionsbox.delete(0, tk.END)
        self.projectionsbox.insert(tk.END, *Projection)
        self.projectionsbox.selection_set(0)

        # self.bind("<Button-1>", self.l_click)
        # self.bind("<Button-3>", self.r_click)
        self.bind("<Escape>", self.reset)
        self.bind("<KeyPress>", self.key_pressed)
        self.bind("<F1>", self.camset)

    def camset(self, *_):
        x, y, z = map(int, sd.askstring("Камера", "Введите координаты камеры через пробел").split())
        Camera.position = Point(x, y, z)

    def reset(self, *_, del_shape=True):
        self.canvas.fill('#393939')
        ZBuffer.clear()
        # pg.display.update()
        if del_shape:
            self.shape = None

        if self._grid.get():
            for i in range(-self.W, self.W, 50):
                Line(Point(i, 0, -self.H), Point(i, 0, self.H)).draw(self.canvas, self.projection, color='gray')
            for i in range(-self.H, self.H, 50):
                Line(Point(-self.W, 0, i), Point(self.W, 0, i)).draw(self.canvas, self.projection, color='gray')

        if self._axis.get():
            ln = 100
            Line(Point(-ln, 0, 0), Point(ln, 0, 0)).draw(self.canvas, self.projection, color='red')  # x axis
            Line(Point(0, -ln, 0), Point(0, ln, 0)).draw(self.canvas, self.projection, color='green')  # y axis
            Line(Point(0, 0, -ln), Point(0, 0, ln)).draw(self.canvas, self.projection, color='blue')  # z axis

    def rotate(self):
        inp = sd.askstring(
            "Поворот", "Введите угол поворота в градусах по x, y, z:")
        if inp is None:
            return
        phi, theta, psi = map(radians, map(float, inp.split(',')))
        m, n, k = self.shape.center

        mat_back = np.array([
            [1, 0, 0, -m],
            [0, 1, 0, -n],
            [0, 0, 1, -k],
            [0, 0, 0, 1]
        ])

        mat_x = np.array([
            [1, 0, 0, 0],
            [0, cos(phi), sin(phi), 0],
            [0, -sin(phi), cos(phi), 0],
            [0, 0, 0, 1]])

        mat_y = np.array([
            [cos(theta), 0, -sin(theta), 0],
            [0, 1, 0, 0],
            [sin(theta), 0, cos(theta), 0],
            [0, 0, 0, 1]])

        mat_z = np.array([
            [cos(psi), -sin(psi), 0, 0],
            [sin(psi), cos(psi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

        mat_fwd = np.array([
            [1, 0, 0, m],
            [0, 1, 0, n],
            [0, 0, 1, k],
            [0, 0, 0, 1]
        ])

        mat = mat_fwd @ mat_x @ mat_y @ mat_z @ mat_back
        self.shape.transform(mat)
        self.reset(del_shape=False)
        self.shape.draw(self.canvas, self.projection)
        pg.display.update()

    def scale(self):
        inp = sd.askstring(
            "Масштаб", "Введите коэффициенты масштабирования по осям x, y, z:")
        if inp is None:
            return
        sx, sy, sz = map(float, inp.split(','))
        mat = np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]])
        self.shape.transform(mat)
        self.reset(del_shape=False)
        self.shape.draw(self.canvas, self.projection)
        pg.display.update()

    def translate(self):
        inp = sd.askstring(
            "Смещение", "Введите вектор смещения по осям x, y, z:")
        if inp is None:
            return
        dx, dy, dz = map(float, inp.split(','))
        mat = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]])
        self.shape.transform(mat)
        self.reset(del_shape=False)
        self.shape.draw(self.canvas, self.projection)
        pg.display.update()

    def _scroll1(self, *args):
        try:
            d = int(args[1])
        except ValueError:
            return
        if 0 <= self.shape_type_idx + d < len(ShapeType):
            self.shape_type_idx += d
            self.shape_type = ShapeType(self.shape_type_idx)
            self.shape = None
            self.shapesbox.yview(*args)

    def _scroll2(self, *args):
        try:
            d = int(args[1])
        except ValueError:
            return
        if 0 <= self.func_idx + d < len(Function):
            self.func_idx += d
            self.func = Function(self.func_idx)
            self.funcsbox.yview(*args)

    def _scroll3(self, *args):
        try:
            d = int(args[1])
        except ValueError:
            return
        if 0 <= self.projection_idx + d < len(Projection):
            self.projection_idx += d
            self.projection = Projection(self.projection_idx)
            self.projectionsbox.yview(*args)
            self.reset(del_shape=False)
            if self.shape is not None:
                self.shape.draw(self.canvas, self.projection)
                pg.display.update()

    def _dist_changed(self, *_):
        App.dist = self.dists.get()
        self.reset(del_shape=False)
        if self.shape is not None:
            self.shape.draw(self.canvas, self.projection)
            pg.display.update()

    def _phi_changed(self, *_):
        App.phi = self.phis.get()
        self.reset(del_shape=False)
        if self.shape is not None:
            self.shape.draw(self.canvas, self.projection)
            pg.display.update()

    def _theta_changed(self, *_):
        App.theta = self.thetas.get()
        self.reset(del_shape=False)
        if self.shape is not None:
            self.shape.draw(self.canvas, self.projection)
            pg.display.update()

    def l_click(self, _):
        self.reset()
        match self.shape_type:
            case ShapeType.Tetrahedron:
                self.shape = Models.Tetrahedron()
            case ShapeType.Octahedron:
                self.shape = Models.Octahedron()
            case ShapeType.Hexahedron:
                self.shape = Models.Hexahedron()
            case ShapeType.Icosahedron:
                self.shape = Models.Icosahedron()
            case ShapeType.Dodecahedron:
                self.shape = Models.Dodecahedron()
            case ShapeType.RotationBody:
                def __thread():
                    t = tk.Tk()
                    t.withdraw()
                    inp = sd.askstring("Параметры", "Введите набор точек, ось вращения и количество разбиений через запятую:", parent=t)
                    if inp is None:
                        return
                    *points, axis, patritions = inp.split(',')
                    if not len(points) % 3 == 0:
                        return
                    poly = []
                    for i in range(0, len(points), 3):
                        poly.append(Point(float(points[i]), float(points[i + 1]), float(points[i + 2])))
                    self.shape = RotationBody(Polygon(poly), axis.strip().upper(), int(patritions))
                    t.destroy()
                    # Coin: 0, 0, 0, 0, 100, 0, 0, 0, 100, Y, 120
                t = Thread(target=__thread)
                t.start()
                t.join()
        if self.shape is not None:
            if App.zbuf.get():
                self.__temp_model()
                self.shape.fill(self.canvas, pg.Color('green'))
            else:
                self.shape.draw(self.canvas, self.projection)
            pg.display.update()

    def __temp_model(self):
        t = Models.Tetrahedron()
        t.transform(np.array([
            [1, 0, 0, -50],
            [0, 1, 0, 0],
            [0, 0, 1, -50],
            [0, 0, 0, 1]]))
        # t.draw(self.canvas, self.projection, color=pg.Color('red'))
        t.fill(self.canvas, pg.Color('red'))

    def r_click(self, _):
        if self.shape is None:
            return

        def __thread():
            t = tk.Tk()
            t.withdraw()

            match self.func:
                case Function.None_:
                    return

                case Function.ReflectOverPlane:
                    # https://www.gatevidyalay.com/3d-reflection-in-computer-graphics-definition-examples/
                    inp = sd.askstring(
                        "Отражение", "Введите плоскость отражения (н-р: XY):", parent=t)
                    if inp is None:
                        return
                    plane = ''.join(sorted(inp.strip().upper()))

                    mat_xy = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])

                    mat_yz = np.array([
                        [-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

                    mat_xz = np.array([
                        [1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

                    match plane:
                        case 'XY':
                            self.shape.transform(mat_xy)
                        case 'YZ':
                            self.shape.transform(mat_yz)
                        case 'XZ':
                            self.shape.transform(mat_xz)
                        case _:
                            mb.showerror("Ошибка", "Неверно указана плоскость")
                    self.reset(del_shape=False)
                    self.shape.draw(self.canvas, self.projection)
                    pg.display.update()

                case Function.ScaleAboutCenter:
                    inp = sd.askstring("Масштаб", "Введите коэффициенты масштабирования по осям x, y, z:", parent=t)
                    if inp is None:
                        return
                    sx, sy, sz = map(float, inp.split(','))
                    m, n, k = self.shape.center
                    mat = np.array([
                        [sx, 0, 0, -m*sx+m],
                        [0, sy, 0, -n*sy+n],
                        [0, 0, sz, -k*sz+k],
                        [0, 0, 0, 1]])
                    self.shape.transform(mat)
                    self.reset(del_shape=False)
                    self.shape.draw(self.canvas, self.projection)
                    pg.display.update()

                case Function.RotateAroundAxis:
                    m, n, k = self.shape.center
                    inp = sd.askstring("Поворот", "Введите ось вращения (н-р: X), угол в градусах:", parent=t)
                    if inp is None:
                        return
                    try:
                        axis, phi = inp.split(',')
                        axis = axis.strip().upper()
                        phi = radians(float(phi))
                    except ValueError:
                        mb.showerror("Ошибка", "Неверно указаны ось и угол")
                        return

                    mat_back = np.array([
                        [1, 0, 0, -m],
                        [0, 1, 0, -n],
                        [0, 0, 1, -k],
                        [0, 0, 0, 1]])
                    self.shape.transform(mat_back)

                    match axis:
                        case 'X':
                            mat = np.array([
                                [1, 0, 0, 0],
                                [0, cos(phi), -sin(phi), 0],
                                [0, sin(phi), cos(phi), 0],
                                [0, 0, 0, 1]])  # вращение вокруг оси x
                        case 'Z':
                            mat = np.array([
                                [cos(phi), -sin(phi), 0, 0],
                                [sin(phi), cos(phi), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])  # вращение вокруг оси z
                        case 'Y':
                            mat = np.array([
                                [cos(phi), 0, sin(phi), 0],
                                [0, 1, 0, 0],
                                [-sin(phi), 0, cos(phi), 0],
                                [0, 0, 0, 1]])  # вращение вокруг оси y

                    self.shape.transform(mat)
                    mat_fwd = np.array([
                        [1, 0, 0, m],
                        [0, 1, 0, n],
                        [0, 0, 1, k],
                        [0, 0, 0, 1]])
                    self.shape.transform(mat_fwd)
                    self.reset(del_shape=False)
                    self.shape.draw(self.canvas, self.projection)
                    pg.display.update()

                case Function.RotateAroundLine:
                    inp = sd.askstring("Поворот", "Введите координаты начала и конца линии в формате x1, y1, z1, x2, y2, z2, угол в градусах:", parent=t)
                    if inp is None:
                        return
                    try:
                        a, b, c, x, y, z, phi = map(float, inp.split(','))
                        phi = radians(phi)
                    except ValueError:
                        mb.showerror("Ошибка", "Неверно указаны координаты начала и конца линии")
                        return

                    l = Line(Point(a, b, c), Point(x, y, z))

                    d = np.linalg.norm([x, y, z])
                    x = x / d
                    y = y / d
                    z = z / d

                    mat_back = np.array([
                        [1, 0, 0, -a],
                        [0, 1, 0, -b],
                        [0, 0, 1, -c],
                        [0, 0, 0, 1]])

                    mat_rot = np.array([
                        [cos(phi) + (1 - cos(phi)) * x ** 2, (1 - cos(phi)) * x * y - sin(phi)*z, (1 - cos(phi)) * x * z + sin(phi)*y, 0],
                        [(1 - cos(phi)) * x * y + sin(phi)*z, cos(phi) + (1 - cos(phi)) * y ** 2, (1 - cos(phi)) * y * z - sin(phi)*x, 0],
                        [(1 - cos(phi)) * z * x - sin(phi)*y, (1 - cos(phi)) * z * y + sin(phi)*x, cos(phi) + (1 - cos(phi)) * z ** 2, 0],
                        [0, 0, 0, 1]
                    ])  # 0, 0, 150, 120, 300, -50, 90

                    mat_fwd = np.array([
                        [1, 0, 0, a],
                        [0, 1, 0, b],
                        [0, 0, 1, c],
                        [0, 0, 0, 1]])

                    mat = mat_fwd @ mat_rot @ mat_back
                    self.shape.transform(mat)
                    self.reset(del_shape=False)
                    l.draw(self.canvas, self.projection, color='orange')
                    self.shape.draw(self.canvas, self.projection)
                    pg.display.update()
            t.destroy()

        t = Thread(target=__thread)
        t.start()
        t.join()

    def key_pressed(self, event: tk.Event):
        if event.keysym == 'z':
            path = fd.askopenfilename(filetypes=[('Файлы с фигурами', '*.shape')])
            if path:
                self.shape = Shape.load(path)
                self.reset(del_shape=False)
                self.shape.draw(self.canvas, self.projection)

        elif event.keysym == 'x':
            path = fd.asksaveasfilename(filetypes=[('Файлы с фигурами', '*.shape')])
            if path:
                self.shape.save(path)

        elif event.keysym == 'f':
            if ShapeType.FuncPlot:
                inp = sd.askstring(
                    "Параметры", "Введите функцию, диапазонамы отсечения [x0, x1] и [y0, y1], количество разбиений по x и y через запятую:")
                if inp is None:
                    return
                func, x0, x1, y0, y1, nx, ny = map(str.strip, inp.split(','))
                self.shape = FuncPlot(func, float(x0), float(x1), float(y0), float(y1), int(nx), int(ny))
                self.shape.draw(self.canvas, self.projection)
                pg.display.update()

    def run(self):
        Thread(target=self._pg_mainloop).start()
        self.mainloop()

    def destroy(self) -> None:
        pg.quit()
        super().destroy()

    def _pg_mainloop(self):
        while True:
            for e in pg.event.get():
                if e.type == pg.MOUSEBUTTONDOWN:
                    if e.button == 1:
                        self.l_click(e.pos)
                    elif e.button == 3:
                        self.r_click(e.pos)
                if e.type == pg.KEYDOWN:
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
                self.reset(del_shape=False)
                if self.shape is not None:
                    if App.zbuf.get():
                        self.__temp_model()
                        if isinstance(self.shape, Polyhedron):
                            self.shape.fill(self.canvas, pg.Color('green'))
                    else:
                        self.shape.draw(self.canvas, self.projection)
                    pg.display.update()


class ZBuffer:
    enabled: bool = False
    data: np.ndarray = np.full((App.H, App.W), np.inf)

    @staticmethod
    def clear():
        ZBuffer.data.fill(np.inf)
        ZBuffer.enabled = App.zbuf.get()

    @staticmethod
    def draw_point(canvas: pg.Surface, x: int, y: int, z: float, color: pg.Color):
        #d = Camera.dist_to(x, y, z)
        if ZBuffer.enabled and 0 <= x < App.W and 0 <= y < App.H:
            if ZBuffer.data[y, x] > z:
                ZBuffer.data[y, x] = z
                canvas.set_at((x, y), color)
        else:
            canvas.set_at((x, y), color)


if __name__ == "__main__":
    app = App()
    app.run()

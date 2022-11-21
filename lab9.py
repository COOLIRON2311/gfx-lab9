import tkinter as tk
from dataclasses import dataclass, field, InitVar
from enum import Enum
from math import cos, pi, radians, sin, sqrt
from threading import Thread
from tkinter import filedialog as fd
from tkinter import simpledialog as sd

import numpy as np
import pygame as pg

from enums import Projection, ShapeType

# pylint: disable=no-member
# pylint: disable=eval-used
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-lines


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
            if isinstance(s, Polyhedron):
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

    def screen_coords(self, projection: Projection) -> 'Point':
        if projection == Projection.Perspective:
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
        return Point(x, y, z)

    def draw(self, canvas: pg.Surface, projection: Projection, color: str = 'white', draw_points: bool = True):
        if draw_points and self.x < 1000 and self.y < 1000:
            canvas.set_at((int(self.x), int(self.y)), pg.Color(color))

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
        p1X, p1Y, p1Z = self.p1.screen_coords(projection)
        p2X, p2Y, p2Z = self.p2.screen_coords(projection)
        self.__wu(canvas, Point(p1X, p1Y, p1Z), Point(p2X, p2Y, p2Z), pg.Color(color))
        return Point(p1X, p1Y, p1Z), Point(p2X, p2Y, p2Z)

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

        for _ in np.arange(x1, x2):
            res.append(z1)
            z1 += step

        return res

    def triangulate(self):
        if len(self.points) == 3:
            return [self]
        res = []
        for i in range(2, len(self.points)):
            res.append(Polygon([self.points[0], self.points[i-1], self.points[i]]))
        return res

    def col_interp(self, c1: np.ndarray, c2: np.ndarray, t: float) -> np.ndarray:
        return c1 + t * (c2 - c1)

    def fill(self, canvas: pg.Surface, color: pg.Color):
        normals = self.triang_normales()
        mod = np.linalg.norm
        c = []

        vecToLight = np.array([
            LightSource.pos.x-self.points[0].x,
            LightSource.pos.y-self.points[0].y,
            LightSource.pos.z-self.points[0].z])
        c.append((normals[0] @ vecToLight) / (mod(normals[0]) * mod(vecToLight)) * color)
        vecToLight = np.array([
            LightSource.pos.x-self.points[1].x,
            LightSource.pos.y-self.points[1].y,
            LightSource.pos.z-self.points[1].z])
        c.append((normals[1] @ vecToLight) / (mod(normals[1]) * mod(vecToLight)) * color)
        vecToLight = np.array([
            LightSource.pos.x-self.points[2].x,
            LightSource.pos.y-self.points[2].y,
            LightSource.pos.z-self.points[2].z])
        c.append((normals[2] @ vecToLight) / (mod(normals[2]) * mod(vecToLight)) * color)

        points = zip([self.points[i].screen_coords(Projection.FreeCamera) for i in range(len(self.points))], c)
        points = sorted(points, key=lambda x: x[0].y)
        p1: Point
        p2: Point
        p3: Point
        p1, p2, p3 = points[0][0], points[1][0], points[2][0]
        c1, c2, c3 = points[0][1], points[1][1], points[2][1]

        l1 = Line(p1, p2)
        l2 = Line(p1, p3)

        hleft = p3.y - p1.y
        hright = p2.y - p1.y

        for y in range(int(p1.y), int(p3.y)):
            tl = 0 if hleft == 0 else (y - p1.y) / hleft
            tr = 0 if hright == 0 else (y - p1.y) / hright
            cl = self.col_interp(c1, c3, tl)
            cr = self.col_interp(c1, c2, tr)
            xl, xr = l1.get_x(y), l2.get_x(y)
            if xl > xr:
                xl, xr = xr, xl
                cl, cr = cr, cl
            for x in range(int(xl), int(xr)):
                # TODO: bullshit
                z = self.interpolate(xl, p1.z, xr, p2.z)[x-int(xl)]
                t = 0 if xr == xl else (x - xl) / (xr - xl)
                cx = self.col_interp(cl, cr, t)
                # canvas.set_at((x, y), cx)
                col = pg.Color(int(cx[0]), int(cx[1]), int(cx[2]))
                ZBuffer.draw_point(canvas, x, y, z, col)

        l1 = Line(p1, p3)
        l2 = Line(p2, p3)

        hleft = p3.y - p1.y
        hright = p3.y - p2.y

        for y in range(int(p2.y), int(p3.y)):
            tl = 0 if hleft == 0 else (y - p1.y) / hleft
            tr = 0 if hright == 0 else (y - p2.y) / hright
            cl = self.col_interp(c1, c3, tl)
            cr = self.col_interp(c2, c3, tr)
            xl, xr = l1.get_x(y), l2.get_x(y)
            if xl > xr:
                xl, xr = xr, xl
                cl, cr = cr, cl
            for x in range(int(xl), int(xr)):
                # TODO: bullshit
                z = self.interpolate(xl, p1.z, xr, p3.z)[x-int(xl)]
                t = 0 if xr == xl else (x - xl) / (xr - xl)
                cx = self.col_interp(cl, cr, t)
                # canvas.set_at((x, y), cx)
                col = pg.Color(int(cx[0]), int(cx[1]), int(cx[2]))
                ZBuffer.draw_point(canvas, x, y, z, col)

    # def fill(self, canvas: pg.Surface, color: pg.Color):
    #     ln = len(self.points)
    #     tlines = [Line(self.points[i], self.points[(i + 1) % ln])
    #               for i in range(ln)]
    #     lines: list[Line] = []
    #     points: set[Point] = set()
    #     for l in tlines:
    #         p1, p2 = l.draw(canvas, Projection.FreeCamera)
    #         lines.append(Line(p1, p2))
    #         points.add(p1)
    #         points.add(p2)
    #     ymax = max(p.y for p in points)
    #     ymin = min(p.y for p in points)

    #     for y in range(int(ymin), int(ymax)):
    #         intersections: list[Point] = []
    #         for line in lines:
    #             if line.p1.y <= y < line.p2.y or line.p2.y <= y < line.p1.y:
    #                 intersections.append(Point(line.get_x(y), y, line.get_z(y)))
    #         intersections.sort(key=lambda p: p.x)
    #         for i in range(0, len(intersections), 2):
    #             z = self.interpolate(intersections[i].x, intersections[i].z, intersections[i+1].x, intersections[i+1].z)
    #             for x in range(int(intersections[i].x), int(intersections[i+1].x)):
    #                 cz = z[x-int(intersections[i].x)]
    #                 ZBuffer.draw_point(canvas, x, y, cz, color)

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
        p1 = np.array(self.points[0])
        p2 = np.array(self.points[1])
        p3 = np.array(self.points[2])
        v1 = p1 - p2
        v2 = p3 - p2
        normal = np.cross(v2, v1)
        return Point(normal[0], normal[1], normal[2])

    def triang_normales(self) -> list[np.ndarray]:
        res = []
        ln = len(self.points)
        for i in range(0, ln):
            v1 = np.array(self.points[(i-1) % ln]) - np.array(self.points[i % ln])
            v2 = np.array(self.points[(i+1) % ln]) - np.array(self.points[i % ln])
            res.append(np.cross(v1, v2))
        return res


@dataclass
class Polyhedron(Shape):
    polygons: list[Polygon]
    _triangulate: InitVar[bool] = field(default=True)

    def __post_init__(self, _triangulate):
        if _triangulate:
            polys = []
            for poly in self.polygons:
                polys.extend(poly.triangulate())
            self.polygons = polys

            for poly in self.polygons:
                poly.normal = poly.calculate_normal()

    def draw(self, canvas: pg.Surface, projection: Projection, color: str = 'white', draw_points: bool = False):
        bfc: bool = App.bfc.get()
        p = Camera.camFront + np.array(Camera.position)
        for poly in self.polygons:
            if bfc:
                v0 = np.array(poly.points[0])
                n = np.array(poly.normal)
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

    def fill(self, canvas: pg.Surface, _: pg.Color):
        count = 0
        colors = [pg.Color("red"), pg.Color("green"), pg.Color("blue"), pg.Color('yellow')]
        for poly in self.polygons:
            poly.fill(canvas, colors[count % 4])
            count += 1


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


class LightSource:
    pos: Point = Point(200, 200, 0)


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
            t = Models.Hexahedron(size, False)
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
        def __init__(self, size=100, triangulate=True):
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
            super().__init__(polygons, triangulate)

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
    W: int = 1000
    H: int = 600
    shape: Shape = None
    shape_type_idx: int
    shape_type: ShapeType
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
        self.projectionsbox = tk.Listbox(
            self.buttons, selectmode=tk.SINGLE, height=1, width=20)
        self.scroll3 = tk.Scrollbar(
            self.buttons, orient=tk.VERTICAL, command=self._scroll2)

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

        self.shapesbox.delete(0, tk.END)
        self.shapesbox.insert(tk.END, *ShapeType)
        self.shapesbox.selection_set(0)

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
        if self.shape is not None:
            if App.zbuf.get():
                # self.__temp_model()
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
                    # elif e.button == 3:
                    #     self.r_click(e.pos)
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
                        # self.__temp_model()
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
        if ZBuffer.enabled and 0 <= x < App.W and 0 <= y < App.H:
            if ZBuffer.data[y, x] > z:
                ZBuffer.data[y, x] = z
                canvas.set_at((x, y), color)
        else:
            canvas.set_at((x, y), color)


if __name__ == "__main__":
    app = App()
    app.run()

from enum import Enum


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


class ShapeType(Enum):
    Tetrahedron = 0
    Hexahedron = 1
    Octahedron = 2
    Icosahedron = 3
    Dodecahedron = 4

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
            case _:
                pass
        return "Неизвестная фигура"

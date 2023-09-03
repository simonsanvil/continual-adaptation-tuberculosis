from dataclasses import dataclass, field
from functools import cached_property
import numbers
from typing import Any, Dict, Iterable, Optional, Union, Tuple, List
from pydantic import BaseModel, Field

from shapely import geometry
import numpy as np

class bbox:
    
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            self.xmin, self.ymin, self.xmax, self.ymax = args[0]
        elif len(args) >= 4:
            self.xmin, self.ymin, self.xmax, self.ymax = args
        elif len(kwargs) >= 4:
            self.xmin = kwargs["xmin"]
            self.xmax = kwargs["xmax"]
            self.ymin = kwargs["ymin"]
            self.ymax = kwargs["ymax"]
        else:
            raise ValueError("Invalid arguments. Must be either 4 numbers or a tuple of 4 numbers or a dict with keys xmin, ymin, xmax, ymax")

    def shapely(self) -> geometry.box:
        return geometry.box(self.xmin, self.ymin, self.xmax, self.ymax)
    
    def dict(self) -> Dict[str, float]:
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax
        }
    
    def __iter__(self) -> Iterable[float]:
        return iter([self.xmin, self.ymin, self.xmax, self.ymax])
    
    def __getitem__(self, key) -> float:
        return [self.xmin, self.ymin, self.xmax, self.ymax][key]
    
    def __repr__(self) -> str:
        return f"bbox({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax})"
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]):
        return cls(**d)

class Rect(BaseModel):
    left: float = None
    top: float = None
    width: float = None
    height: float = None
    label: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    def ___init__(self, left:float=None, top:float=None, width:float=None, height:float=None, **kwargs):
        data = {}
        data['left'] = left or kwargs.get("xmin")
        data['top'] = top or kwargs.get("ymin")
        if data['left'] is None or data['top'] is None:
            raise ValueError("Must provide either left and top or xmin and ymin")
        data['width'] = width 
        data['height'] = height
        data['label'] = kwargs.pop("label", None)
        if kwargs.get("xmax") and data['width'] is None:
            data['width'] = kwargs["xmax"] - data.get('left')
        if kwargs.get("ymax") and data['height'] is None:
            data['height'] = kwargs["ymax"] - data.get('top')
        if data['width'] is None or data['height'] is None:
            raise ValueError("Must provide either width and height or xmax and ymax")
        for k in ['left', 'top', 'width', 'height']:
            assert isinstance(data[k], numbers.Number), f"{k} must be a number"        
            assert data[k] >= 0, f"{k} must be positive"
        
        data['meta'] = {**kwargs.pop("meta", {}), **kwargs}
        super().__init__(**data)

    @property
    def right(self) -> float:
        return self.left + self.width
    
    @property
    def bottom(self) -> float:
        return self.top + self.height
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        return (self.left, self.top, self.width, self.height)
    
    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)
    
    @property
    def cxcywh(self) -> Tuple[float, float, float, float]:
        return (self.left + self.width/2, self.top + self.height/2, self.width, self.height)
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.left + self.width/2, self.top + self.height/2)
    
    def intersects(self, other: "Rect") -> bool:
        return self.shapely().intersects(other.shapely())
    
    def __getattr__(self, key):
        if key not in ["left", "top", "width", "height", "label", "meta", "right", "bottom", "area", "xywh", "xyxy", "center"]:
            if key in self.meta:
                return self.meta[key]
        return self.__getattribute__(key)
    
    def __iter__(self) -> Iterable[float]:
        return iter([self.left, self.top, self.width, self.height])
    
    def __getitem__(self, key) -> float:
        return [self.left, self.top, self.right, self.bottom][key]
    
    def __eq__(self, other: "Rect") -> bool:
        self_vec = np.array(list(self))
        other_vec = np.array(list(other))
        return np.allclose(self_vec, other_vec)
    
    def todict(self) -> Dict[str, float]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "label": self.label,
            "meta": {k: v for k, v in self.meta.items() if k not in ["left", "top", "width", "height", "label"]}
        }

    def shapely(self) -> geometry.box:
        return geometry.box(self.left, self.top, self.right, self.bottom)

    def bbox(self) -> bbox:
        return bbox(self.left, self.top, self.right, self.bottom)

    @classmethod
    def from_bbox(cls, 
            bbox:Union[Dict[str, float], Tuple[float, float, float, float]],
            bbox_format:str="xyxy",
            label=None,
            **kwargs
        ) -> "Rect":
        if bbox_format == "xyxy":
            if isinstance(bbox, dict):
                return cls(
                    left=bbox["minx"],
                    top=bbox["miny"],
                    width=bbox["maxx"] - bbox["minx"],
                    height=bbox["maxy"] - bbox["miny"],
                    label=label,
                    meta=kwargs
                )
            elif isinstance(bbox, Iterable):
                return cls(
                    left=bbox[0],
                    top=bbox[1],
                    width=bbox[2] - bbox[0],
                    height=bbox[3] - bbox[1],
                    label=label,
                    meta=kwargs
                )
            else:
                raise ValueError("Invalid bbox. Must be either a dict or an iterable in the order (minx, miny, maxx, maxy)")
        elif bbox_format == "xywh":
            if isinstance(bbox, dict):
                return cls(
                    left=bbox.get("xmin", bbox.get("left")),
                    top=bbox.get("ymin", bbox.get("top")),
                    width=bbox["width"],
                    height=bbox["height"],
                    label=label,
                    meta=kwargs
                )
            elif isinstance(bbox, Iterable):
                return cls(
                    left=bbox[0],
                    top=bbox[1],
                    width=bbox[2],
                    height=bbox[3],
                    label=label,
                    meta=kwargs
                )
            else:
                raise ValueError("Invalid bbox. Must be either a dict or an iterable in the order (xmin, ymin, width, height)")
    
    @classmethod
    def create(cls, *args, **kwargs) -> "Rect":
        if len(args) == 0:
            pass
        elif len(args) == 1:
            if isinstance(args[0], Rect):
                return args[0]
            elif isinstance(args[0], dict):
                return cls(**args[0])
            elif isinstance(args[0], Iterable):
                left, top, width, height = args[0]
                kwargs.update(dict(left=left, top=top, width=width, height=height))
        elif len(args) == 4:
            left, top, width, height = args
            kwargs.update(dict(left=left, top=top, width=width, height=height))
        elif len(args) <= 5:
            left, top, width, height, label = (list(args) + [None] * 5)[:5]
            kwargs.update(dict(left=left, top=top, width=width, height=height, label=label))
        else:
            raise ValueError(f"Up to 5 positional arguments are allowed. {len(args)} given.")
        cr_kwargs = kwargs.copy()
        cr_kwargs['meta'] = kwargs.pop("meta", {})
        for k in kwargs:
            if k not in ["left", "top", "width", "height", "label", "meta"]:
                cr_kwargs['meta'][k] = kwargs[k]
        return cls(**cr_kwargs)
    
    def plot(self, ax:"plt.Axes", color:str="r", linewidth:float=1, **kwargs):
        """Plot the rectangle on a matplotlib axis"""
        from matplotlib import patches
        ax.add_patch(
            patches.Rectangle(
                (self.left, self.top),
                self.width,
                self.height,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
                **kwargs
            )
        )

    def __repr__(self) -> str:
        return f"Rect(left={self.left}, top={self.top}, width={self.width}, height={self.height}, label={self.label})"

class Rects(BaseModel):
    """A collection of (BBOX) rectangles"""

    rects:List[Union[Rect, Dict[str,float], Iterable[float]]]
    meta:Dict[str, Any]=None

    def __init__(self, rects:List[Union[Rect, Dict[str,float], Iterable[float]]]=None, **kwargs):
        data = {}
        rects = rects or kwargs.pop("rects", [])
        data["rects"] = [Rect.create(r) for r in rects]
        data['meta'] = {**kwargs.pop("meta", {}), **kwargs}
        super().__init__(**data)

    def __getattr__(self, key):
        if key not in ["_rects", "rects", "meta"]:
            return self.meta[key]
        return self.__getattribute__(key)

    def __iter__(self) -> Iterable[Rect]:
        return iter(self.rects)
    
    def __getitem__(self, key) -> Rect:
        return self.rects[key]
    
    def __len__(self) -> int:
        return len(self.rects)
    
    def __add__(self, other: "Rects") -> "Rects":
        if not isinstance(other, Rects):
            other = Rects(other)
        return Rects(self.rects + other.rects, meta=self.meta.copy())
    
    def __sub__(self, other: "Rects") -> "Rects":
        if not isinstance(other, Rects):
            other = Rects(other)
        subrects = [rect for rect in self.rects if rect not in other.rects]
        return Rects(subrects, meta=self.meta.copy())
    
    def __repr__(self) -> str:
        return f"Rects({[tuple(round(a,2) for a in r) for r in self.rects]})"
    
    def todict(self) -> List[Dict[str, float]]:
        return [rect.todict() for rect in self.rects]
    
    def shapely(self) -> geometry.MultiPolygon:
        return geometry.MultiPolygon([rect.shapely() for rect in self.rects])
    
    def bbox(self) -> List[bbox]:
        return [rect.bbox() for rect in self.rects]

    def append(self, rect: Union[Rect, Dict[str,float], Iterable[float]]) -> None:
        if isinstance(rect, dict):
            self.rects.append(Rect.from_bbox(rect))
        self.rects.append(Rect.create(rect))

    def setdiff(self, other: "Rects", index:bool=False) -> list:
        self_vec = np.array([list(rect) for rect in self.rects])
        other_vec = np.array([list(rect) for rect in other.rects])
        setdiff = np.setdiff1d(self_vec, other_vec, assume_unique=True)
        if len(setdiff) == 0:
            return []
        if index:
            return np.where(np.atleast_2d(np.isin(self_vec, setdiff))[:,0])[0].tolist()
        return setdiff.tolist()
    
    def union(self, other: "Rects") -> "Rects":
        union_rects = [rect for rect in other.rects if rect not in self.rects]
        return Rects(self.rects + union_rects, meta=self.meta.copy())
    
    def plot(self, ax:"Axes"=None, xlim:Tuple[float,float]=None, ylim:Tuple[float,float]=None, label:str=None, **kwargs) -> "plt.Axes":
        from matplotlib import pyplot as plt
        if ax is None:
            ax = plt.gca()
            if xlim is None and self.rects:
                max_x = max([rect.right for rect in self.rects])
                xlim = (0, max_x)
            if ylim is None and self.rects:
                max_y = max([rect.bottom for rect in self.rects])
                ylim = (max_y, 0)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(max_y, 0)
        for i, rect in enumerate(self.rects):
            if i==0:
                rect.plot(ax, label=label, **kwargs)
            else:
                rect.plot(ax, **kwargs)
        return ax
    
@dataclass
class rectchange:
    old: Rects
    current: Rects
    rtol: float = 5e-02
    atol: float = 2
    created_inds: list = field(init=False, default_factory=list)
    removed_inds: list = field(init=False, default_factory=list)
    unmodified_inds: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self._identify_changes(rtol=self.rtol, atol=self.atol)

    @property
    def created(self):
        return Rects([self.current[i] for i in self.created_inds])
    
    @property
    def removed(self):
        return Rects([self.old[i] for i in self.removed_inds])
    
    @property
    def unmodified(self):
        return Rects([self.old[i] for i in self.unmodified_inds])
    
    @property
    def changes(self):
        return self.created + self.removed
    
    @property
    def rects(self):
        return self.created + self.unmodified
    
    @property
    def has_changes(self):
        return len(self.created_inds) > 0 or len(self.removed_inds) > 0

    def _identify_changes(self, rtol:float=5e-02, atol:float=2):
        created_rects_inds = []
        unmodified_rects_inds = []
        for i,rn in enumerate(self.current):
            is_new_rect = True
            for j, ro in enumerate(self.old):
                if np.isclose(np.array(list(rn)), np.array(list(ro)), rtol=rtol, atol=atol).all():
                    is_new_rect = False
                    unmodified_rects_inds.append(j)
                    break
            if is_new_rect:
                # print("rect at index {} is new".format(i))
                created_rects_inds.append(i)
        
        self.created_inds=created_rects_inds
        self.unmodified_inds=unmodified_rects_inds
        self.removed_inds=[i for i in range(len(self.old)) if i not in unmodified_rects_inds]

        return self
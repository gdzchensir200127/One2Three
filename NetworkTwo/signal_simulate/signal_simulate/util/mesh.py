from dataclasses import dataclass, field
from typing import BinaryIO, Dict, Optional, Union

import numpy as np

from .ply_util import write_ply


@dataclass
class TriMesh:
    """
    A 3D triangle mesh with optional data at the vertices and faces.
    """

    # [N x 3] array of vertex coordinates.
    verts: np.ndarray

    # [M x 3] array of triangles, pointing to indices in verts.
    faces: np.ndarray

    # [P x 3] array of normal vectors per face.
    normals: Optional[np.ndarray] = None

    faces_normal: Optional[np.ndarray] = None

    # Extra data per vertex and face.
    vertex_channels: Optional[Dict[str, np.ndarray]] = field(default_factory=dict)
    face_channels: Optional[Dict[str, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "TriMesh":
        """
        Load the mesh from a .npz file.
        """
        if isinstance(f, str):
            with open(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            verts = obj["verts"]
            faces = obj["faces"]
            normals = obj["normals"] if "normals" in keys else None
            vertex_channels = {}
            face_channels = {}
            for key in keys:
                if key.startswith("v_"):
                    vertex_channels[key[2:]] = obj[key]
                elif key.startswith("f_"):
                    face_channels[key[2:]] = obj[key]
            return cls(
                verts=verts,
                faces=faces,
                normals=normals,
                vertex_channels=vertex_channels,
                face_channels=face_channels,
                faces_normal=None,
            )
    @classmethod
    def load_from_obj(cls, f: Union[str,BinaryIO]) -> "TriMesh":
        """
        load the mesh from a .obj file.
        """
        if isinstance(f, str):
            with open(f,'rb') as reader:
                return cls.load_from_obj(reader)
        else:
            verts = []
            faces = []
            normals = []
            faces_normal = []

            a = 0
            for line in f:
                line = line.decode('utf-8')
                a = a+1
                # print(a)
                parts = line.strip().split()

                if not parts:
                    continue
                
                if parts[0] == 'v':
                    # temp.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'f':
                    vert1 = parts[1].split('/')
                    vert2 = parts[2].split('/')
                    vert3 = parts[3].split('/')
                    faces.append([int(vert1[0]),int(vert2[0]),int(vert3[0])])
                    # if len(vert1) > 1:
                    #     faces_normal.append([int(vert1[1]),int(vert2[1]),int(vert3[1])])

                elif parts[0] == 'vn':
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            return cls(
                verts=np.array(verts),
                faces=np.array(faces),
                normals=np.array(normals) if len(normals)>0 else None,
                faces_normal=np.array(faces_normal) if len(faces_normal)>0 else None,
                vertex_channels=None,
                face_channels=None,
            )



    def save(self, f: Union[str, BinaryIO]):
        """
        Save the mesh to a .npz file.
        """
        if isinstance(f, str):
            with open(f, "wb") as writer:
                self.save(writer)
        else:
            obj_dict = dict(verts=self.verts, faces=self.faces)
            if self.normals is not None:
                obj_dict["normals"] = self.normals
            for k, v in self.vertex_channels.items():
                obj_dict[f"v_{k}"] = v
            for k, v in self.face_channels.items():
                obj_dict[f"f_{k}"] = v
            np.savez(f, **obj_dict)

    def has_vertex_colors(self) -> bool:
        return self.vertex_channels is not None and all(x in self.vertex_channels for x in "RGB")

    def write_ply(self, raw_f: BinaryIO):
        write_ply(
            raw_f,
            coords=self.verts,
            rgb=(
                np.stack([self.vertex_channels[x] for x in "RGB"], axis=1)
                if self.has_vertex_colors()
                else None
            ),
            faces=self.faces,
        )

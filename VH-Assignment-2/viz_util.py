"""
some util functions for visualization

Author: Xianghui Xie, December 5, 2023
"""
import numpy as np
import trimesh

# Openpose joints colors
OP_COLORS = np.array([
        255.,     0.,    85.,
        255.,     0.,     0.,
        255.,    85.,     0.,
        255.,   170.,     0.,
        255.,   255.,     0.,
        170.,   255.,     0.,
         85.,   255.,     0.,
          0.,   255.,     0.,
        255.,     0.,     0.,
          0.,   255.,    85.,
          0.,   255.,   170.,
          0.,   255.,   255.,
          0.,   170.,   255.,
          0.,    85.,   255.,
          0.,     0.,   255.,
        255.,     0.,   170.,
        170.,     0.,   255.,
        255.,     0.,   255.,
         85.,     0.,   255.,
          0.,     0.,   255.,
          0.,     0.,   255.,
          0.,     0.,   255.,
          0.,   255.,   255.,
          0.,   255.,   255.,
          0.,   255.,   255.
]).reshape(-1, 3)


# map from SMPL joint index to openpose joint index, to get colors
# XXXXXXXXXXXXX=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
SMPL2OPENPOSE=[8, 12,9, 8, 13,10,1, 14,11,1, 19, 22, 0,  5,  2,  15, 5,  2,  6,  3,  7,  4,  7,  4]
SMPL_JOINTS_COLORS = [OP_COLORS[x] for x in SMPL2OPENPOSE]
# SMPL_BONES=np.array([0,1, 0,2, 0,3, 3,6, 6,9, 9,14, 9,13, 9,12, 12,15, 14,17, 17,19, 19,21, 21,23, 13,16, 16,18,
#             18,20, 20,22, 1,4, 4,7, 7,10, 5,8, 8,11]).reshape((-1, 2))
SMPL_BONES=np.array([[         0,          1],
       [         0,          2],
       [         0,          3],
       [         1,          4],
       [         2,          5],
       [         3,          6],
       [         4,          7],
       [         5,          8],
       [         6,          9],
       [         7,         10],
       [         8,         11],
       [         9,         12],
       [         9,         13],
       [         9,         14],
       [        12,         15],
       [        13,         16],
       [        14,         17],
       [        16,         18],
       [        17,         19],
       [        18,         20],
       [        19,         21],
       [        20,         22],
       [        21,         23]])
COLOR_LIST3 = np.array([
    [0.65098039, 0.74117647, 0.85882353],  # grey
    [251 / 255.0, 128 / 255.0, 114 / 255.0],  # orange
    [23/255., 190/255., 207/255.] # blue
])

class Joints3DVisualizer:
    def __init__(self, radius=0.015):
        """

        :param radius: the radius of the sphere for visualize joints
        """
        self.radius = radius

    def j3d_to_skeleton(self, j3d)->trimesh.Trimesh:
        """
        convert joints to spheres + bones as a single mesh
        :param j3d: (N, 3)
        :param bcolor: the color for each bone, if not given, use openpose colors
        :return: mesh
        """
        joint_spheres = self.j3d_to_mesh(j3d, self.radius)
        bones = self.j3d_to_bones(j3d, SMPL_BONES, self.radius/2.)
        mesh = trimesh.util.concatenate(joint_spheres, bones)
        return mesh

    def j3d_to_mesh(self, j3d, radius=0.015, jcolor=None)->trimesh.Trimesh:
        """
        convert a list of 3d joints to one single mesh, each joint is visualized as a sphere
        :param j3d: (N, 3)
        :param radius: the radius of the sphere
        :return: trimesh.Trimesh object
        """
        verts, faces, verts_colors = [], [], []
        offset = 0
        for idx, j in enumerate(j3d):
            sphere:trimesh.Trimesh = trimesh.primitives.Sphere(subdivisions=2, radius=radius, center=j).to_mesh()
            if jcolor is None:
                color = SMPL_JOINTS_COLORS[idx]
            else:
                color = jcolor
            vc = color[None].repeat(len(sphere.vertices), 0)
            verts.append(sphere.vertices)
            faces.append(sphere.faces + offset)
            verts_colors.append(vc)
            offset += len(sphere.vertices)
        mesh = trimesh.Trimesh(np.concatenate(verts, 0),
                               np.concatenate(faces, 0),
                               vertex_colors=np.concatenate(verts_colors, 0))
        return mesh

    def j3d_to_bones(self, j3d, bones, radius=0.01, bcolors=None):
        """
        convert joints into skeleton, each bone is visualized as a cylinder
        :param j3d: (N, 3) coordinates for the joints
        :param bones: (N, 2), index to joints for each bone
        :param bcolors:
        :return:
        """
        verts, faces, vcolors = [], [], []
        offset = 0
        for idx, bone in enumerate(bones):
            start, end = bone
            joint_start, joint_end = j3d[start], j3d[end]
            bone_vector = joint_end - joint_start
            bone_length = np.linalg.norm(bone_vector)
            cylinder = trimesh.primitives.Cylinder(radius=radius, height=bone_length*0.85)

            height = bone_length
            c1, c2 = np.array([0, 0, height / 2], dtype=np.float32), np.array([0, 0, -height / 2], dtype=np.float32)
            R, t, scale = self.estimate_similarity_transform(np.stack([c1, c2], 1), np.stack([joint_start, joint_end], 1))
            cylinder = cylinder.to_mesh()
            cylinder.vertices = np.matmul(cylinder.vertices * scale, R.T) + t.flatten()
            verts.append(cylinder.vertices)
            faces.append(cylinder.faces)
            if bcolors is None:
                color = SMPL_JOINTS_COLORS[end]
            else:
                color = bcolors[end]
            vc = color[None].repeat(len(cylinder.vertices), 0)
            vcolors.append(vc)
            faces.append(cylinder.faces + offset)
            offset += len(cylinder.vertices)
        mesh = trimesh.Trimesh(np.concatenate(verts, 0),
                               np.concatenate(faces, 0),
                               vertex_colors=np.concatenate(vcolors, 0))
        return mesh

    def estimate_similarity_transform(self, S1, S2):
        # 1. Remove mean
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu1

        # 2. Compute variance of source to estimate scale
        var1 = np.sum(X1 ** 2)

        # 3. Outer product of source and target
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale * (R.dot(mu1))

        return R, t, scale



def setup_renderer(N, device, rend_size):
    """
    create pytorch3d mesh renderer and correspondence cameras for visulization
    :param N:
    :param device:
    :param rend_size:
    :return:
    """
    from pytorch3d.renderer import TexturesVertex, look_at_view_transform, PerspectiveCameras, RasterizationSettings, \
        MeshRasterizer, SoftPhongShader, MeshRenderer, PointLights

    Rs, Ts = look_at_view_transform([1.8]*N, [0]*N, [0, 90, 180], device=device)
    focals = np.array([[rend_size*1.5, rend_size*1.5]]).repeat(N, 0)
    image_size = np.array([[rend_size, rend_size]]).repeat(N, 0)
    cameras = PerspectiveCameras(image_size=image_size,
                                 device=device, R=Rs, T=Ts,
                                 focal_length=focals/rend_size,
                                 in_ndc=True)
    raster_settings = RasterizationSettings(
                image_size=rend_size,
                blur_radius=0,
                faces_per_pixel=1,
                clip_barycentric_coords=False,
                max_faces_per_bin=50000
            )
    shader = SoftPhongShader(device=device, lights=PointLights(location=((0, 2.0, 0),),
                                                               specular_color=((0.1, 0.1, 0.1),),
                                                               diffuse_color=((0.5, 0.5, 0.5),),
                                                               device=device))
    renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    raster_settings=raster_settings),
                    shader=shader
            )
    return renderer, cameras


def test():
    ""
    import pickle as pkl
    joints = pkl.load(open('data/scans/125611502399807_J3d.pkl', 'rb'))['joints'] # (45, 3)
    joints_body = joints[:24]
    vizer = Joints3DVisualizer(radius=0.015)
    mesh = vizer.j3d_to_skeleton(joints_body)
    mesh.export('data/test/125611502399807_J3d.ply')

    mesh.show()
    # viewer =

if __name__ == '__main__':
    test()



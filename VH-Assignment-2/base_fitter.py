"""
A basic fitting framework based on torch optimization

Author: Xianghui Xie, September 02, 2023.
"""
import sys, os
import os.path as osp

import matplotlib.pyplot as plt
import torch
import pickle as pkl
import trimesh
from tqdm import tqdm
import numpy as np

from typing import List
from smplx import SMPL


class SMPLFitter:
    def __init__(self, model_root: str, asset_root:str, device='cuda', debug=False):
        """

        :param model_root: root path to SMPL model files
        :param asset_root: root to body prior and other assets
        :param device:
        :param debug:
        """
        self.model_root = model_root
        self.asset_root = asset_root
        self.debug = debug
        self.device = device
        if self.debug:
            # create figure and plot
            import matplotlib.pyplot as plt
            from viz_util import setup_renderer

            # setup renderer
            N, device, rend_size = 3, self.device, 256
            self.renderer, self.cameras = setup_renderer(N, device, rend_size)
            self.rend_size = rend_size
            self.viz_N = N

            plt.ion()
            fig = plt.figure(figsize=(15, 5))
            ax = fig.gca()
            im = ax.imshow(np.random.rand(rend_size, rend_size*N, 3))
            self.plt_im = im
            self.fig, self.ax = fig, ax
            self.plot_freq = 10 # render results after how many iterations

    def fit_scan(self, loss_func, opt_params:dict,
                 scan:trimesh.Trimesh,
                 scan_path: str, gender: str,
                 save_path:str=None, save_name='')->SMPL:
        """

        :param loss_func: a function that takes a dict of data as input and return dict of loss values
            i.e. loss_dict = loss_func(data_dict)
        :param opt_params: a data dict of hyperparameters for the optimization, it should include the following keys:
                iterations: how many iterations for the fitting
                lr: the learning rate for the optimizer
                loss_weights: a dict of loss weights
        :param scan: the scan to be fit, directly loaded from obj file

        :param scan_path: file path to a 3D human scan
        :param gender: gender of the subject
        :param save_path: path where the results will be saved
        :param save_name: additional name to identify this experiment
        :return: the optimized smpl model
        """
        # Load additional data
        j3d = pkl.load(open(scan_path.replace('.obj', '_J3d.pkl'), 'rb'))['joints']

        # Load human body pose prior
        body_prior = pkl.load(open(osp.join(self.asset_root, 'body_prior.pkl'), 'rb'), encoding='latin')

        # Prepare SMPL model
        bs = 1
        global_orient = torch.zeros((bs, 3)).to(self.device)
        body_pose = torch.zeros((bs, 69)).to(self.device)
        transl = torch.zeros((bs, 3)).to(self.device)
        betas = torch.zeros((bs, 10)).to(self.device)
        smpl = SMPL(self.model_root, global_orient=global_orient,
                    body_pose=body_pose, transl=transl, betas=betas, gender=gender)
        smpl = smpl.to(self.device)

        # Prepare optimizer
        optimizer = torch.optim.Adam([smpl.transl,
                                      smpl.global_orient,
                                      smpl.body_pose,
                                      smpl.betas],
                                     lr=opt_params['lr'], betas=(0.9, 0.999))

        data_dict = {
            "smpl":smpl,
            "scan_verts":torch.from_numpy(scan.vertices).float().to(self.device),
            "scan_faces":torch.from_numpy(scan.faces).long().to(self.device),
            "joints": torch.from_numpy(j3d[None]).float().to(self.device),

            # additional term used to compute loss if any
            "body_prior":{
                # mean of the body pose prior distribution (pose_dim, )
                "mean": torch.from_numpy(body_prior['mean']).float().unsqueeze(0).to(self.device), # (1, 63)
                # (pose_dim, pose_dim), cholesky decomposition of the precision matrix for the body pose prior distribution
                "prec_chole": torch.from_numpy(body_prior['precision']).float().to(self.device) # (63, 63)
            }
        }
        loss_weights = opt_params['loss_weights']

        # Optimization loop
        loop = tqdm(range(opt_params['iterations']))
        for it in loop:
            optimizer.zero_grad()
            loss_dict = loss_func(data_dict)
            total_loss = self.sum_losses(loss_dict, loss_weights)
            total_loss.backward()
            optimizer.step()

            # print the losses
            l_str = 'Iter: {}/{}'.format(it+1, opt_params['iterations'])
            for k in loss_dict:
                l_str += ', {}: {:0.4f}'.format(k, (loss_weights[k]*loss_dict[k]).mean().item())
                loop.set_description(l_str)

            # render output if debug
            if self.debug and (it%self.plot_freq == 0 or it == opt_params['iterations']-1):
                from viz_util import COLOR_LIST3
                from pytorch3d.structures import Meshes
                from pytorch3d.renderer import TexturesVertex
                import cv2
                from IPython.display import clear_output
                # render the mesh and show
                with torch.no_grad():
                    verts_smpl = smpl()['vertices'][0].cpu().numpy()
                    faces_smpl = smpl.faces
                verts_scan, faces_scan = scan.vertices, scan.faces
                verts = np.concatenate([verts_smpl, verts_scan], 0)
                faces = np.concatenate([faces_smpl, faces_scan + len(verts_smpl)], 0)
                verts_colors = np.concatenate([COLOR_LIST3[1:2].repeat(len(verts_smpl), 0),
                                               COLOR_LIST3[2:3].repeat(len(verts_scan), 0)])
                tex = TexturesVertex([torch.from_numpy(verts_colors).to(self.device).float()]*self.viz_N)
                meshes = Meshes([torch.from_numpy(verts).to(self.device).float()]*self.viz_N,
                                [torch.from_numpy(faces).to(self.device).long()]*self.viz_N, textures=tex)
                rend = self.renderer(meshes, cameras=self.cameras)
                imgs = [rend[x, ..., :3].cpu().numpy() for x in range(self.viz_N)]
                img = (np.concatenate(imgs, 1)*255).astype(np.uint8)
                cv2.putText(img, f"Optimization it {it+1}/{opt_params['iterations']}", (img.shape[1]//3, 15), cv2.FONT_HERSHEY_PLAIN, 1,
                                (255, 0, 0), 1)
                # self.plt_im.set_data(img)
                # self.fig.canvas.draw()
                # self.fig.canvas.flush_events()
                clear_output(wait=True)
                plt.imshow(img)
                plt.show()
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.pause(0.2)

        # Save output
        scan_name = osp.splitext(osp.basename(scan_path))[0]
        outfile = osp.join(save_path, f'{scan_name}{save_name}_params.pkl')
        os.makedirs(save_path, exist_ok=True)
        with torch.no_grad():
            output = smpl()
        trimesh.Trimesh(output['vertices'][0].cpu().numpy(), smpl.faces, process=False).export(osp.join(save_path, f'{scan_name}{save_name}_smpl.obj'))
        pkl.dump(
            {
                "global_orient":smpl.global_orient[0].detach().cpu().numpy(),
                "body_pose":smpl.body_pose[0].detach().cpu().numpy(),
                "betas":smpl.betas[0].detach().cpu().numpy(),
                "transl":smpl.transl[0].detach().cpu().numpy(),
                "gender":gender
            },
            open(outfile, 'wb')
        )

        return smpl

    @staticmethod
    def sum_losses(loss_dict, weight_dict):
        """

        :param loss_dict: a dict of losses
        :param weight_dict: loss weights
        :return:
        """
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k]*loss_dict[k]

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss





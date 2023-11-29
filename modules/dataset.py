import torch
import json
import numpy as np
from torch.utils.data import Dataset

def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    phis = phis % (2 * np.pi)
    res[(phis >= (2 * np.pi - front / 2)) | (phis < front / 2)] = 0

    res[(phis >= front / 2) & (phis < (np.pi - front / 2))] = 1

    res[(phis >= (np.pi - front / 2)) & (phis < (np.pi + front / 2))] = 2

    # res[(phis >= (np.pi + front))] = 3
    res[(phis >= (np.pi + front / 2)) & (phis < (2 * np.pi - front / 2))] = 3
    # override by thetas
    # res[thetas <= overhead] = 4
    # res[thetas >= (np.pi - overhead)] = 5
    return res

class CameraDataset(Dataset):
    def __init__(self, final_images=None, masks=None, mode='round1', uv_cache=None, device='cuda', json_name=None) -> None:
        super().__init__()
        self.device = device
        if mode == 'round1':
            azims_np = np.arange(5) * (360. / 16.)
            #azims_np = np.array([0, 30, 90, 150, 210, 270, 330])
            elevs_np = 80 * np.ones_like(azims_np)
            #elevs_np = np.array([90, 60, 110, 60, 110, 60, 110])
            radius_np = 1.1 * np.ones_like(azims_np)
            azims_np = np.deg2rad(azims_np)
            elevs_np = np.deg2rad(elevs_np)
            self.azims = torch.from_numpy(azims_np).to(self.device).to(torch.float32)
            self.elevs = torch.from_numpy(elevs_np).to(self.device).to(torch.float32)
            self.radius = torch.from_numpy(radius_np).to(self.device).to(torch.float32)
            self.fov = torch.FloatTensor([np.pi / 3]).to(self.device).to(torch.float32) #degree
        else:
            azims_np = np.arange(5) * (360. / 9.)
            #azims_np = np.append(azims_np, [0, 90, 180, 270])
            elevs_np = 80 * np.ones_like(np.arange(5))
            #elevs_np = np.append(elevs_np, [60, 60, 60, 60])
            radius_np = 1.1 * np.ones_like(azims_np)
            azims_np = np.deg2rad(azims_np)
                # import pdb; pdb./()
            elevs_np = np.deg2rad(elevs_np)
            self.azims = torch.from_numpy(np.array(azims_np)).to(self.device).to(torch.float32)
            self.elevs = torch.from_numpy(np.array(elevs_np)).to(self.device).to(torch.float32)
            self.radius = torch.from_numpy(np.array(radius_np)).to(self.device).to(torch.float32)
            self.fov = torch.FloatTensor([np.pi / 3]).to(self.device).to(torch.float32) #degree
        # import pdb; pdb.set_trace()
        self.dir = get_view_direction(self.elevs, self.azims, overhead=np.pi / 6, front=np.pi / 3)
        self.final_images = final_images if final_images is None else torch.cat(final_images, dim=0) #always in device
        if uv_cache is not None:
            self.uv_features, self.alphas, self.face_idx = uv_cache

        #self.masks = masks if masks is None else torch.cat(masks, dim=0)
    def __len__(self):
        return 1000000

    def get_all_data(self):
        camera_poses = {'elevs': self.elevs, \
            'azims': self.azims, 'radius': self.radius, 'fov': self.fov, 'gt_rgb': self.final_images, 'dir': self.dir}
        return camera_poses
    def __getitem__(self, idx):
        idx = idx % self.azims.shape[0]
        camera_poses = {'elevs': self.elevs[idx][None, ...], \
                        'azims': self.azims[idx][None, ...],\
                        'radius': self.radius[idx][None, ...],\
                        'fov': self.fov[None, ...],\
                        'gt_rgb': self.final_images[idx][None, ...],
                        #'masks': self.masks[idx][None, ...] ,
                        'uv_feaures': self.uv_features[idx][None, ...],
                        'alphas': self.alphas[idx][None, ...],
                        'face_idx': self.face_idx[idx][None, ...],
                        'dir': self.dir[idx][None, ...],
                        }
        return camera_poses
    
    def collect(self, batch):
        datas = {
                'radius': torch.cat(list([item['radius'] for item in batch]), dim=0),
                'elevs': torch.cat(list([item['elevs'] for item in batch]), dim=0),
                'azims': torch.cat(list([item['azims'] for item in batch]), dim=0),
                 #'dir': torch.cat(list([item['dir'] for item in batch]), dim=0),
                 'fov': torch.cat(list([item['fov'] for item in batch]), dim=0),
                 # 'imgs': torch.cat(list([item['img'] for item in batch]), dim=0),
                 # 'masks': torch.cat(list([item['mask'] for item in batch]), dim=0),
                 'dir': torch.cat(list([item['dir'] for item in batch]), dim=0),
                }
        if self.final_images is not None:
            datas['gt_rgb'] = torch.cat(list([item['gt_rgb'] for item in batch]), dim=0)
        if self.uv_features is not None:
            datas.update({
                 'uv_features': torch.cat(list([item['uv_feaures'] for item in batch]), dim=0),
                 'alphas': torch.cat(list([item['alphas'] for item in batch]), dim=0),
                 'face_idx': torch.cat(list([item['face_idx'] for item in batch]), dim=0),
                })
        return datas
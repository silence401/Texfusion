import cv2 
import os
from cv2 import textureFlattening
import torch.nn.functional as F
import torch
import torchvision.transforms as tf
from tqdm import tqdm
from config.config import ModelConfig
import kaolin
import pyrallis
import datetime
import matplotlib.pyplot as plt
from diffusers import ControlNetModel
from modules.dataset import CameraDataset
from modules.mvfusion import MvFusion
from modules.ddim import DDIMScheduler
from modules.mesh.mesh import Mesh, rescale, load_mesh
from modules.mesh.obj import  write_obj
from modules.render import *
from modules.texturefileds import TextureFileds
from modules.vgg import VGGPerceptualLoss

class TexFusion(object):
    """
    TextFusion
    """
    def __init__(self, cfg, device ='cuda'):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.initialize()
    
    def initialize(self):
        depth_control = ControlNetModel.from_pretrained(self.cfg.depth_control_path, torch_dtype=torch.float16).to(self.device)
        self.mvd = MvFusion.from_pretrained(self.cfg.sd_path, \
                controlnet=depth_control, torch_dtype=torch.float16).to(self.device)
        self.vgg_loss = VGGPerceptualLoss().to(self.device)
        self.mvd.scheduler = DDIMScheduler.from_config(self.mvd.scheduler.config)
        self.mvd.scheduler.set_timesteps(50, device=self.device)
        self.timesteps = self.mvd.scheduler.timesteps
        self.texture_fileds = TextureFileds(3, 3).to(self.device)
        self.cache = None 
        self.direction = ['front', 'side', 'back', 'side', 'overhead', 'bottom']
        self.count = 0

    
    def preprocess_mesh(self, mesh):
        #normalize
        mesh = rescale(mesh)
        # UV unwarp 
        if mesh.v_tex is None:
            mesh = self.xatlas_uvmap(mesh)
        self.faces = mesh.t_pos_idx
        self.uv_face_attr = kaolin.ops.mesh.index_vertices_by_faces(
                mesh.v_tex.unsqueeze(0),
                mesh.t_tex_idx).detach().to(self.device)
        self.verts = mesh.v_pos
        return mesh
    
    def xatlas_uvmap(self, mesh):
        import xatlas
        v_pos = mesh.v_pos.detach().cpu().numpy()
        t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy()
        vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)
        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
        uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
        faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')
        new_mesh = Mesh(v_tex=uvs, t_tex_idx=faces, base=mesh)
        return new_mesh


    def reset(self, mesh_path, prompt, mode):
        """
        reset the model
        """
        self.mode = mode
        self.mesh = load_mesh(mesh_path, device=self.device)
        self.mesh = self.preprocess_mesh(self.mesh)
        
        if self.mode == 'latent':
            zT = torch.randn(1, 4,  self.cfg.res[1] // 6, self.cfg.res[0] // 6).to(self.device)
            self.texture_map = torch.nn.parameter.Parameter(zT, requires_grad=False).to(self.device)
            self.texture_map_cur = torch.nn.parameter.Parameter(zT, requires_grad=False).to(self.device)
        else:
            self.texture_map = torch.nn.parameter.Parameter(torch.randn(1, 3, self.resolution, self.resolution).to(self.device), requires_grad=True).to(self.device)
        
        self.dataset = CameraDataset(device=self.device, mode='round1')
        self.camera_poses = self.get_camera_poses()
        self.numviews = self.camera_poses['elevs'].shape[0]
        self.count = 0

        ### 1.first render all depth and cache and quality
        self.render_images(self.texture_map, self.camera_poses, dims=[64, 64])  ## update class perporty

        ### 2. update quality cache
        self.update_Q() ## update quality map


        ### 3. process sd input data
        self.depths = self.mvd.preprocess_control_image(self.depth_512.permute(0, 3, 1, 2)) #permute B H W C ==> B C H W
        self.depths = self.depths.repeat_interleave(3, dim=1)
        #self.prompt_embeds = self.mvd.encode_prompt(prompt) #(prompt_embeds, negative_prompt_embeds)
        self.prompt_embeds = []
        #import pdb; pdb.set_trace()
        for i in range(self.numviews):
            self.prompt_embeds.append(self.mvd.encode_prompt(self.direction[self.camera_poses['dir'][i]] + ' ' + prompt + ' best quality, high quality, extremely detailed, good geometry', "deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke")) 

    def texture(self, mesh_path, prompt, output_path, mode='latent'):
        # import pdb; pdb.set_trace()
        os.makedirs(output_path, exist_ok=True)
        self.reset(mesh_path, prompt, mode)
        os.makedirs(os.path.join(output_path, 'mesh'), exist_ok=True)
        write_obj(os.path.join(output_path, 'mesh'), self.mesh)
        self.output_path = output_path
        def sd_inference(i=0):
            """
            test diffusion inference is right?
            """
            latents = torch.randn((1, 4, 64, 64)).to(self.device)
            for t in tqdm(self.timesteps):
                with torch.no_grad():
                    latents = self.mvd(prompt_embeds=self.mvd.encode_prompt("a road"), control_image=self.depths[i].unsqueeze(0),  latents=latents, t=t, controlnet_conditioning_scale=0.0)
            decode_img = self.mvd.decode_latents(latents, self.prompt_embeds[0].dtype).detach().to(torch.float32)
            
            ### save to debug ###
            # img1 = self.depths[i].to(torch.float32).permute(1, 2, 0).cpu().numpy()
            # img2 = decode_img[0].permute(1, 2, 0).cpu().numpy()
            # cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            # cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            # cv2.imwrite('./decode_depth.png', img1 * 255)
            # cv2.imwrite('./decode_img.png', img2 * 255)
            return decode_img, latents
         # import pdb; pdb.set_trace()

        background_img, ref_latent = sd_inference()
        self.background_latent = self.mvd.encode_images(background_img).to(self.prompt_embeds[0].dtype)
        self.background_noise = torch.randn((1, 4, 64, 64)).to(self.device).to(self.prompt_embeds[0].dtype)

        # import pdb; pdb.set_trace()
        for ts in tqdm(range(len(self.timesteps))):      
            self.interlaced_denoise(ts, tau=0.5)
        
        ###update_camera_pose###
        self.dataset = CameraDataset(device=self.device, mode='round2')
        self.camera_poses = self.get_camera_poses()
        self.cache = None
        self.render_images(self.texture_map, self.camera_poses)
        self.numviews = self.camera_poses['elevs'].shape[0]
        self.texture_map = F.interpolate(self.texture_map, scale_factor=2., mode='nearest')
        self.prompt_embeds = []
        for i in range(self.numviews):
            # import pdb; pdb.set_trace()
            self.prompt_embeds.append(self.mvd.encode_prompt(self.direction[self.camera_poses['dir'][i]] + 'view of ' + prompt + \
                        ' best quality, high quality, extremely detailed, good geometry', "deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke")) 
        
        self.depths = self.mvd.preprocess_control_image(self.depth_512.permute(0, 3, 1, 2)) #permute B H W C ==> B C H W
        self.depths = self.depths.repeat_interleave(3, dim=1)
        self.update_Q() 

        #import pdb; pdb.set_trace()
        for ts in tqdm(range(20, len(self.timesteps))):
            self.interlaced_denoise(ts, tau=0)
        
        self.final_images = []
        with torch.no_grad():
            for i in range(self.numviews):
                view_img = self.render_images(texture_map=self.texture_map, index=i).permute(0, 3, 1, 2)
                view_img = (self.mask[i].to(torch.float) * view_img) + ((1 - self.mask[i].to(torch.float)) * self.background_latent) 
                decode_img = self.mvd.decode_latents(view_img.to(torch.float16)).detach().to(torch.float32)
                self.final_images.append(decode_img)
                tmp_img =  tf.ToPILImage()(self.final_images[i][0]).convert('RGB')
                tmp_img.save(os.path.join(output_path, '{:04d}.png'.format(i)))
        
        self.nerf()

            
    def interlaced_denoise(self, ts=0, tau=0.5):
        """
        interlaced denoise multi views
        tau: "we use
                η = 1, τ = 0.5 in the coarse stage, and η = 1, τ = 0 in the
                high-resolution refinement stage, which we find to be the
                most robust configuration."
        """
        # import pdb; pdb.set_trace()
        for v in range(self.numviews):
            texture_map_noise = torch.randn_like(self.texture_map).to(self.device).to(self.prompt_embeds[0].dtype)
            if (v > 0) and (ts < len(self.timesteps) - 1):
                with torch.no_grad():
                    cur_texture_map = self.mvd.scheduler.add_noise_timesteps(self.texture_map, texture_map_noise, self.timesteps[ts + 1], self.timesteps[ts])
                    cur_texture_map[self.texture_update_mask == 0] = self.texture_map[self.texture_update_mask == 0]
            else:
                cur_texture_map = self.texture_map
            
            with torch.no_grad():
                view_img = self.render_images(texture_map=cur_texture_map, index=v).permute(0, 3, 1, 2)
                view_latent = view_img


                ### 1. denoise
                view_latent = self.mvd(prompt_embeds=self.prompt_embeds[v], control_image=self.depths[v].unsqueeze(0), \
                        latents=view_latent, t=self.timesteps[ts], tau=tau, controlnet_conditioning_scale=1.0)

                ## follow  inpainting pipeline 
                init_latent_proper = self.background_latent
                if ts != len(self.timesteps) - 1:
                    noise_timestep = self.timesteps[ts + 1]
                    init_latent_proper = self.mvd.scheduler.add_noise(
                            self.background_latent, self.background_noise,
                            noise_timestep,
                    )

                view_latent = (self.mask[v].to(torch.float) * view_latent) + ((1 - self.mask[v].to(torch.float)) * init_latent_proper)

            self.update_texture(view_latent, v)
                
    def update_texture(self, img, idx):
        texture_map_bkp = self.texture_map.detach().clone()
        #nvdiffrast cache use
        uv_features = self.cache[idx]

        #correspoding kaolin texture_mapping operation
        uv_features = uv_features * 2 - 1
        uv_features[:, :, 1] = -uv_features[:, :, 1]

        u = torch.round(((uv_features[:, :, 0]+1) * self.texture_map.shape[3]-1)/2).to(torch.int)  #align kaolin.mesh.texture_mapping
        v = torch.round(((uv_features[:, :, 1]+1) * self.texture_map.shape[2]-1)/2).to(torch.int)

        new_uv = torch.stack((v, u), dim=-1) #H, W, 2
        new_uv = new_uv.clamp(0, (self.texture_map.shape[2] - 1))
        view_mask = self.mask[idx].permute(1, 2, 0) # H, W, 1
        new_uv = new_uv[view_mask[:, :, 0] == 1] #N, 2

        """
        #test inverse render conflicts in current resolution
        self.test_overlap = torch.zeros_like(self.texture_map)
        for i in range(new_uv.shape[0]):
            self.test_overlap[0, :1, new_uv[i, 0], new_uv[i, 1]] += 1
        """

        #direct assign value to texture map
        self.texture_map[0, :, new_uv[:, 0], new_uv[:, 1]] = \
                                    img[0].permute(1, 2, 0)[view_mask[..., 0] == 1].permute(1, 0)
        
        #initialize texture_update_mask and texture quality
        if idx == 0:
            self.texture_update_mask = torch.zeros_like(self.texture_map) > 0
            self.texture_update_mask[:, :, new_uv[:, 0], new_uv[:, 1]] = True
            self.cur_Q = self.Q[0]
        else:
            tmp_update_mask = torch.zeros_like(self.texture_map) > 0
            tmp_update_mask[:, :, new_uv[:, 0], new_uv[:, 1]] = True
            tmp_update_mask = tmp_update_mask & (self.Q[idx] > self.cur_Q)
            self.cur_Q = torch.maximum(self.cur_Q, self.Q[idx])
            self.texture_map[~tmp_update_mask] = texture_map_bkp[~tmp_update_mask]
            self.texture_update_mask = self.texture_update_mask | tmp_update_mask
        
        #import pdb; pdb.set_trace()
        texture_map_mask_pil = tf.ToPILImage()(self.texture_update_mask[0][:3, ...].to(torch.float))
        texture_map_mask_pil.save("mask_{:04d}.png".format(self.count))
        self.count = self.count + 1
        #self.texture_map = self.texture_map.clamp(0, 1)

    def update_Q(self):
        """
        update texture quality since texture map resolution will change
        """
        #import pdb; pdb.set_trace()
        self.Q = torch.zeros((self.numviews, 1, self.texture_map.shape[2], self.texture_map.shape[3])).to(self.device)
        for i in range(self.numviews):
            uv_features = self.cache[i]
            uv_features = uv_features * 2 - 1
            uv_features[:, :, 1] = -uv_features[:, :, 1]
            u = torch.round(((uv_features[:, :, 0]+1) * self.texture_map.shape[3]-1)/2).to(torch.int)  #align kaolin.mesh.texture_mapping
            v = torch.round(((uv_features[:, :, 1]+1) * self.texture_map.shape[2]-1)/2).to(torch.int)
            new_uv = torch.stack((v, u), dim=-1) #H, W, 2
            new_uv = new_uv.clamp(0, (self.texture_map.shape[2] - 1))
            mask = self.mask[i].permute(1, 2, 0) # H, W, 1
            # import pdb; pdb.set_trace()
            new_uv = new_uv[mask[:, :, 0] == 1] #N, 2
            self.Q[i, 0, new_uv[:, 0], new_uv[:, 1]] = self.quality[i][mask == 1]


    def render_images(self, texture_map, camera_poses=None, index=None, dims=[64, 64], mode='nearest'):
        # import pdb; pdb.set_trace()
        if self.cache is None:
            # we always need 512x512 resolution cache
            img, mask, cdq = render_single_view_texture(self.verts, self.faces, self.uv_face_attr,
                    self.texture_map, elev=camera_poses['elevs'], azim=camera_poses['azims'], \
                        radius=camera_poses['radius'], fov=camera_poses['fov'], dims=[512, 512], interpolate_mode='nearest',\
                        return_cache=True, return_depth=True, return_derivatives=True)

            self.cache_512 = cdq[0].detach()
            self.depth_512 = cdq[1].detach()
            self.derivatives_512 = cdq[2].detach() #[B, 4, H, W]
            
            self.quality_512 = torch.abs(self.derivatives_512[..., 0] * self.derivatives_512[..., 3] - \
                    self.derivatives_512[..., 1] * self.derivatives_512[..., 2])  #ref texfusion paper #B, 1, H, W
            
            self.quality_512 = self.quality_512.unsqueeze(-1)
            self.mask_512 = mask.detach() 
            def save_quality(quality, mask):
                """
                test quality value
                """
                import matplotlib.pyplot as plt
                for i in range(quality.shape[0]):
                    qua = quality[i]
                    qua = qua - qua.min()
                    mask_ = mask[i, 0]
                    qua[mask_==0] = 0.
                    qua = qua / (qua.max() - qua.min())
                    plt.imsave(os.path.join(self.output_path, 'qua{:03d}.png'.format(i)), qua.cpu().numpy(), cmap='hot')
            _, mask, cdq = render_single_view_texture(self.verts, self.faces, self.uv_face_attr,
                self.texture_map, elev=camera_poses['elevs'], fov=camera_poses['fov'], azim=camera_poses['azims'], \
                    radius=camera_poses['radius'], dims=dims,\
                    return_cache=True, return_depth=True, return_derivatives=True)
            self.cache = cdq[0].detach()
            self.derivatives = cdq[2].detach() #[B, 4, H, W]
            self.quality = torch.abs(self.derivatives[..., 0] * self.derivatives[..., 3] - \
                self.derivatives[..., 1] * self.derivatives[..., 2])  #ref texfusion paper #B, 1, H, W
            self.quality = self.quality.unsqueeze(-1)
            
            print(self.quality.max(), self.quality.min())
            self.mask = mask.detach()

        else:
            img = self.render_from_cache(texture_map, index, mode=mode)
            return img
    
    def render_from_cache(self, texture_map, index, mode='nearest'):
        uv_feature = self.cache[index].unsqueeze(0)
        image_features = kaolin.render.mesh.texture_mapping(uv_feature, texture_map, mode=mode)
        return image_features
    
    def get_camera_poses(self, mode='default'):
        camera_poses = self.dataset.get_all_data()
        return camera_poses
    
    def nerf(self):
        """
        nerf
        """
        uv_cache = renderfunc_mvfusion(self.mesh, dims=[512, 512], data=self.dataset.get_all_data(), device=self.device)
        
        new_dataset = CameraDataset(final_images=self.final_images, masks=self.mask, uv_cache=uv_cache, mode='round2', device=self.device)
        self.train_dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=4, collate_fn=new_dataset.collect, shuffle=False)
        iteration = 0
        optimizer = torch.optim.Adam(self.texture_fileds.parameters(), lr=1e-2)
       
        for data in self.train_dataloader:
            rgbs, alphas = renderfunc_mvfusion(self.mesh, \
                    color_func=self.texture_fileds, data=data, dims=[512, 512], look_at_height=0.0, \
                        uv_cache=(data['uv_features'], data['alphas'], data['face_idx']), device=self.device)
            optimizer.zero_grad()
            view_masks = alphas.detach()
            #gt_rgbs = F.interpolate(data['gt_rgb'], (1024, 1024), mode='bilinear', align_corners=True)
            loss1 = F.mse_loss(rgbs * view_masks, data['gt_rgb'] * view_masks)
            #loss2 = self.vgg_loss(rgbs * view_masks, data['gt_rgb'] * view_masks) * 0.05
            loss = loss1 # + loss2
            loss.backward()
            optimizer.step()
            print("loss:", loss1) # loss2)
            iteration = iteration + 1
            if iteration > 550:
                break
        #plt.imsave('rgbs1.png', rgbs[0].permute(1, 2, 0).detach().cpu().numpy())
        self.extrct_texture()
    
    def extrct_texture(self, resolution=[1024, 1024]):
        """
        """
        import nvdiffrast.torch as dr
         # clip space transform 
        ctx = dr.RasterizeGLContext()
        
        uv_clip = self.mesh.v_tex[None, ...] * 2.0 - 1.0

        # pad to four component coordinate
        uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

        # rasterize
        rast, _ = dr.rasterize(ctx, uv_clip4, self.mesh.t_tex_idx.int(), resolution)

        # Interpolate world space position
        def interpolate(attr, rast, attr_idx, rast_db=None):
            return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')
        gb_pos, _ = interpolate(self.mesh.v_pos[None, ...], rast, self.mesh.t_pos_idx.int())

        # Sample out textures from MLP
        with torch.no_grad():
            all_tex = self.texture_fileds((gb_pos.reshape(-1, 3) + 1) / 2.)
            all_tex = all_tex['color'].reshape((resolution[0], resolution[1], 3))
        self.texture_map = all_tex.permute(2, 0, 1).unsqueeze(0)
        plt.imsave(os.path.join(self.output_path, 'mesh', 'tex_final.png'), self.texture_map[0].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy())
@pyrallis.wrap()
def main(cfg: ModelConfig):
    texfusion = TexFusion(cfg, device='cuda')
    # import pdb; pdb.set_trace()
    mesh_path = '/workspace/code/baidu/ar/neural_engine/algorithms/text23D/data/nascar.obj'
    prompt = 'yellow car'
    output_path = os.path.join('outputs', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    texfusion.texture(mesh_path, prompt, output_path)
    
if __name__ == '__main__':
    main()
    

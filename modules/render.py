import torch 
import kaolin
import numpy as np 
def renderfunc_mvfusion(mesh, dims, data, color_func=None, look_at_height=0.0, uv_cache=None, device='cuda'):
    
    elev = data['elevs']
    azim = data['azims']
    fov = data['fov']
    radius = data['radius']
    camera_transform = get_camera_from_view(elev, azim, r=radius,
            look_at_height=look_at_height).to(device)
    camera_projection = kaolin.render.camera.generate_perspective_projection(fov[0].cpu()).to(device)
     
    if uv_cache is None:
        face_vertices_camera, face_vertices_image, face_normals = kaolin.render.mesh.prepare_vertices(
            mesh.v_pos, mesh.t_pos_idx, camera_projection, camera_transform=camera_transform)
        
        face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
            mesh.v_pos.unsqueeze(0),
            mesh.t_pos_idx)

        with torch.no_grad():
            uv_features, face_idx, rast, pos, tri = kaolin.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1], 
            face_vertices_image, face_attributes, backend='nvdiffrast')
            mask = (face_idx > -1).float()[..., None]
            alphas = (mask > 0).detach()
            uv_cache = (uv_features, alphas, face_idx)
        return uv_cache
    else:
        uv_features, alphas, face_idx = uv_cache
        uv_features = uv_features.view(face_idx.shape[0], -1, 3)
        rgbs = torch.zeros(uv_features.shape[0], uv_features.shape[1], 3, device=device, dtype=torch.float) 
        with torch.cuda.amp.autocast(enabled=False):
            mask_flatten = (alphas > 0).view(face_idx.shape[0], -1)
            for i in range(face_idx.shape[0]):
                mask_image_features = color_func((uv_features[i][mask_flatten[i]].detach() + 1.) / 2)['color'].to(torch.float)
                rgbs[i][mask_flatten[i]] = mask_image_features
        rgbs = rgbs.view(face_idx.shape[0], dims[0], dims[1], 3)
        return  rgbs.permute(0, 3, 1, 2), alphas.permute(0, 3, 1, 2)


def render_single_view_texture(verts, faces, uv_face_attr, texture_map, elev=0, azim=0, radius=2, fov=np.pi/3,
                                   look_at_height=0.0, dims=None, white_background=False, device='cuda', \
                                    interpolate_mode='bilinear', return_cache=False, return_depth=False, return_derivatives=False, return_normal=True):


    camera_transform = get_camera_from_view(elev, azim, r=radius,
            look_at_height=look_at_height).to(device)
    
    camera_projection = kaolin.render.camera.generate_perspective_projection(fov[0].cpu().numpy()).to(device)
    face_vertices_camera, face_vertices_image, face_normals = kaolin.render.mesh.prepare_vertices(
            verts.to(device), faces.to(device), camera_projection, camera_transform=camera_transform)
    uv_face_attr = uv_face_attr.expand([face_vertices_camera.shape[0], -1, -1, -1])

    uv_features, face_idx, rast, pos, tri = kaolin.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1], face_vertices_image, 
            uv_face_attr.contiguous(), backend='nvdiffrast', return_derivatives=return_derivatives)
    
    # import pdb; pdb.set_trace()
    uv_features = uv_features.detach()
    texture_map = texture_map.expand([face_vertices_camera.shape[0], -1, -1, -1])
    mask = (face_idx > -1)[..., None]
    #mask = soft_mask.unsqueeze(-1)
    image_features = kaolin.render.mesh.texture_mapping(uv_features, texture_map, mode=interpolate_mode)
    image_features = image_features * mask
    if white_background:
        image_features += 1 * (1 - mask)
        
    cdq = [] #cache depth quality
    if return_cache:
        cdq.append(uv_features)
    if return_depth:
        depth = rast[0][..., None, 2]
        cdq.append(depth)
    if return_derivatives:
        cdq.append(rast[1])
    # if return_normal:
    #     cdq.append()

    return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), cdq

def get_camera_from_view(elev, azim, r=3.0, look_at_height=0.0, it=0, use_origin_pose=True):
    x = r * torch.sin(elev) * torch.sin(azim)
    y = r * torch.cos(elev)
    z = r * torch.sin(elev) * torch.cos(azim)
    pos = torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), dim=1)
   
    look_at = torch.zeros_like(pos)
    look_at[:, 1] = look_at_height
   
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).cuda().expand_as(pos)
    look_at = look_at.expand_as(pos)
    camera_proj = kaolin.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_proj
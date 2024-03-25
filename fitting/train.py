import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from img_utils.display_smpl import display_model

def init(smpl_layer, target, device, cfg):
    params = {}
    params["pose_params"] = torch.zeros(target.shape[0], 72)
    params["shape_params"] = torch.zeros(target.shape[0], 10)
    params["scale"] = torch.ones([1])

    smpl_layer = smpl_layer.to(device)
    params["pose_params"] = params["pose_params"].to(device)
    params["shape_params"] = params["shape_params"].to(device)
    target = target.to(device)
    params["scale"] = params["scale"].to(device)

    params["pose_params"].requires_grad = True
    params["shape_params"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SHAPE)
    params["scale"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SCALE)

    optim_params = [{'params': params["pose_params"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    {'params': params["shape_params"], 'lr': cfg.TRAIN.LEARNING_RATE},
                    {'params': params["scale"], 'lr': cfg.TRAIN.LEARNING_RATE*10},]
    optimizer = optim.Adam(optim_params)

    index = {}
    smpl_index = []
    dataset_index = []
    for tp in cfg.DATASET.DATA_MAP:
        smpl_index.append(tp[0])
        dataset_index.append(tp[1])

    index["smpl_index"] = torch.tensor(smpl_index).to(device)
    index["dataset_index"] = torch.tensor(dataset_index).to(device)

    return smpl_layer, params, target, optimizer, index

def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_single_pic(res, smpl_layer, epoch, logger, dataset_name, target):
    _, _, verts, Jtr = res
    fit_path = "output/{}/picture".format(dataset_name)
    create_dir_not_exist(fit_path)
    logger.info('Saving pictures at {}'.format(fit_path))
    display_model(
        {'verts': verts.cpu().detach(),
            'joints': Jtr.cpu().detach()},
        model_faces=smpl_layer.th_faces,
        with_joints=True,
        kintree_table=smpl_layer.kintree_table,
        savepath=fit_path+"/epoch_{:0>4d}".format(epoch),
        batch_idx=60,
        show=False,
        only_joint=False)
    logger.info('Picture saved')

def train(smpl_layer, target,
          logger, writer, device,
          args, cfg, meters):
    res = []
    smpl_layer, params, target, optimizer, index = \
        init(smpl_layer, target, device, cfg)
    pose_params = params["pose_params"]
    shape_params = params["shape_params"]
    scale = params["scale"]
    
    with torch.no_grad():
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        params["scale"]*=(torch.max(torch.abs(target))/torch.max(torch.abs(Jtr)))

    for epoch in tqdm(range(cfg.TRAIN.MAX_EPOCH)):
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        loss = F.smooth_l1_loss(scale*Jtr.index_select(1, index["smpl_index"]),
                                target.index_select(1, index["dataset_index"]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meters.update_early_stop(float(loss))
        if meters.update_res:
            res = [pose_params, shape_params, verts, Jtr]
        if meters.early_stop:
            logger.info("Early stop at epoch {} !".format(epoch))
            break

        if epoch % cfg.TRAIN.WRITE == 0 or epoch<10:
            print("Epoch {}, lossPerBatch={:.6f}, scale={:.4f}".format(
                     epoch, float(loss),float(scale)))
            writer.add_scalar('loss', float(loss), epoch)
            writer.add_scalar('learning_rate', float(
                optimizer.state_dict()['param_groups'][0]['lr']), epoch)
            # save_single_pic(res,smpl_layer,epoch,logger,args.dataset_name,target)

    logger.info('Train ended, min_loss = {:.4f}'.format(
        float(meters.min_loss)))
    return res

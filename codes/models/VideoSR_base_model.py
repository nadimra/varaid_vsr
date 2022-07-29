import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import CharbonnierLoss, LapLoss
import torchvision.transforms
from models.modules.segmentation.mit_semseg.utils import colorEncode
import numpy
import scipy.io

logger = logging.getLogger('base')


class VideoSRBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoSRBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)

        """
        Define segmentation network
        """
        self.netS = networks.define_S().to(self.device)

        if opt['dist']:
            self.netG = DistributedDataParallel(
                self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        """
        Load segementation network
        """
        self.colors = scipy.io.loadmat('models/modules/segmentation/data/color150.mat')['colors']

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
                self.seg_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
                self.seg_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
                self.seg_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'lp':
                self.cri_pix = LapLoss(max_levels=5).to(self.device)
                self.seg_pix = LapLoss(max_levels=5).to(self.device)
            else:
                raise NotImplementedError(
                    'Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning(
                            'Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)
            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)

        #print("\nBase Model VarL:")
        #print(self.var_L.size())

        if need_GT:
            self.real_H = data['GT'].to(self.device)
            #print("\nBase Model real_H:")
            #print(self.real_H.size())
            """
            set real Segmenation by calling segmentation module with HR image
            """

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        """
        set fake_S by calling segmentation network on LR image
        """
        imgList = []
        batchList = []
        B, N, C, H, W = self.fake_H.size() 
        imgPred1= torch.squeeze(self.fake_H, 3)
        print(imgPred1.size())
        imgPred2= torch.squeeze(self.real_H, 3)


        res1 = self.predict_segmentation(imgPred1,False)
        res2 = self.predict_segmentation(imgPred2,False)

        """
        for b in range(B):
            imgList = []
            for idx in range(N):
                imgPred = self.predict_segmentation(self.fake_H[b, idx, :, :, :],False)
                imgList.append(imgPred)
            imgList = torch.stack(imgList)
            batchList.append(imgList)
        self.fake_S = torch.stack(batchList)
        """

        #print("\nFake S:")
        #print(self.fake_S.size())

        """
        imgList = []
        batchList = []
        B, N, C, H, W = self.real_H.size() 
        print("Real batch size: {}".format(B))
        for b in range(B):
            imgList = []
            for idx in range(N):
                imgPred = self.predict_segmentation(self.real_H[b, idx, :, :, :],True)
                imgList.append(imgPred)
            imgList = torch.stack(imgList)
            batchList.append(imgList)
        self.real_S = torch.stack(batchList)

        """
        #print("\nReal S:")
        #print(self.real_S.size())

        alpha = 0.3
        l_pix = self.l_pix_w * (1-alpha)*self.cri_pix(self.fake_H, self.real_H) #+ (alpha)*self.seg_pix(self.fake_S, self.real_S)

        """
        add extra information to loss function to do with fake_S-real_S
        """

        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['restore'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG,
                              self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def predict_segmentation(self,img_data,real=False):
        # Load and normalize one image as a singleton tensor batch
        # pil_image = PIL.Image.open('ADE_val_00001519.jpg').convert('RGB')
        # img_original = numpy.array(pil_image)
        # img_data = pil_to_tensor(pil_image)
        
        transform_norm = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
                std=[0.229, 0.224, 0.225])  # across a large photo dataset.
        ])
        img_data =transform_norm(img_data)
        singleton_batch = {'img_data': img_data[None].cuda()}
        output_size = img_data.shape[1:]

        # Run the segmentation at the highest resolution.
        with torch.no_grad():
            scores = self.netS(singleton_batch, segSize=output_size)
            
        # Get the predicted scores for each pixel
        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()

        # Filter prediction based on person class
        index = 12
        if index is not None:
            pred = pred.copy()
            pred[pred != index] = -1
            
        # colorize prediction
        pred_color = colorEncode(pred, self.colors).astype(numpy.uint8)
        pred_color = torch.from_numpy(pred_color)
        return pred_color

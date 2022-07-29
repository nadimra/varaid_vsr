import models.modules.Sakuya_arch as Sakuya_arch
import models.modules.segmentation.mit_semseg.models.models as SegModels
import torch

####################
# define network
####################
# Generator


def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'LunaTokis':
        netG = Sakuya_arch.LunaTokis(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                     groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                     back_RBs=opt_net['back_RBs'])
    else:
        raise NotImplementedError(
            'Generator model [{:s}] not recognized'.format(which_model))

    return netG


def define_S():
    # Network Builders
    net_encoder = SegModels.ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights='models/modules/segmentation/ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
    net_decoder = SegModels.ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='models/modules/segmentation/ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    netS = SegModels.SegmentationModule(net_encoder, net_decoder, crit)
    netS.eval()
    netS.cuda()
    return netS
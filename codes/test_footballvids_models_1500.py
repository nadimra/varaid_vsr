'''
test Zooming Slow-Mo models on arbitrary datasets
write to txt log file
[kosame] TODO: update the test script to the newest version
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import data.util as data_util
import models.modules.Sakuya_arch as Sakuya_arch
import argparse

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--modelNum', type=int, default=1)
    parser.add_argument('--modelName', type=str, default="ModelA")
    parser.add_argument('--frameClip', type=int, default=5)
    parser.add_argument('--datasetFolder', type=str, default='/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LR/')
    parser.add_argument('--saveImgs', default=False, action='store_true')
    args = parser.parse_args()

    model_name = args.modelName
    model_num = args.modelNum*1500 #1500 itr per epoch
    scale = args.scale
    test_dataset_folder = args.datasetFolder
    N_ot = args.frameClip #3
    save_imgs = args.saveImgs
    N_in = 1+ N_ot // 2
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #### model 
    #### TODO: change your model path here

    model_path = '../experiments/{}/models/{}_G.pth'.format(model_name,model_num)
    model = Sakuya_arch.LunaTokis(64, N_ot, 8, 5, 40)
    
    #### dataset
    data_mode = 'FootballVids' 

    if data_mode == 'FootballVids':
        if save_imgs:
            test_txt = '/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/footballvids_img_test.txt'
        else:
            test_txt = '/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/footballvids_testlist.txt'
        f = open(test_txt, "r")
        guide = f.read().splitlines()

    #### evaluation
    flip_test = False #True#
    crop_border = 0

    # temporal padding mode
    padding = 'replicate'
    
    ############################################################################
    if torch.cuda.is_available():
        device = torch.device('cuda') 
    else:
        device = torch.device('cpu')
    if save_imgs:
        save_folder = '../results/{}/imgs/{}'.format(model_name,model_num)
    else:
        save_folder = '../results/{}/{}'.format(model_name,model_num)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    model_params = util.get_model_total_params(model)

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Model parameters: {} M'.format(model_params))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip Test: {}'.format(flip_test))
   

    def single_forward(model, imgs_in):
        with torch.no_grad():
            # imgs_in.size(): [1,n,3,h,w]
            b,n,c,h,w = imgs_in.size()
            h_n = int(4*np.ceil(h/4))
            w_n = int(4*np.ceil(w/4))
            imgs_temp = imgs_in.new_zeros(b,n,c,h_n,w_n)
            imgs_temp[:,:,:,0:h,0:w] = imgs_in

            model_output = model(imgs_temp)
            # model_output.size(): torch.Size([1, 3, 4h, 4w])
            model_output = model_output[:, :, :, 0:scale*h, 0:scale*w]
            if isinstance(model_output, list) or isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output
        return output
    
    model.load_state_dict(torch.load(model_path), strict=True)

    model.eval()
    model = model.to(device)

    avg_psnr_l = []
    avg_psnr_y_l = []
    sub_folder_name_l = []
    avg_ssim_l = []

    #total_time = []
    # for each sub-folder
    matches_l = [f for f in sorted(os.listdir(test_dataset_folder))]
    for match in matches_l:
        clip_path = test_dataset_folder+match+'/'
        clips_l = [f for f in sorted(os.listdir(clip_path))]
        for clip in clips_l:
            sector_path = clip_path +clip+'/'
            sector_l = [f for f in sorted(os.listdir(sector_path))]
            for sector in sector_l:
                gt_tested_list = []
                concat_folder = match+'/'+clip+'/'+sector
                sub_folder_name = sector_path +sector
                #print(concat_folder, guide,'\n')
                if concat_folder in guide:
                    #print(sub_folder_name)
                
                    sub_folder_name_l.append(sub_folder_name)
                    save_sub_folder = osp.join(save_folder,concat_folder)

                    # Listg of all LR images
                    img_LR_l = sorted(glob.glob(sub_folder_name + '/*'))

                    if save_imgs:
                        util.mkdirs(save_sub_folder)

                    #### read LR images
                    imgs = util.read_seq_imgs(sub_folder_name)
                    
                    #### read GT images
                    img_GT_l = []
                    sub_folder_GT = osp.join(sub_folder_name.replace('/LR/', '/HR/'))

                    for img_GT_path in sorted(glob.glob(osp.join(sub_folder_GT,'*'))):
                        img_GT_l.append(util.read_image(img_GT_path))
                    
                    avg_psnr, avg_psnr_sum, cal_n = 0,0,0
                    avg_psnr_y, avg_psnr_sum_y = 0,0
                    avg_ssim, avg_ssim_sum = 0,0

                    if len(img_LR_l) == len(img_GT_l):
                        skip = True
                    else:
                        skip = False
                    
                    select_idx_list = util.test_index_generation(skip, N_ot, len(img_LR_l))

                    
                    # process each image
                    for select_idxs in select_idx_list:
                        # get input images
                        select_idx = select_idxs[0]
                        gt_idx = select_idxs[1]
                        imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

                        output = single_forward(model, imgs_in)

                        outputs = output.data.float().cpu().squeeze(0)            

                        if flip_test:
                            # flip W
                            output = single_forward(model, torch.flip(imgs_in, (-1, )))
                            output = torch.flip(output, (-1, ))
                            output = output.data.float().cpu().squeeze(0)
                            outputs = outputs + output
                            # flip H
                            output = single_forward(model, torch.flip(imgs_in, (-2, )))
                            output = torch.flip(output, (-2, ))
                            output = output.data.float().cpu().squeeze(0)
                            outputs = outputs + output
                            # flip both H and W
                            output = single_forward(model, torch.flip(imgs_in, (-2, -1)))
                            output = torch.flip(output, (-2, -1))
                            output = output.data.float().cpu().squeeze(0)
                            outputs = outputs + output

                            outputs = outputs / 4

                        # save imgs
                        for idx, name_idx in enumerate(gt_idx):
                            if name_idx in gt_tested_list:
                                continue
                            gt_tested_list.append(name_idx)
                            output_f = outputs[idx,:,:,:].squeeze(0)

                            output = util.tensor2img(output_f)
                            if save_imgs:                
                                cv2.imwrite(osp.join(save_sub_folder, '{:04d}.png'.format(name_idx+1)), output)

                            #if 'Custom' not in data_mode:
                                
                            #### calculate PSNR
                            output = output / 255.

                            #if 'Vimeo_fast' in data_mode:
                            #    GT = np.copy(img_GT_l[0])
                            #else:
                            GT = np.copy(img_GT_l[name_idx])

                            
                            if crop_border ==0:
                                cropped_output = output
                                cropped_GT = GT
                                if output.shape != GT.shape:
                                    if cropped_GT.shape[0] > cropped_output.shape[0] or cropped_GT.shape[1]>cropped_output.shape[1]:
                                        cropped_GT = cropped_GT[0:cropped_output.shape[0],0:cropped_output.shape[1]]

                                    else:    
                                        cropped_output = cropped_output[0:cropped_GT.shape[0],0:cropped_GT.shape[1]]
                                    #print("GT: {} Output: {}".format(cropped_GT.shape,cropped_output.shape))
                            else:
                                cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
                                cropped_GT = GT[crop_border:-crop_border, crop_border:-crop_border, :]
                                
                            crt_psnr = util.calculate_psnr(cropped_output * 255, cropped_GT * 255)
                            crt_ssim = util.calculate_ssim(cropped_output * 255, cropped_GT * 255)
                            cropped_GT_y = data_util.bgr2ycbcr(cropped_GT, only_y=True)
                            cropped_output_y = data_util.bgr2ycbcr(cropped_output, only_y=True)
                            crt_psnr_y = util.calculate_psnr(cropped_output_y * 255, cropped_GT_y * 255)
                            logger.info('{:3d} - {:25}.png \tPSNR: {:.6f} dB  PSNR-Y: {:.6f} dB'.format(name_idx + 1, name_idx+1, crt_psnr, crt_psnr_y))
                            avg_psnr_sum += crt_psnr
                            avg_psnr_sum_y += crt_psnr_y
                            avg_ssim_sum += crt_ssim
                            cal_n += 1
                                

                    avg_psnr = avg_psnr_sum / cal_n
                    avg_psnr_y = avg_psnr_sum_y / cal_n
                    avg_ssim = avg_ssim_sum / cal_n
                    logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} SSIM: {:.6f}dB for {} frames; '.format(sub_folder_name, avg_psnr, avg_psnr_y, avg_ssim, cal_n))
            
                    avg_psnr_l.append(avg_psnr)
                    avg_psnr_y_l.append(avg_psnr_y)
                    avg_ssim_l.append(avg_ssim)

                    
            
    ############################################################################
    # Print tidy outputs after testing
    logger.info('################ Tidy Outputs ################')
    for name, psnr, psnr_y in zip(sub_folder_name_l, avg_psnr_l, avg_psnr_y_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB. '
                    .format(name, psnr, psnr_y))
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip Test: {}'.format(flip_test))
    
    logger.info('Total Average PSNR: {:.6f} dB PSNR-Y: {:.6f} dB SSIM: {:.6f}'
                .format(
                    sum(avg_psnr_l) / len(avg_psnr_l), sum(avg_psnr_y_l) / len(avg_psnr_y_l), sum(avg_ssim_l) / len(avg_ssim_l)))
    
    #logger.info('Total Runtime: {:.6f} s Average Runtime: {:.6f} for {} images.'.format(sum(total_time), sum(total_time)/171, 171))

if __name__ == '__main__':
    main()

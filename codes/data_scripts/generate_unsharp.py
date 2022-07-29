import os
import sys
import cv2
import numpy as np
import os.path as osp
import glob

try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    import utils.util as utils
except ImportError:
    pass


def generate_unsharp(savedir):
    # params: upscale factor, input directory, output directory
    saveHRpath = os.path.join(savedir, 'HR')
    matches_l = [f for f in sorted(os.listdir(saveHRpath))]
    for match in matches_l:
        clip_path = saveHRpath+'/'+match+'/'
        clips_l = [f for f in sorted(os.listdir(clip_path))]
        for clip in clips_l:
            sector_path = clip_path +clip+'/'
            sector_l = [f for f in sorted(os.listdir(sector_path))]
            for sector in sector_l:
                sub_folder_name = sector_path +sector
                sub_folder_laplacian = osp.join(sub_folder_name.replace('/HR/', '/HR_Unsharp/'))
                utils.mkdirs(sub_folder_laplacian)

                for img_HR_path in sorted(os.listdir(sub_folder_name)):
                    imgSave = sub_folder_name+'/'+img_HR_path
                    im = cv2.imread(imgSave)
                    #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    #im = cv2.filter2D(im, -1, kernel)
                    gaussian = cv2.GaussianBlur(im, (0, 0), 2.0)
                    im = cv2.addWeighted(im, 2.0, gaussian, -1.0, 0)
                    
                    sub_folder_laplacian_img = osp.join(imgSave.replace('/HR/', '/HR_Unsharp/'))
                    #print(sub_folder_laplacian_img)
                    cv2.imwrite(sub_folder_laplacian_img, im)


if __name__ == "__main__":
    generate_unsharp('/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/')
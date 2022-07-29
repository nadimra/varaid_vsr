'''create lmdb files for Vimeo90K-7 frames training dataset (multiprocessing)
Will read all the images to the memory
'''

import os,sys
import os.path as osp
import glob
import pickle
from multiprocessing import Pool
import numpy as np
import lmdb
import cv2
try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    import data.util as data_util
    import utils.util as util
except ImportError:
    pass


def reading_image_worker(path, key):
    '''worker for reading images'''
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)

def playervids():
    '''create lmdb for the playervids-7 frames dataset, each image with fixed size
    GT: [3, 256, 448]
        Only need the 4th frame currently, e.g., 00001_0001_4
    LR: [3, 64, 112]
        With 1st - 7th frames, e.g., 00001_0001_1, ..., 00001_0001_7
    key:
        Use the folder and subfolder names, w/o the frame index, e.g., 00001_0001
    '''
    #### configurations
    mode = 'HR'  # GT | LR
    batch = 3000 # TODO: depending on your mem size
    if mode == 'HR':
        img_folder = '/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/PlayerVids/HR'
        lmdb_save_path = '../data_scripts/playervids_train_HR.lmdb'
        txt_file = '/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/PlayerVids/playervids_trainlist.txt'
        H_dst, W_dst = 140, 279
    elif mode == 'LR':
        img_folder = '/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/PlayerVids/LR'
        lmdb_save_path = '../data_scripts/playervids_train_LR.lmdb'
        txt_file = '/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/PlayerVids/playervids_trainlist.txt'
        H_dst, W_dst = 35, 69
    n_thread = 40
    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    #### whether the lmdb file exist
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]
    all_img_list = []
    keys = []
    for line in train_l:
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]

        file_l = glob.glob(osp.join(img_folder, folder, sub_folder) + '/*')
        all_img_list.extend(file_l)
        for j in range(7):
            keys.append('{}_{}_{}'.format(folder, sub_folder, j + 1))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)
    if mode == 'HR': 
        all_img_list = [v for v in all_img_list if v.endswith('.png')]
        keys = [v for v in keys]
    print('Calculating the total size of images...')
    data_size = sum(os.stat(v).st_size for v in all_img_list)

    #### read all images to memory (multiprocessing)
    print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
    
    #### create lmdb environment
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    txn = env.begin(write=True)  # txn is a Transaction object

    #### write data to lmdb
    pbar = util.ProgressBar(len(all_img_list))

    i = 0
    for path, key in zip(all_img_list, keys):
        pbar.update('Write {}'.format(key))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        key_byte = key.encode('ascii')
        H, W, C = img.shape  # fixed shape
        assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, img)
        i += 1
        if  i % batch == 1:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish reading and writing {} images.'.format(len(all_img_list)))
            
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    if mode == 'HR':
        meta_info['name'] = 'PlayerVids_train_HR'
    elif mode == 'LR':
        meta_info['name'] = 'PlayerVids_train_LR'
    meta_info['resolution'] = '{}_{}_{}'.format(3, H_dst, W_dst)
    key_set = []
    for key in keys:
        print(key)
        a, b,_ = key.split('_')
        key_set.append('{}_{}'.format(a, b))
    meta_info['keys'] = key_set
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'PlayerVids_train_keys.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


if __name__ == "__main__":
    playervids()
    #test_lmdb('/data/datasets/SR/vimeo_septuplet/vimeo7_train_GT.lmdb', 'vimeo7')

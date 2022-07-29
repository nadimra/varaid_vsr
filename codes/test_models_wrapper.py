import os
import argparse

modelNameList = ['ModelL']
# Testing 

modelD_config = {
    'scale':2,
    'frameClip':5,
    'datasetFolder':'/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LRx2/'
}

modelE_config = {
    'scale':8,
    'frameClip':5,
    'datasetFolder':'/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LRx8/'
}

modelF_config = {
    'scale':4,
    'frameClip':3,
    'datasetFolder':'/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LR/'
}

modelG_config = {
    'scale':4,
    'frameClip':5,
    'datasetFolder':'/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LR/'
}

modelH_config = {
    'scale':4,
    'frameClip':5,
    'datasetFolder':'/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LR/'
}
modelI_config = {
    'scale':4,
    'frameClip':5,
    'datasetFolder':'/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LR/'
}

modelJ_config = {
    'scale':4,
    'frameClip':5,
    'datasetFolder':'/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LR/'
}

modelK_config = {
    'scale':4,
    'frameClip':3,
    'datasetFolder':'/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LR/'
}

modelL_config = {
    'scale':4,
    'frameClip':5,
    'datasetFolder':'/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LR/'
}

modelTest_config = {
    'scale':4,
    'frameClip':5,
    'datasetFolder':'/vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LR/'
}



for modelName in modelNameList:
    for epoch in range(2,21):
        if modelName == 'ModelL':
            os.system("python test_footballvids_models.py --modelNum {} --modelName {} --scale {} --frameClip {} --datasetFolder {}"
            .format(epoch,modelName,modelL_config['scale'],modelL_config['frameClip'],modelL_config['datasetFolder']))

#python test_footballvids_models_1500.py --modelNum 3 --modelName ModelH --scale 4 --frameClip 5 --datasetFolder /vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LR/ --saveImgs
#python test_footballvids_models.py --modelNum 4 --modelName ModelTest --scale 4 --frameClip 5 --datasetFolder /vol/bitbucket/nr421/Zooming-Slow-Mo-CVPR-2020/datasets/FootballVids/LR/ --saveImgs
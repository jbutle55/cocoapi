import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
dataDir='../'
dataType='val2014'
annFile = '/home/justin/Data/aerial-cars-private/aerial_yolo/valid/fixed_annotations2.json'
# annFile = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/Datasets/COCO2014/annotations/instances_val2014.json'
cocoGt=COCO(annFile)

#initialize COCO detections api
resFile='%s/results/%s_%s_fake%s100_results.json'
resFile = resFile%(dataDir, prefix, dataType, annType)
resFile = '/Models/detectron2/aerial_cars/coco_instances_results.json'
# resFile = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/cocoapi/results/instances_val2014_fakesegm100_results.json'
cocoDt=cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds[0:100]
imgId = imgIds[np.random.randint(100)]

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
print(cocoEval.params.iouThrs)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
# cocoEval.summarize_per_category()
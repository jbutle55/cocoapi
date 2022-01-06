__author__ = 'tsungyi'

import numpy as np
import datetime
import time
from collections import defaultdict
from . import mask as maskUtils
import copy
import matplotlib.pyplot as plt
import os


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                     for imgId in p.imgIds
                     for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                         for catId in catIds
                         for areaRng in p.areaRng
                         for imgId in p.imgIds
                         ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        iou = maskUtils.iou(d, g, iscrowd)
        return iou

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def compute_single_IoU(self, imgId, dt, gt):
        # Compute IoU between gt and dt of non-matching categories
        p = self.params
        if p.iouType == 'segm':
            g = gt['segmentation']
            d = dt['segmentation']
        elif p.iouType == 'bbox':
            g = [gt['bbox']]
            d = [dt['bbox']]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(gt['iscrowd'])]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computeIoU_nonmatching(self, imgId, catId1, catId2):
        # Compute IoU between gt and dt of non-matching categories
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId1]
            dt = self._dts[imgId, catId2]
        else:
            print('Please use p.useCats!')
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet, roc=True):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            # Predictions and gts of the category in question (catId)
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        if roc:
            p.scoreThrs = np.linspace(0.0, 1.0, 50)
            # List all preds and gts NOT of category in question (catId)
            non_target_dts = []
            non_target_gts = []
            for i in range(0, 1000):
                if i != catId and len(self._dts[imgId, i]) > 0:
                    for item in self._dts[imgId, i]:
                        non_target_dts.append(item)
                if i != catId and len(self._gts[imgId, i]) > 0:
                    for item in self._gts[imgId, i]:
                        non_target_gts.append(item)

            # Combine lists of gts and preds of class in question and not in question
            dt_combined = dt + non_target_dts
            gt_combined = gt + non_target_gts

            # Determine all TP, TN, FP, FN between all preds and gts in image
            num_tn = {f'{t}': 0 for t in p.scoreThrs}
            num_tp = {f'{t}': 0 for t in p.scoreThrs}
            num_fn = {f'{t}': 0 for t in p.scoreThrs}
            num_fp = {f'{t}': 0 for t in p.scoreThrs}
            # num_tn = 0
            # num_tp = 0
            # num_fn = 0
            # num_fp = 0
            for idt in dt_combined:
                # catId is the category in question
                target_id = 1
                if idt['category_id'] != target_id:
                    # Only look at ROC for a single class at a time
                    continue

                dt_cat = idt['category_id']
                pred_found = False
                # For each prediction, loop through all gts to see if it's a TP first, then other possibilites
                for igt in gt_combined:
                    if igt['category_id'] != target_id:
                        continue
                    # catId is the category in question
                    gt_cat = igt['category_id']

                    # First calc. IoU to see if any overlap between detection and gt
                    single_iou = self.compute_single_IoU(imgId, idt, igt)

                    # If IoU greater than 0.5, potential for TP or FN
                    if single_iou.squeeze() > 0.5:
                        pred_found = True
                        for sidx, score in enumerate(p.scoreThrs):
                            dt_score = idt['score']

                            # Setup for only a single class
                            if dt_score >= score:  # Positive pred
                                num_tp[f'{score}'] += 1
                            else:
                                num_fn[f'{score}'] += 1

                            """if gt_cat == catId:
                                if dt_cat == gt_cat:  # TP
                                    # Check score above or below thresh
                                    if dt_score > score:  # Predicting "Is 1" so TP
                                        num_tp[f'{score}'] += 1
                                    else:  # Otherwise predicting "Not 1" so FN
                                        num_fn[f'{score}'] += 1

                            else:
                                if dt_cat != gt_cat:
                                    if dt_cat == catId:  # FP
                                        num_fp[f'{score}'] += 1
                                    elif dt_cat != catId:  # TN
                                        num_tn[f'{score}'] += 1
                                elif dt_cat == gt_cat:  # TN
                                    if dt_score > score:
                                        num_tn[f'{score}'] += 1"""

                if not pred_found:
                    # Means there was no good prediction for any gt
                    # Therefore could be TN or FP
                    for sidx, score in enumerate(p.scoreThrs):
                        dt_score = idt['score']

                        # Setup for only a single class
                        if dt_score >= score:  # False Positive pred
                            num_fp[f'{score}'] += 1
                        else:
                            num_tn[f'{score}'] += 1

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))

        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']

        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
            'image_id':     imgId,
            'category_id':  catId,
            'aRng':         aRng,
            'maxDet':       maxDet,
            'dtIds':        [d['id'] for d in dt],
            'gtIds':        [g['id'] for g in gt],
            'dtMatches':    dtm,
            'gtMatches':    gtm,
            'dtScores':     [d['score'] for d in dt],
            'gtIgnore':     gtIg,
            'dtIgnore':     dtIg,
            'trueNeg':      num_tn,
            'falseNeg':     num_fn,
            'falsePos':     num_fp,
            'truePos':      num_tp
        }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)  # IoU Thresholds
        R           = len(p.recThrs)  # Recall Thresholds
        K           = len(p.catIds) if p.useCats else 1  # Number Categories
        A           = len(p.areaRng)  # Area ranges
        M           = len(p.maxDets)  # Max number of preds
        C           = len(p.scoreThrs)  # ROC Score Levels
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))
        false_pos_rate = -np.ones((T, K, A, M, C))
        true_pos_rate = -np.ones((T, K, A, M, C))


        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        false_pos_rate[:, k, a, m, :] = 0
                        true_pos_rate[:, k, a, m, :] = 0
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        false_pos_rate[:, k, a, m, :] = 0
                        true_pos_rate[:, k, a, m, :] = 0
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)  # Total false positive
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        q_r = np.zeros((R,))  # Created to create recall same size as precision
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist(); q_r = q_r.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)

                    # ROC data
                    tp_total = {}
                    fp_total = {}
                    tn_total = {}
                    fn_total = {}
                    for c, score in enumerate(p.scoreThrs):
                        tp_total[f'{score}'] = np.sum([e['truePos'][f'{score}'] for e in E])
                        fp_total[f'{score}'] = np.sum([e['falsePos'][f'{score}'] for e in E])
                        tn_total[f'{score}'] = np.sum([e['trueNeg'][f'{score}'] for e in E])
                        fn_total[f'{score}'] = np.sum([e['falseNeg'][f'{score}'] for e in E])

                        # [iou_thresh, category, area, max_dets]
                        if (tp_total[f'{score}'] + fn_total[f'{score}']) == 0:
                            true_pos_rate[:, k, a, m, c] = 0
                        else:
                            true_pos_rate[:, k, a, m, c] = tp_total[f'{score}'] / (tp_total[f'{score}'] + fn_total[f'{score}'])
                        if (tn_total[f'{score}'] + fp_total[f'{score}']) == 0:
                            false_pos_rate[:, k, a, m, c] = 0
                        else:
                            false_pos_rate[:, k, a, m, c] = fp_total[f'{score}'] / (tn_total[f'{score}'] + fp_total[f'{score}'])
                            # tn = tn_total[f'{score}']
                            # fp =fp_total[f'{score}']
                            # print(f'True Neg to False Pos Ratio: {tn / fp}')

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
            'fpr': false_pos_rate,
            'tpr': true_pos_rate
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self, roc=False):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100, f1=None ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            if f1:
                titleStr = 'F1 Score'
                typeStr = '(F1)'
            elif ap == 1:
                titleStr = 'Average Precision'
                typeStr = '(AP)'
            else:
                titleStr = 'Average Recall'
                typeStr = '(AR)'
            # titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            # typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if f1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                rec = s[:, :, :, aind, mind]

                # dimension of recall: [TxKxAxM]
                s = self.eval['recall_full']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                prec = s[:, :, :, aind, mind]

                f1_scores = 2 * (rec * prec) / (rec + prec)

                if len(f1_scores[f1_scores > -1]) == 0:
                    mean_f1 = -1
                else:
                    mean_f1 = np.mean(f1_scores[f1_scores > -1])

            elif ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            if f1:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_f1))
            else:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        # Below for testing detection of only small objects in narrow lots
        # def _summarizeDets():
        #     print('Summarizing small objects only...')
        #     stats = np.zeros((23,))
        #     stats[0] = _summarize(1, areaRng='s1', maxDets=self.params.maxDets[2])
        #     stats[1] = _summarize(1, areaRng='s2', maxDets=self.params.maxDets[2])
        #     stats[2] = _summarize(1, areaRng='s3', maxDets=self.params.maxDets[2])
        #     stats[3] = _summarize(1, areaRng='s4', maxDets=self.params.maxDets[2])
        #     stats[4] = _summarize(1, areaRng='s5', maxDets=self.params.maxDets[2])
        #     stats[5] = _summarize(1, areaRng='s6', maxDets=self.params.maxDets[2])
        #     stats[6] = _summarize(1, areaRng='s7', maxDets=self.params.maxDets[2])
        #     stats[7] = _summarize(1, areaRng='s8', maxDets=self.params.maxDets[2])
        #     stats[8] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        #     stats[9] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        #     stats[10] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        #     stats[11] = _summarize(1, areaRng='all', maxDets=self.params.maxDets[2], iouThr=0.5)
        #     stats[12] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2], iouThr=0.5)
        #     stats[13] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2], iouThr=0.5)
        #     stats[14] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2], iouThr=0.5)
        #     stats[15] = _summarize(1, areaRng='s1', maxDets=self.params.maxDets[2], iouThr=0.5)
        #     stats[16] = _summarize(1, areaRng='s2', maxDets=self.params.maxDets[2], iouThr=0.5)
        #     stats[17] = _summarize(1, areaRng='s3', maxDets=self.params.maxDets[2], iouThr=0.5)
        #     stats[18] = _summarize(1, areaRng='s4', maxDets=self.params.maxDets[2], iouThr=0.5)
        #     stats[19] = _summarize(1, areaRng='s5', maxDets=self.params.maxDets[2], iouThr=0.5)
        #     stats[20] = _summarize(1, areaRng='s6', maxDets=self.params.maxDets[2], iouThr=0.5)
        #     stats[21] = _summarize(1, areaRng='s7', maxDets=self.params.maxDets[2], iouThr=0.5)
        #     stats[22] = _summarize(1, areaRng='s8', maxDets=self.params.maxDets[2], iouThr=0.5)

        #    # _summarizeROC()
        #    return stats

        def _summarizeDets():
            print('Summarizing...')
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            _summarizeROC()
            # print('F1 Testing Below')
            # _summarizeFscore()
            return stats

        def _summarizeKps():
            stats = np.zeros((11,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            stats[10] = _summarize(0, maxDets=20, areaRng='all')

            return stats

        def _summarizeROC(iouThr=0.5):
            print('Evaluating ROC')
            # Summarize ROC stats
            p = self.params
            t = np.where(iouThr == p.iouThrs)[0]

            fpr = self.eval['fpr'][t, :, 0, 2, :]  # Use all areas and max detections
            tpr = self.eval['tpr'][t, :, 0, 2, :]

            # Average over all class categories
            fpr_single = np.mean(fpr, axis=1)[0]
            tpr_single = np.mean(tpr, axis=1)[0]

            np.set_printoptions(precision=4)

            # print('FPR and TPR values...')
            # print(f'FPR: {fpr_single}')
            # print(f'TPR: {tpr_single}')

            print(f'ROC file at: {os.getcwd()}')
            with open('roc_records.txt', 'a+') as file:
                file.write(f'target id: 1 - Cars')
                file.write('\n')
                # file.write(str(p.scoreThrs))
                # file.write('\n')
                file.write(str(fpr_single))
                file.write('\n')
                file.write(str(tpr_single))
                file.write('\n')
                file.write('\n')
                # file.write(str(fpr))
                # file.write('\n')
                # file.write(str(tpr))

            # self.plot_roc(fpr_mean, tpr_mean)
            return

        def _summarizeFscore():
            stats = np.zeros((4,))
            stats[0] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2], f1=True)
            stats[1] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2], f1=True)
            stats[2] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2], f1=True)
            stats[3] = _summarize(1, areaRng='all', maxDets=self.params.maxDets[2], f1=True)

            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

    # add for per category metric from here
    def summarize_per_category(self):
        '''
        Compute and display summary metrics for evaluation results *per category*.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize_single_category(ap=1, iouThr=None, categoryId=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ CategoryId={:>3d} | IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if categoryId is not None:
                    category_index = [i for i, i_catId in enumerate(p.catIds) if i_catId == categoryId]
                    s = s[:, :, category_index, aind, mind]
                else:
                    s = s[:, :, :, aind, mind]

            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if categoryId is not None:
                    category_index = [i for i, i_catId in enumerate(p.catIds) if i_catId == categoryId]
                    s = s[:, category_index, aind, mind]
                else:
                    s = s[:, :, aind, mind]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            # print(iStr.format(titleStr, typeStr, catId, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets_per_category():
            category_stats = np.zeros((12, len(self.params.catIds)))
            for category_index, category_id in enumerate(self.params.catIds):
                category_stats[0][category_index] = _summarize_single_category(1,
                                                                               categoryId=category_id)
                category_stats[1][category_index] = _summarize_single_category(1,
                                                                               iouThr=.5,
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[2][category_index] = _summarize_single_category(1,
                                                                               iouThr=.75,
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[3][category_index] = _summarize_single_category(1,
                                                                               areaRng='small',
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[4][category_index] = _summarize_single_category(1,
                                                                               areaRng='medium',
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[5][category_index] = _summarize_single_category(1,
                                                                               areaRng='large',
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[6][category_index] = _summarize_single_category(0,
                                                                               maxDets=self.params.maxDets[0],
                                                                               categoryId=category_id)
                category_stats[7][category_index] = _summarize_single_category(0,
                                                                               maxDets=self.params.maxDets[1],
                                                                               categoryId=category_id)
                category_stats[8][category_index] = _summarize_single_category(0,
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[9][category_index] = _summarize_single_category(0,
                                                                               areaRng='small',
                                                                               maxDets=self.params.maxDets[2],
                                                                               categoryId=category_id)
                category_stats[10][category_index] = _summarize_single_category(0,
                                                                                areaRng='medium',
                                                                                maxDets=self.params.maxDets[2],
                                                                                categoryId=category_id)
                category_stats[11][category_index] = _summarize_single_category(0,
                                                                                areaRng='large',
                                                                                maxDets=self.params.maxDets[2],
                                                                                categoryId=category_id)
            return category_stats


        def _summarizeKps_per_category():
            category_stats = np.zeros((10, len(self.params.catIds)))
            for category_index, category_id in self.params.catIds:
                category_stats[0][category_index] = _summarize_single_category(1,
                                                                               maxDets=20,
                                                                               categoryId=category_id)
                category_stats[1][category_index] = _summarize_single_category(1,
                                                                               maxDets=20,
                                                                               iouThr=.5,
                                                                               categoryId=category_id)
                category_stats[2][category_index] = _summarize_single_category(1,
                                                                               maxDets=20,
                                                                               iouThr=.75,
                                                                               categoryId=category_id)
                category_stats[3][category_index] = _summarize_single_category(1,
                                                                               maxDets=20,
                                                                               areaRng='medium',
                                                                               categoryId=category_id)
                category_stats[4][category_index] = _summarize_single_category(1,
                                                                               maxDets=20,
                                                                               areaRng='large',
                                                                               categoryId=category_id)
                category_stats[5][category_index] = _summarize_single_category(0,
                                                                               maxDets=20,
                                                                               categoryId=category_id)
                category_stats[6][category_index] = _summarize_single_category(0,
                                                                               maxDets=20,
                                                                               iouThr=.5,
                                                                               categoryId=category_id)
                category_stats[7][category_index] = _summarize_single_category(0,
                                                                               maxDets=20,
                                                                               iouThr=.75,
                                                                               categoryId=category_id)
                category_stats[8][category_index] = _summarize_single_category(0,
                                                                               maxDets=20,
                                                                               areaRng='medium',
                                                                               categoryId=category_id)
                category_stats[9][category_index] = _summarize_single_category(0,
                                                                               maxDets=20,
                                                                               areaRng='large',
                                                                               categoryId=category_id)
            return category_stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize_per_category = _summarizeDets_per_category
        elif iouType == 'keypoints':
            summarize_per_category = _summarizeKps_per_category
        self.category_stats = summarize_per_category()

    def __str__(self):
        self.summarize_per_category()
    # add for metric per category end here

    def compute_roc(self):

        """
        Handle all ROC calculations and accumulation of data.
        Will calculate N ROC statistics for N classes
        Will also average these statistics by N for single all-encompassing ROC
        Steps for a set confidence threshold:
        1. Calculate TPR/Recall/Sensitivity
        2. Calculate Specificity
        3. Calculate FPR = 1 - Specificity
        4. Average across N classes
        5. Plot if desired
        """

        return

    def plot_roc(self, fpr, tpr):
        # Reverse order of fpr and tpr so highest IoU threshold is first
        fpr_flip = np.flip(fpr, axis=0)
        tpr_flip = np.flip(tpr, axis=0)
        plt.plot(fpr_flip)
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.show()
        return


class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        # TODO Can create new size divisions here in below 2 lines
        self.areaRng = [[0 ** 2, 1e5 ** 2],
                        [0 ** 2, 4**2],  # s1
                        [4 ** 2, 8 ** 2],  # s2
                        [8 ** 2, 12 ** 2],  # s3
                        [12 ** 2, 16 ** 2],  # s4
                        [16 ** 2, 20 ** 2],  # s5
                        [20 ** 2, 24 ** 2],  # s6
                        [24 ** 2, 28 ** 2],  # s7
                        [28 ** 2, 32 ** 2],  # s8
                        [0 ** 2, 32 ** 2],  # Small
                        [32 ** 2, 96 ** 2],  # Medium
                        [96 ** 2, 1e5 ** 2]]  # Large
        self.areaRngLbl = ['all', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 'small', 'medium', 'large']

        #self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        #self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1
        self.scoreThrs = np.linspace(0, 1, 10)  # For ROC confidence levels

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None

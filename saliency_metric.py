import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve, distance_transform_edt as bwdist

class cal_fm(object):
    # Fmeasure(maxFm,meanFm)---Frequency-tuned salient region detection(CVPR 2009)
    def __init__(self, num, thds=255):
        self.num = num
        self.thds = thds
        self.precision = np.zeros((self.num, self.thds))
        self.recall = np.zeros((self.num, self.thds))
        self.meanF = np.zeros((self.num, 1))
        self.maxF = np.zeros((self.num, 1))

    def update(self, gt, pred):
        for i in range(self.thds):
            threshold = i / (self.thds - 1)
            binary_pred = pred >= threshold

            gt_numpy = gt.cpu().numpy()
            binary_pred_numpy = binary_pred.cpu().numpy()

            tp = np.sum((binary_pred_numpy == 1) & (gt_numpy == 1))
            fp = np.sum((binary_pred_numpy == 1) & (gt_numpy == 0))
            fn = np.sum((binary_pred_numpy == 0) & (gt_numpy == 1))

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            self.precision[self.num - 1, i] = precision
            self.recall[self.num - 1, i] = recall

    def compute_fmeasure(self):
        beta_square = 0.3
        fmeasure = (1 + beta_square) * (self.precision * self.recall) / (beta_square * self.precision + self.recall + 1e-8)
        self.meanF = np.mean(fmeasure, axis=1)
        self.maxF = np.max(fmeasure, axis=1)

    def get_results(self):
        return self.meanF, self.maxF
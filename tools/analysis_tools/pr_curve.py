import pickle
import os.path as osp
import matplotlib.pyplot as plt
import ast

PR_path = 'N:/Programing/Python/GiraffeDet/mmdetection/work_dirs/fast-rcnn_r50-caffe_fpn_1x_package/result.pkl'
f = open(PR_path, 'rb')
info = pickle.load(f)
print(info)
# plt.plot(recall, precision)
# plt.show()
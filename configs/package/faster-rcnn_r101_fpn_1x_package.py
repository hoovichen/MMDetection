# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster-rcnn_r101-caffe_fpn_1x_coco.py'
# _base_.visualizer.vis_backends = [
# dict(type='LocalVisBackend'),
# dict(type='TensorboardVisBackend'),
# dict(type='WandbVisBackend'),]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        type='FreeAnchorRetinaHead',
        bbox_head=dict(num_classes=5),
        ))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('Box-package', 'Box', 'Box_broken', 'Open_package', 'Package')
data_root = 'data/package/'
palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228)]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        ann_file='train2017/_annotations.coco.json',
        data_prefix=dict(img='train2017/')
    )
)
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        ann_file='val2017/_annotations.coco.json',
        data_prefix=dict(img='val2017/')

    )
)
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        ann_file='test2017/_annotations.coco.json',
        data_prefix=dict(img='test2017/')
    )
)
val_evaluator = dict(
    ann_file=data_root + 'val2017/_annotations.coco.json'
)
test_evaluator = dict(
    ann_file=data_root + 'test2017/_annotations.coco.json'
)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer, accumulative_counts=2)
train_cfg = dict(max_epochs=12, val_interval=7)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './model/resnet101_caffe.pth'

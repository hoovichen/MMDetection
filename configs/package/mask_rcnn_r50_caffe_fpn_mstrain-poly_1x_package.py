# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5),
        mask_head=dict(num_classes=5)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('Box-package', 'Box', 'Box_broken', 'Open_package', 'Package')
data_root = 'data/package/'
palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228)]
data = dict(
    samples_per_gpu=1,
    worker_per_gpy=1,
    train=dict(
        img_prefix=data_root + 'train2017/',
        classes=classes,
        ann_file=data_root + 'train2017/_annotations.coco.json'),
    val=dict(
        img_prefix=data_root + 'val2017/',
        classes=classes,
        ann_file=data_root + 'val2017/_annotations.coco.json'),
    test=dict(
        img_prefix=data_root + 'test2017/',
        classes=classes,
        ann_file=data_root + 'test2017/_annotations.coco.json'))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='train2017/_annotations.coco.json'
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='val2017/_annotations.coco.json'
    )
)
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='test2017/_annotations.coco.json'
    )
)
val_evaluator = dict(
    ann_file=data_root + 'val2017/_annotations.coco.json'
)
test_evaluator = dict(
    ann_file=data_root + 'test2017/_annotations.coco.json'
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
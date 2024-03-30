# Image Captioning

1/ Download COCO dataset:
- [Train2014](http://images.cocodataset.org/zips/train2014.zip)
- [Test2014](http://images.cocodataset.org/zips/test2014.zip)
- [Annotations2014](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

The folder structure might look like this:
```
.
└── image-captioning/
    ├── models
    ├── image_captioning.ipynb
    ├── data_loader.py
    ├── vocabulary.py
    ├── model.py
    ├── ...
    └── opt/
        └── cocoapi/
            ├── annotations/
            │   └── captions/
            │       ├── captions_train2014.json
            │       └── ....
            └── images/
                ├── train2014/
                │   ├── COCO_train2014_000000000009.jpg
                │   └── ...
                └── test2014/
                    ├── COCO_test2014_000000000001.jpg
                    └── ...
```

2/ Explore the file `image_captioning.ipynb` (you might need to install some packages).
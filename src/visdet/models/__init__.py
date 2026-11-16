"""Framework adapters with one interface.

`train(cfg, data_yaml, out_dir, device) -> weights path` and
`predict(cfg, weights, images, device) -> DataFrame` in cached-image pixels with columns
image_id, cls, score, x, y, w, h. `images` has image_id and path columns.
"""


def get(framework: str):
    if framework == "ultralytics":
        from visdet.models import ultralytics as adapter
    elif framework == "torchvision":
        from visdet.models import torchvision as adapter
    else:
        raise ValueError(f"unknown framework {framework}")
    return adapter

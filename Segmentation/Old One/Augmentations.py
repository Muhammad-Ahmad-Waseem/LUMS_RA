import albumentations

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# ////////////////////////Augmentations\\\\\\\\\\\\\\\\\\\\\\\\
def get_training_augmentation():
    train_transform = [albumentations.HorizontalFlip(p=0.5),
                       # albumentations.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                       albumentations.PadIfNeeded(min_height=1024, min_width=1024, always_apply=True, border_mode=0),
                       # albumentations.RandomCrop(height=320, width=320, always_apply=True),
                       albumentations.GaussNoise(p=0.4),
                       # albumentations.IAAPerspective(p=0.5),
                       albumentations.OneOf(
                           [
                               albumentations.CLAHE(p=1),
                               albumentations.RandomBrightness(p=1),
                               albumentations.RandomGamma(p=1),
                           ],
                           p=0.5,
                       ),
                       albumentations.OneOf(
                           [
                               albumentations.Sharpen(p=1),
                               albumentations.Blur(blur_limit=3, p=1),
                               albumentations.MotionBlur(blur_limit=3, p=1),
                           ],
                           p=0.5,
                       ),
                       albumentations.OneOf(
                           [
                               albumentations.RandomContrast(p=1),
                               albumentations.HueSaturationValue(p=1),
                           ],
                           p=0.5,
                       ),
                       albumentations.Lambda(mask=round_clip_0_1)
                       ]

    return albumentations.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albumentations.PadIfNeeded(1024, 1024)
    ]
    return albumentations.Compose(test_transform)
import augly.video as vidaugs


if __name__ == "__main__":
    COLOR_JITTER_PARAMS = {
        "brightness_factor": 0.15,
        "contrast_factor": 1.3,
        "saturation_factor": 2.0,
    }

    AUGMENTATIONS = [
        vidaugs.ColorJitter(**COLOR_JITTER_PARAMS),
        vidaugs.OneOf(
            [
                vidaugs.RandomEmojiOverlay(),
                vidaugs.Shift(x_factor=0.25, y_factor=0.25),
            ]
        ),
    ]

    TRANSFORMS = vidaugs.Compose(AUGMENTATIONS)

    video_path = "/data/lifengjun/SlowFast/020001.mp4"
    out_video_path = "/data/lifengjun/SlowFast/020001_process.mp4"

    TRANSFORMS(video_path, out_video_path)  # transformed video now stored in `out_video_path`
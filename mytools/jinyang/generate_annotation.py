import csv
import os
import json
import argparse

if __name__ == '__main__':
    template = {
        "1": "Make two waves with palms up",
        "2": "Make two waves with palms down",
        "3": "Point one finger up at the ceiling and rotate clockwise",
        "4": "Point one hand up to the ceiling and rotate counterclockwise",
        "5": "OK gesture",
        "6": "Point thumb 15 degrees vertically to the left",
        "7": "Point thumb 15 degrees vertically to the up",
        "8": "Point thumb 15 degrees vertically to the right",
        "9": "Point thumb 15 degrees vertically to the down",
        "10": "Cross two index fingers facing the camera",
        "11": "Waving to the left",
        "12": "Waving to the right",
        "13": "Without action"
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_type", default="RGB")
    args = parser.parse_args()

    modes = ['train', 'val', 'test']
    label = ['original_vido_id video_id frame_id path labels']
    video_id = 0

    for mode in modes:
        # targetdir = f'/data/lifengjun/gesture_dataset/dataset/image/{args.video_type}/{mode}'
        targetdir = '/data/lifengjun/gesture_dataset/dataset/image/img_0827/{}'.format(mode)
        res = []
        jsontext = []
        targetdir_path = os.listdir(targetdir)
        targetdirs = sorted(targetdir_path)
        for labels in targetdirs:
            video_path = os.path.join(targetdir, labels)
            videos = sorted(os.listdir(video_path))
            for original_vido_id in videos:
                images_path = os.path.join(video_path, original_vido_id)
                images = sorted(os.listdir(images_path))
                line = {"id": f"{original_vido_id}", "template": f"{template[str(labels)]}"} if mode != 'test' else {"id": f"{video_id}"}
                if len(images) == 0:
                    print(images_path)
                else:
                    jsontext.append(line)
                for frame_idx, image in enumerate(images):
                    line = [f"{original_vido_id} {video_id} {frame_idx} {images_path}/{image} {labels}"]
                    res.append(line)
                video_id += 1
        print(len(res), len(jsontext))
        with open(os.path.join(f'{mode}.csv'), 'w', newline="") as f:
            # 基于打开的文件，创建 csv.writer 实例
            writer = csv.writer(f)

            # 写入数据。
            # writerows() 一次写入多行。
            writer.writerow(label)
            writer.writerows(res)
            f.close()

        name = "validation" if mode == 'val' else mode
        with open(os.path.join(f'something-something-v2-{name}.json'), 'w') as f:
            f.write(json.dumps(jsontext))
            f.close()

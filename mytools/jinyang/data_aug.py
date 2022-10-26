import os

if __name__ == "__main__":

    for mode in ['train']:
        os.system(
            f"python vflip.py --source /data/sujiapeng/1-2-3-4-11-12-13-new/video/RGB/{mode}/3/ --target /data/sujiapeng/1-2-3-4-11-12-13-new/video/RGB/{mode}/4/ --vflip True")
        os.system(
            f"python vflip.py --source /data/sujiapeng/1-2-3-4-11-12-13-new/video/RGB/{mode}/4/ --target /data/sujiapeng/1-2-3-4-11-12-13-new/video/RGB/{mode}/3/ --vflip True")
        os.system(
            f"python vflip.py --source /data/sujiapeng/1-2-3-4-11-12-13-new/video/RGB/{mode}/11/ --target /data/sujiapeng/1-2-3-4-11-12-13-new/video/RGB/{mode}/12/ --vflip True")
        os.system(
            f"python vflip.py --source /data/sujiapeng/1-2-3-4-11-12-13-new/video/RGB/{mode}/12/ --target /data/sujiapeng/1-2-3-4-11-12-13-new/video/RGB/{mode}/11/ --vflip True")
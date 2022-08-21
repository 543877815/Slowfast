import numpy as np
import shutil
import os

if __name__ == "__main__":
    ## 这里写的逻辑是将source的文件以六位数的方式重新命名
    ## 需要注意的是当source==target的时候可能会出现覆盖问题
    label = 12
    for l in range(1, label+1):
        id = 1
        source_path = 'H:\\gesture_project\\dataset\\video\\IR1\\{}'.format(l)
        target_path = 'H:\\gesture_project\\dataset\\video\\IR\\{}'.format(l)
        os.makedirs(target_path, exist_ok=True)
        for file in os.listdir(source_path):
            cls = os.path.basename(source_path)
            name = "{}{}.avi".format(cls.zfill(2), str(id).zfill(4))
            target = os.path.join(target_path, name)
            source = os.path.join(source_path, file)
            print("{}=>{}".format(source, target))
            id += 1
            shutil.move(source, target)
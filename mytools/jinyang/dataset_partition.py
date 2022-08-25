import numpy as np
import shutil
import os

def partition1():
    ## 这里的逻辑是根据group自动分组数据集
    group = 4
    data_type = "IR"
    label = 13
    test_target = "H:\\gesture_project\\dataset\\video\\test\\{}".format(data_type)
    train_target = "H:\\gesture_project\\dataset\\video\\train\\{}".format(data_type)

    for i in range(1, label):
        source_path = "H:\\gesture_project\\dataset\\video\\{}\\{}".format(data_type, i)
        files = os.listdir(source_path)
        files.sort()
        total = len(files)
        idx = np.random.permutation(total)
        test_total = int(total / group)
        test_idx = idx[:test_total]
        train_idx = idx[test_total:]

        train_target_path = os.path.join(train_target, str(i))
        test_target_path = os.path.join(test_target, str(i))
        os.makedirs(train_target_path, exist_ok=True)
        os.makedirs(test_target_path, exist_ok=True)

        for i in test_idx:
            source = os.path.join(source_path, files[i])
            target = os.path.join(test_target_path, files[i])
            print("{}=>{}".format(source, target))
            shutil.copy(source, target)

        for i in train_idx:
            source = os.path.join(source_path, files[i])
            target = os.path.join(train_target_path, files[i])
            print("{}=>{}".format(source, target))
            shutil.copy(source, target)


def partition2():
    ## 这里的逻辑是将train随机划分到test中
    import numpy as np
    import shutil
    import os
    group = 5
    test_target = r"H:\gesture_project\mydivide\test"
    train_target = r"H:\gesture_project\mydivide\train"

    for label in os.listdir(train_target):
        label_path = os.path.join(train_target, label)
        image_path = os.listdir(label_path)
        image_path.sort()
        total = len(image_path)
        idx = np.random.permutation(total)
        test_total = int(total / group)
        test_idx = idx[:test_total]
        target_path = os.path.join(test_target, label)
        os.makedirs(target_path, exist_ok=True)
        for i in test_idx:
            source = os.path.join(label_path, image_path[i])
            target = os.path.join(target_path, image_path[i])
            print("{}=>{}".format(source, target))
            shutil.move(source, target)


if __name__ == "__main__":
    partition1()
    partition2()
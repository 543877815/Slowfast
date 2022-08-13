import os
import time
from multiprocessing import Process


def func(index):
    print("第%s封邮件已经发送..." % (index))


if __name__ == '__main__':
    for i in range(10):
        p = Process(target=func, args=(i,))
        p.start()
    print("发出第十封邮件...")
import os
import config
from markup_data import MarkUp
from uuid import uuid4

if __name__ == '__main__':

    instance = MarkUp()
    random_str = str(uuid4())
    instance.mark_up_video("/home/atticus/PycharmProjects/real-time-facial-landmarks/samples/bbal9a.mpg", random_str)



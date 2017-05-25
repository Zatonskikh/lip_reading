import os
import config
from markup_data import MarkUp
from uuid import uuid4
from utils.utils import start_collect_data

if __name__ == '__main__':

    instance = MarkUp()
    if config.COLLECT_DATA:
        start_collect_data(instance)
    else:
        random_str = str(uuid4())
        instance.mark_up_video(os.path.join(config.DATA_PATH, "otsutstvie/IMG_1081.MOV"), random_str)
    print "Not found %d" % instance.error_count

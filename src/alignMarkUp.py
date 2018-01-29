import os
import re
import shutil
START_ALIGN = '0'
STOP_ALIGN = '85000'
FLAG_RENAME = False
FLAG_ALIGN = False
FLAG_CP = True
DEST_DIR_DATASET = "/home/atticus/lipNet_impl/uir2/training/unseen_speakers/datasets/val"

def set_up_align_dir(dir_path):
    align_dir_path = os.path.join(dir_path, 'align')
    if os.path.isdir(align_dir_path):
        os.remove(align_dir_path)
    else:
        os.mkdir(align_dir_path)
    return align_dir_path


def dir_processing(path_to_dir, count, align_dir_path = None):
    files = os.listdir(path_to_dir)
    if align_dir_path is None:
        align_dir_path = set_up_align_dir(path_to_dir)
    videos = filter(lambda file: re.search(".avi", file), files)
    video_list = [os.path.join(path_to_dir, video) for video in videos]
    for video in video_list:
        if FLAG_RENAME:
            os.rename(video, os.path.join(path_to_dir, "{}.avi".format(count)))
            count = count + 1
        if FLAG_ALIGN:
            video_name = os.path.basename(video).split(".")[0]
            with open(os.path.join(align_dir_path, video_name) + ".align", "w") as f:
                f.write(START_ALIGN + " " + STOP_ALIGN + " " + path_to_dir.split("/")[-1])
        if FLAG_CP:
            video_name = os.path.basename(video).split(".")[0]
            shutil.copyfile(video, os.path.join(DEST_DIR_DATASET, video_name) + ".avi")
    return count

def rm_not_existing_aligns(path_to_dir):
    path_to_align = os.path.join(os.path.dirname(path_to_dir), 'align')
    align_list = [os.path.join(path_to_align, align) for align in os.listdir(path_to_align)]
    files = os.listdir(path_to_dir)
    dirs_existing = filter(lambda file: '.' not in file, files)
    not_existing_dirs =  [x for x in range(1, 2089) if str(x) not in dirs_existing]
    for x in align_list:
        if int(os.path.basename(x.split('.')[0])) in not_existing_dirs:
            print ('removed {} file'.format(x))
            os.remove(x)


def check_frame_count(path_to_dir):
    for p, w ,d in os.walk(path_to_dir):
        if (len(d) > 50):
            print("dir: {}, count: {}".format(p, len(d)))
            bad_image = filter(lambda fil: int(fil.split("_")[1].split(".")[0][1:]) >= 50, d)
            for img in bad_image:
                os.remove(os.path.join(p, img))


def copy_frames_to_50_frames(path_to_dir):
    for p, w ,d in os.walk(path_to_dir):
        if (p.split('/')[-1] == 'val'):
            continue
        if (len(d) < 50):
            print("dir: {}, count: {}".format(p, len(d)))
            last_image = len(d) - 1
            if last_image < 10:
                last_image_name = "mouth_00" + str(last_image) + ".png"
            else:
                last_image_name = "mouth_0" + str(last_image) + ".png"
            for i in range(len(d), 50):
                if i < 10:
                    img_name = "mouth_00" + str(i) + ".png"
                else:
                    img_name = "mouth_0" + str(i) + ".png"
                shutil.copyfile(os.path.join(p, last_image_name), os.path.join(p, img_name))




def re_align(path_to_dir):
    align_dir_path = '/home/atticus/lipNet_impl/uir2/training/unseen_speakers/datasets/re_align'
    old_align_dir_path = '/home/atticus/lipNet_impl/uir2/training/unseen_speakers/datasets/align'
    for p, w ,d in os.walk(path_to_dir):
        align_name = p.split('/')[-1]
        if (align_name == 'val'):
            continue
        frame_count = len(d)
        with open(os.path.join(old_align_dir_path, align_name + ".align"), "r") as al:
            info = al.read()
            word = info.split(" ")[-1]

        with open(os.path.join(align_dir_path, align_name) + ".align", "w") as f:
            f.write(START_ALIGN + " " + str(frame_count*1000) + " " + word)



if __name__ == '__main__':
    # dir_path = "/home/atticus/PycharmProjects/real-time-facial-landmarks/data_test"
    # files = os.listdir(dir_path)
    # dirs = filter(lambda file: re.search("^[^.]*$", file), files)
    # count = 2089
    # for directory in dirs:
    #     count = dir_processing(os.path.join(dir_path,directory), count, os.path.join(dir_path, 'align'))
    path_to_dir = '/home/atticus/lipNet_impl/uir2/training/unseen_speakers/datasets/val'
    #rm_not_existing_aligns(path_to_dir)
    #check_frame_count(path_to_dir)
    #re_align(path_to_dir)
    copy_frames_to_50_frames(path_to_dir)
# import cv2
from scipy.misc import imsave, imread
import pickle
from pathlib import Path
import numpy as np

def normalize(x):
    x = x.astype('float32')
    x /= 255.0
    return x


def load_normalize_and_save(num, input_data_dir= 'rawpic/', output_batch_path = 'batch_1.p' ):
    pic_normal_list = []
    pos_list = np.loadtxt(input_data_dir + '_pos.out')

    for i in range(num):
        file_num=str(i).zfill(5)
        f_name = input_data_dir + 'frame-' + str(file_num) +'.png'
        print('load ' + f_name)

        if not Path(f_name).exists():
            print(f_name + ' NOT EXIST')
            continue

        im = imread(f_name, mode='RGB') #cv2.imread(f_name)
        # print('im.shape = ' + str(im.shape))
        pic_normal_list.append(normalize(im))



    import time
    start = time.time()
    np_array_pic_normal = np.array(pic_normal_list)
    print( input_dir + " images load finish, please wait to save...")
    pickle.dump((np_array_pic_normal, pos_list), open(output_batch_path, 'wb'))
    print("{} use {} MB".format(output_batch_path,  file_size_MB(output_batch_path)))

def file_size_MB(f_path):
    size = Path(f_path).stat().st_size
    return (size/1000000)   # to MB

def test_load_normalize_and_save(batch_ind = 1):
    save_dir = 'test_normal/'
    pic_normal_list, pos_list = pickle.load(open('batch_'+ str(batch_ind)+'.p', 'rb'))

    for i, img in enumerate( pic_normal_list):
        img *= 255.0
        imsave(save_dir+ str(i) + '.png',img)
        #cv2.imwrite(save_dir+ str(i) + '.png',img)

    np.savetxt(save_dir + '_pos.out', pos_list, fmt='%3d') 



if __name__ == "__main__":
    num = 3000
    for i in range(10):
        input_dir = 'train_data_%02d' % i +'/'
        load_normalize_and_save(num, input_data_dir= input_dir, output_batch_path = 'batch_%02d.p' % i)
    # test_load_normalize_and_save()

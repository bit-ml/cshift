import os

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from tqdm import tqdm

import guided_filter
import network


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def my_cartoonize(bacth_imgs, model_path):
    input_photo = tf.placeholder(tf.float32, [None, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo,
                                            network_out,
                                            r=1,
                                            eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    bacth_imgs = bacth_imgs.astype(np.float32) / 127.5 - 1
    output = sess.run(final_out, feed_dict={input_photo: bacth_imgs})
    output = (np.squeeze(output) + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output


def cartoonize(load_folder, save_folder, model_path):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo,
                                            network_out,
                                            r=1,
                                            eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    name_list = os.listdir(load_folder)
    import pdb
    pdb.set_trace()
    for name in tqdm(name_list):
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            image = cv2.imread(load_path)
            image = resize_crop(image)
            batch_image = image.astype(np.float32) / 127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            output = sess.run(final_out, feed_dict={input_photo: batch_image})
            output = (np.squeeze(output) + 1) * 127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, output)
        except:
            print('cartoonize {} failed'.format(load_path))


if __name__ == '__main__':
    model_path = '/root/code/multi-domain-graph/experts/models'  #
    model_path = 'saved_models'
    img_path = './test_images/actress2.jpg'
    img_path_ = './test_images/china6.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_ = cv2.imread(img_path_)
    img_ = cv2.resize(img_, (256, 256), cv2.INTER_CUBIC)
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.concatenate((img[None], img_[None]), 0)
    import pdb
    pdb.set_trace()

    res = my_cartoonize(img, model_path)
    cv2.imwrite('t0.png', np.uint8(res[0]))
    cv2.imwrite('t1.png', np.uint8(res[1]))
    import pdb
    pdb.set_trace()
    '''
   
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    import pdb 
    pdb.set_trace()
    cartoonize(load_folder, save_folder, model_path)
    '''
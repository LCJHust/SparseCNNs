import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import os, sys
import numpy as np
import math
from datetime import datetime
import time
from PIL import Image
from math import ceil
from tensorflow.python.ops import gen_nn_ops
# modules
from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage
from Input_dataset import *
from ResNet50 import *
from SparseNet import *
import cv2 as cv

def compute_error(preds, labels):
    mask = tf.greater(tf.abs(labels), 0.0)
    residuals = tf.boolean_mask(labels - preds, mask)
    mae = tf.reduce_mean(tf.abs(residuals))
    return mae


def training(FLAGS, is_finetue=False):
    max_steps = FLAGS.max_steps
    batch_size = FLAGS.batch_size
    log_dir = FLAGS.log_dir
    train_dir = FLAGS.train_dir
    val_dir = FLAGS.val_dir
    # # finetune_ckpt = FLAGS.finetune
    image_w = FLAGS.image_w
    image_h = FLAGS.image_h
    image_c = FLAGS.image_c
    learning_rate = FLAGS.learning_rate
    max_to_keep = FLAGS.max_to_keep
    # should be changed if your model stored by different convention
    # startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-')[-1])


    startstep = 0
    INITIAL_LEARNING_RATE = 0.0002
    MOVING_AVERAGE_DECAY = 0.9999
    NUM_EPOCHS_PER_DECAY = 350.0
    LEARNING_RATE_DECAY_FACTOR = 0.1
    TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST / batch_size
    # max_to_keep = 50
    # learning_rate = 0.0002

    image_filenames, label_filenames = get_filename_list(train_dir)
    val_image_filenames, val_label_filenames = get_filename_list(val_dir)


    with tf.Graph().as_default():
        
        train_data_node = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, image_c])
        train_labels_node = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, 3])
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        global_step = tf.Variable(0, trainable=False)
        # For CamVid
        with tf.name_scope("data_loading"):
            images, labels = CamVidInputs(image_filenames, label_filenames, batch_size)
            val_images, val_labels = CamVidInputs(val_image_filenames, val_label_filenames, batch_size)
        # preds = build_resnet50(images, get_disp_resnet50, True, 'depth_net')
        
        with tf.name_scope("depth_completion"):
            preds, b_mask = build_sparsenet(labels)
            
        tf.summary.image('pred_img', preds)
     
        with tf.name_scope("compute_loss"):
            loss = compute_error(preds, labels)
        
#         loss = sparse_image_similarity_loss(preds, b_mask, labels  )
        tf.summary.scalar('loss', loss)
        merged_summary_op = tf.summary.merge_all()

        train_vars = [var for var in tf.trainable_variables()]
        vars_to_restore = slim.get_model_variables()

        optim = tf.train.AdamOptimizer(learning_rate, 0.9)  # Adam
        train_op = slim.learning.create_train_op(loss, optim,
                                             variables_to_train=train_vars)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        incr_global_step = tf.assign(global_step, global_step + 1)

        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in train_vars])

    # Saver
        saver = tf.train.Saver([var for var in tf.model_variables()] + \
                           [global_step],
                           max_to_keep=max_to_keep)

        sv = tf.train.Supervisor(logdir=log_dir,
                             save_summaries_secs=0,
                             saver=None)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7

        with sv.managed_session(config=config) as sess:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)  # added self tensorboard

            print('Trainable variables: ')
            for var in train_vars:
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))

            start_time = time.time()

            for step in range(1, max_steps):
                fetches = {
                    "train": train_op,
                    "global_step": global_step,
                    "incr_global_step": incr_global_step
                }

                if step % 10 == 0:
                    fetches["loss"] = loss
                results = sess.run(fetches)

                print(results)

                summary_str = sess.run(merged_summary_op)
                summary_writer.add_summary(summary_str, step)

                if step % 100 == 0:
                    time_per_iter = (time.time() - start_time) / 100
                    start_time = time.time()
                    print('Iteration: [%7d] | Time: %4.4fs/iter | Loss: %.3f' \
                          % (step, time_per_iter, results["loss"]))
                if step % 1000 == 0:
                    saver.save(sess, os.path.join(log_dir, 'model'), global_step=step)

def testing(FLAGS):
    image_w = FLAGS.image_w
    image_h = FLAGS.image_h
    image_c = FLAGS.image_c

    test_dir = FLAGS.test_dir
    test_ckpt = FLAGS.test_ckpt
    batch_size = 1

    if not os.path.exists(FLAGS.out_images):
        os.mkdir(FLAGS.out_images)

    image_filenames, label_filenames = get_filename_list(test_dir)

    test_data_node = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, image_c], name='raw_input')
    test_label_node = tf.placeholder(tf.float32, shape=[batch_size, image_h, image_w, image_c], name='sparse_points')

    model = build_resnet50(test_data_node, get_disp_resnet50, True, 'depth_net')

#     loss = compute_error(test_data_node, test_label_node, batch_size)


    phase_train = tf.placeholder(tf.bool, name='phase_train')

    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session() as sess:
        # Load checkpoint
        saver.restore(sess, test_ckpt)
        # Read test images and labels
        images, labels = get_all_test_data(image_filenames, label_filenames)

        pred_all = []


        inputs = np.zeros((batch_size, image_h, image_w, image_c), dtype=np.float)

        for image_batch, label_batch in zip(images, labels):
            feed_dict = {
                    test_data_node: image_batch,
                    phase_train: False
                }
            
            preds = sess.run(model, feed_dict=feed_dict)


#             loss = compute_error(preds, label_batch[:,:,:,:,0], batch_size)
#             print(loss)

            pred_all.append(preds)

#             hist += get_hist(preds, label_batch)
#             # count+=1
#             acc_total = np.diag(hist).sum() / hist.sum()
#             iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
#             print("acc: ", acc_total)
#             print("mean IU: ", np.nanmean(iu))

    np.save(FLAGS.out_images + '/depth_points', pred_all)



















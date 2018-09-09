import tensorflow as tf
import os
import model


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

FLAGS = tf.app.flags.FLAGS


## Mode selection
tf.app.flags.DEFINE_string('mode',                   "train",        "mode:[train, finetune or test]")

## Dataset dir & Log dir
tf.app.flags.DEFINE_string('log_dir',      "./Log_dir/0906/",        "dir to store ckpt")
tf.app.flags.DEFINE_string('train_dir',    "./val.dataset",        "path to training dataset")
tf.app.flags.DEFINE_string('val_dir',        "./val.dataset",        "path to validation dataset")
tf.app.flags.DEFINE_string('test_dir',          "./test.txt",        "path to testing dataset")

## Training settings
tf.app.flags.DEFINE_integer('batch_size',                "5",        "batch_size ")
tf.app.flags.DEFINE_float('learning_rate',          "0.0002",        "initial learning rate ")
tf.app.flags.DEFINE_integer('max_steps',             "20000",        "max steps")
tf.app.flags.DEFINE_integer('max_to_keep',              "10",        "max number of ckpt to keep")
tf.app.flags.DEFINE_integer('image_h',                 "352",        "image_height")
tf.app.flags.DEFINE_integer('image_w',                "1216",        "image_width")
tf.app.flags.DEFINE_integer('image_c',                   "1",        "image_channels(RGB)")

## Testing settings
tf.app.flags.DEFINE_string('test_ckpt',                   "",        "checkpoint file")
tf.app.flags.DEFINE_string('out_images',    "./pred_images/",        "dir to save predicted images")
tf.app.flags.DEFINE_boolean('save_images',             False,        "whether to save the predicted images")

def checkArgs():
    if FLAGS.mode == "train":
        print("The model is set to Training.")
        print("Max training Iteration: %d" % FLAGS.max_steps)
        print("Initial learning rate: %f" % FLAGS.learning_rate)
        print("Batch size: %d" % FLAGS.batch_size)

    print("Log_dir: %s" % FLAGS.log_dir)

def main(args):
    checkArgs()
    if FLAGS.mode == "train":
        model.training(FLAGS)
    elif FLAGS.mode == "test":
        model.testing(FLAGS)

if __name__ == '__main__':
    tf.app.run()
    


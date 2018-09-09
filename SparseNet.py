import tensorflow as tf
import tensorflow.contrib.slim as slim

alpha_recon_image = 0.85

def sparse_conv(tensor,binary_mask = None,filters=16,kernel_size=11,strides=1,l2_scale=0.0, activation=tf.nn.relu):
    """
    Arguments
        tensor: Tensor input.
        binary_mask: Tensor, a mask with the same size as tensor, channel size = 1
        filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width.
        l2_scale: float, A scalar multiplier Tensor. 0.0 disables the regularizer.
    Returns:
    Output tensor, binary mask.
 """
    if binary_mask == None: #first layer has no binary mask
        b,h,w,c = tensor.get_shape()       # batch_size, height, width, channels
        channels=tf.split(tensor,c,axis=3)   # 
        #assume that if one channel has no information, all channels have no information
        binary_mask = tf.where(tf.equal(channels[0], 0), tf.zeros_like(channels[0]), tf.ones_like(channels[0])) #mask should only have the size of (B,H,W,1)

    features = tf.multiply(tensor,binary_mask)
    #features = tf.layers.conv2d(features, filters=filters, kernel_size=kernel_size, strides=(strides, strides), trainable=True, use_bias=False, padding="same",kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale))
    features = tf.layers.conv2d(features, filters=filters, kernel_size=kernel_size, strides=(strides, strides),\
                                trainable=True, use_bias=False, padding="same",kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_scale), activation=activation)   # changed activation='relu'

    norm = tf.layers.conv2d(binary_mask, filters=filters,kernel_size=kernel_size,strides=(strides, strides),kernel_initializer=tf.ones_initializer(),trainable=False,use_bias=False,padding="same")
    norm = tf.where(tf.equal(norm,0),tf.zeros_like(norm),tf.reciprocal(norm))
    _,_,_,bias_size = norm.get_shape()

    b = tf.Variable(tf.constant(0.0, shape=[bias_size]),trainable=True)
    feature = tf.multiply(features,norm)+b
    mask = tf.layers.max_pooling2d(binary_mask,strides = strides,pool_size=3,padding="same")

    return feature,mask

def build_sparsenet(input_img):
    # input              -->   features_1, mask_1
    features_1, mask_1 = sparse_conv(input_img,  binary_mask=None,  filters=16, kernel_size=11, strides=1)
    # features_1, mask_1 -->   features_2, mask_2
    features_2, mask_2 = sparse_conv(features_1, binary_mask=mask_1, filters=16, kernel_size=7, strides=1)
    # features_2, mask_2 -->   features_3, mask_3
    features_3, mask_3 = sparse_conv(features_2, binary_mask=mask_2, filters=16, kernel_size=5, strides=1)
    # features_3, mask_3 -->   features_4, mask_4
    features_4, mask_4 = sparse_conv(features_3, binary_mask=mask_3, filters=16, kernel_size=3, strides=1)
    # features_4, mask_4 -->   features_5, mask_5
    features_5, mask_5 = sparse_conv(features_4, binary_mask=mask_4, filters=16, kernel_size=3, strides=1)

    preds = tf.multiply(features_5, mask_5) + 0.001
    preds = tf.reduce_mean(preds, axis=3, keep_dims=True)
   
    return preds, mask_5

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
    mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')
        
    sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
    sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d
    
    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def image_similarity(x, y):
    return alpha_recon_image * SSIM(x, y) + (1-alpha_recon_image) * tf.abs(x-y) 

def sparse_image_similarity_loss(pred, mask, labels):
    sparse_pred = tf.multiply(pred, mask)
    depths = tf.reduce_sum(labels+ 0.01*labels + 0.0001*labels,axis=3,keep_dims=True)
    return tf.reduce_mean(image_similarity(sparse_pred, depths))

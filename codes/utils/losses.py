import tensorflow as tf
from keras import backend as K
from tensorflow.python.ops import array_ops

def generalised_dice_coef(y_true, y_pred, type_weight='Square'):
    """
    It computes the generalised dice coefficient
    :param y_true: true labels (ground truth)
    :param y_pred: predicted labels
    :return: generalised dice coefficient score between y_true and y_pred
    """
    prediction = tf.cast(y_pred, tf.float32)

    ref_vol = tf.reduce_sum(y_true, axis=0)
    intersect = tf.reduce_sum(y_true * prediction, axis=0)
    seg_vol = tf.reduce_sum(prediction, axis = 0)

    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))

    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *tf.reduce_max(new_weights), weights)

    generalised_dice_numerator = 2 * tf.reduce_sum(tf.multiply(weights, intersect))
    generalised_dice_denominator = tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_score = generalised_dice_numerator / generalised_dice_denominator

    return generalised_dice_score

def gen_dice_multilabel(y_true, y_pred, numLabels=5):
    """
    It computes the generalised dice coefficient loss making an average for each class (binary case)
    for a multi-class problem with numLabels classes
    :param y_true: true labels (ground truth)
    :param y_pred: predicted labels
    :param numLabels: number of classes
    :return: dice coefficient loss for a multi-class problem
    """
    dice = 0
    for index in range(numLabels):
        dice += generalised_dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index], type_weight='Square')
    return 1. - dice/5.


def focal_loss_softmax(labels,logits,gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.softmax(logits,dim=-1) # [batch_size,num_classes]
    labels=tf.one_hot(labels,depth=y_pred.shape[1])
    L=-labels*((1-y_pred)**gamma)*tf.log(y_pred)
    L=tf.reduce_sum(L,axis=1)
    return L

def seg_loss(y_true, y_pred, loss_weight_divisor=None):
    y_true_flat = K.batch_flatten(y_true)
    y_pred_flat = K.batch_flatten(y_pred)
    y_pred_flat = K.clip(y_pred_flat, K.epsilon(), 1 - K.epsilon())

    weight1 = K.sum(K.cast(K.equal(y_true_flat, 1), tf.float32))
    weight1 = K.maximum(weight1, 1)  # prevent division by zero
    weight0 = K.sum(K.cast(K.equal(y_true_flat, 0), tf.float32))
     
    weighted_binary_crossentropy = -K.mean(y_true_flat * K.log(y_pred_flat) * weight0 / (weight1 * loss_weight_divisor)
                                           + (1 - y_true_flat) * K.log(1 - y_pred_flat), axis=-1)
    return weighted_binary_crossentropy












def dice_coef(y_true, y_pred, smooth = 1.):
    """
    It computes the dice coefficient
    :param y_true: true labels (ground truth)
    :param y_pred: predicted labels
    :param smooth: parameter to ensure stability
    :return: dice coefficient score between y_true and y_pred
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, numLabels=5):
    """
    It computes the dice coefficient loss making an average for each class (binary case)
    for a multi-class problem with numLabels classes
    :param y_true: true labels (ground truth)
    :param y_pred: predicted labels
    :param numLabels: number of classes
    :return: dice coefficient loss for a multi-class problem
    """
    dice = 0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return 1. - dice/5.



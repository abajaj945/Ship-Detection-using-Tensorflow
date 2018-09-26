import glob
import os
import utils_box as box
import tensorflow as tf
import cv2 
import numpy
import json
from collections import namedtuple
import model_layers as layer
from tensorflow.python.platform import tf_logging as logging
import numpy as np
import imgdbg as imgdbg
import math
import settings

def learn_rate_decay(step, params):
    """ Model building utility function. Learning rate decay parametrized from
    command-line parameters lr0, lr1 and lr2."""
    if params['decay_type'] == "exponential":
        lr = params['lr1'] + tf.train.exponential_decay(params['lr0'], step, params['lr2'], 1/math.e)
    elif params['decay_type'] == "cosine-restarts":
        # empirically  determined t_mul rates for cosine_restarts. With these rates, learning rate is
        # guaranteed to decay to its min value by the end of "iterations" with "decay-restarts" restarts
        # and a first restart at "iterations" / 8.
        t_muls = [1.0, 7.0, 2.1926, 1.48831, 1.23692, 1.11434, 1.04422]
        t_mul = t_muls[params["decay_restarts"]]
        m_mul = params["decay_restart_height"]
        first_decay_steps = params["iterations"] // 8 if params["decay_restarts"] > 0 else params["iterations"]
        lr = params['lr1'] + tf.train.cosine_decay_restarts(params['lr0'], step, first_decay_steps, t_mul, m_mul)
    return lr


def extract_filename_without_extension(filename):
    basename = os.path.basename(filename)
    barename, extension = os.path.splitext(basename)
    return barename, filename

def load_file_list(directory):
    # load images, load jsons, associate them by name, XYZ.jpg with XYZ.json
    img_files1 = glob.glob(directory + "/*.jpg")
    img_files2 = glob.glob(directory + "/*.jpeg")
    img_files  = img_files1 + img_files2
    roi_files  = glob.glob(directory + "/*.json")
    img_kv = list(map(extract_filename_without_extension, img_files))
    roi_kv = list(map(extract_filename_without_extension, roi_files))
    all_kv = img_kv + roi_kv
    img_dict = dict(img_kv)
    roi_dict = dict(roi_kv)
    all_dict = dict(all_kv)
    outer_join = [(img_dict[k] if k in img_dict else None,
                   roi_dict[k] if k in roi_dict else None) for k in all_dict]
    # keep only those where the jpg and the json are both available
    inner_join = list(filter(lambda e: e[0] is not None and e[1] is not None, outer_join))
    if len(inner_join) == 0:
        return [], []
    else:
        img_list, roi_list = zip(*inner_join)  # unzip, results are a tuple of img names and a tuple of roi names
        return list(img_list), list(roi_list)   
def load_img_and_json_files(img_filename, roi_filename):
    img_bytes = tf.read_file(img_filename)
    json_bytes = tf.read_file(roi_filename)
    pixels, rois = decode_image_and_json_bytes(img_bytes, json_bytes)
    return pixels, rois, img_filename

def decode_json_py(str):
    obj = json.loads(str.decode('utf-8'))
    rois = np.array([(roi['x1'], roi['y1'], roi['x2'], roi['y2']) for roi in obj["rois"]], dtype=np.float32)
    return rois


def decode_image(img_bytes):
    pixels = tf.image.decode_image(img_bytes, channels=3)
    return tf.cast(pixels, tf.uint8)


def decode_image_and_json_bytes(img_bytes, json_bytes):
    # decode jpeg
    pixels = decode_image(img_bytes)
    # parse json
    rois = tf.py_func(decode_json_py, [json_bytes], [tf.float32])
    rois = tf.reshape(rois[0], [-1, 4])
    return pixels, rois

def yolo_roi_attribution(tile, rois, yolo_cfg):
    # Tile divided in grid_nn x grid_nn grid
    # Recognizing cell_n boxes per grid cell
    # For each tile, for each grid cell, determine the cell_n largest ROIs centered in that cell
    # Output shape [tiles_n, grid_nn, grid_nn, cell_n, 3] 3 for x, y, w

    # dynamic number of rois
    rois = tf.reshape(rois, [-1, 4])  # I know the shape but Tensorflow does not
    rois_n = tf.shape(rois)[0]  # known shape [n, 4]

    if yolo_cfg.cell_n == 2 and yolo_cfg.cell_swarm:
        yolo_target_rois = box.n_experimental_roi_selection_strategy(tile, rois, rois_n,
                                                                     yolo_cfg.grid_nn,
                                                                     yolo_cfg.cell_n,
                                                                     yolo_cfg.cell_grow)
    elif not yolo_cfg.cell_swarm:
        yolo_target_rois = box.n_largest_rois_in_cell_relative(tile, rois, rois_n,
                                                               yolo_cfg.grid_nn,
                                                               yolo_cfg.cell_n)
    else:
        raise ValueError('Ground truth ROI selection strategy cell_swarm is only implemented for cell_n=2')

    # maybe not needed
    yolo_target_rois = tf.reshape(yolo_target_rois, [yolo_cfg.grid_nn,
                                                     yolo_cfg.grid_nn,
                                                     yolo_cfg.cell_n, 4])  # 4 for x, y, w, h

    return yolo_target_rois

def features_and_labels(image, yolo_target_rois, target_rois, fname):
    features = {'image': image}
    labels = {'yolo_target_rois': yolo_target_rois, 'target_rois': target_rois, 'fname': fname}
    return features, labels

def generate_slice(pixels, rois, fname, yolo_cfg, rnd_hue, rnd_orientation):
    # dynamic image shapes
    img_shape = tf.cast(tf.shape(pixels), tf.float32)  # known shape [height, width, 4]
    img_shape = tf.reshape(img_shape, [3])  # tensorflow needs help here
    img_h, img_w, _ = tf.unstack(img_shape)

    # dynamic number of rois
    rois = tf.reshape(rois, [-1, 4])  # I know the shape but Tensorflow does not
    rois_n = tf.shape(rois)[0] # known shape [n, 4]

    target_rois = box.rois_in_image_relative(pixels, rois, img_h, img_w, settings.max_rois)  # shape [rois_n, 4] convert in range(0,1)

    
    if rnd_orientation:
        pixels, target_rois = box.random_orientation(pixels, target_rois, 1.0)

    # Compute ground truth ROIs assigned to YOLO grid cells
    tile=tf.constant([0.0,0.0,1.0,1.0])
    yolo_target_rois = yolo_roi_attribution(tile, target_rois, yolo_cfg)

    if rnd_hue:  # random hue shift for all training images
        image_tiles = random_hue(image_tiles)


    features, labels = features_and_labels(pixels, yolo_target_rois, target_rois, fname)
    return features, labels


def train_input_fn(directory,batch_size,yolo_cfg):
    img_filelist, roi_filelist = load_file_list(directory)

    fileset = tf.data.Dataset.from_tensor_slices((tf.constant(img_filelist), tf.constant(roi_filelist)))
    fileset = fileset.shuffle(1000)  # shuffle filenames
    dataset = fileset.map(load_img_and_json_files)
    dataset = dataset.map(lambda pix, rois, fname: generate_slice(pix, rois, fname,
                                                                 yolo_cfg=yolo_cfg,
                                                                 rnd_hue=hparams['data_rnd_hue'],
                                                                 rnd_orientation=hparams['data_rnd_orientation']))
    
    
    dataset=dataset.batch(batch_size).repeat()
    iterator=dataset.make_one_shot_iterator()
    features,labels=iterator.get_next()
    
    return features,labels



def eval_input_fn(directory,batch_size,yolo_cfg):
    img_filelist, roi_filelist = load_file_list(directory)

    fileset = tf.data.Dataset.from_tensor_slices((tf.constant(img_filelist[:20]), tf.constant(roi_filelist[:20])))
    fileset = fileset.shuffle(1000)  # shuffle filenames
    dataset = fileset.map(load_img_and_json_files)
    dataset = dataset.map(lambda pix, rois, fname: generate_slice(pix, rois, fname,
                                                                 yolo_cfg=yolo_cfg,
                                                                 rnd_hue=False,
                                                                 rnd_orientation=False))
    
    
    dataset=dataset.batch(batch_size)
    iterator=dataset.make_one_shot_iterator()
    features,labels=iterator.get_next()
    return features,labels


def model_core_squeezenet12(x, mode, params, info):
    y, info = layer.conv2d_batch_norm_relu_dropout_l(x, mode, params, info, filters=32, kernel_size=6, strides=2)  # output 128x128
    y, info = layer.maxpool_l(y, info)  # output 64x64
    y, info = layer.sqnet_squeeze(y, mode, params, info, 21)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*26)
    y, info = layer.sqnet_squeeze(y, mode, params, info, 36)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*46)
    y, info = layer.maxpool_l(y, info)  # output 32x32
    y, info = layer.sqnet_squeeze(y, mode, params, info, 41)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*36)
    y, info = layer.sqnet_squeeze(y, mode, params, info, 31)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*26)
    y, info = layer.maxpool_l(y, info)  # output 16x16
    y, info = layer.sqnet_squeeze(y, mode, params, info, 21)
    y, info = layer.sqnet_expand(y, mode, params, info, 2*16, last=True)
    return y, info



def model_fn(features, labels, mode, params):
    """The model, with loss, metrics and debug summaries"""

    # YOLO parameters
    grid_nn = params["grid_nn"]  # each tile is divided into a grid_nn x grid_nn grid
    cell_n = params["cell_n"]  # each grid cell predicts cell_n bounding boxes.
    info = None

    # model inputs
    X = tf.to_float(features["image"]) / 255.0 # input image format is uint8 with range 0 to 255
    
    X=tf.reshape(X,[-1,768,768,3])

    # The model itself is here
    #Y, info = model_core_squeezenet12(X, mode, params, info)
    #Y, info = model_core_squeezenet17(X, mode, params, info)
    #Y, info = model_core_darknet(X, mode, params, info)
    #Y, info = model_core_darknet17(X, mode, params, info)
    Y, info = model_core_squeezenet12(X, mode, params,info)
    logging.debug(X.shape)
    # YOLO head: predicts bounding boxes around ships
    box_x, box_y, box_w, box_h, box_c, box_c_logits, info = layer.YOLO_head(Y, mode, params, info, grid_nn, cell_n)

    # Debug: print the model structure
    if mode == tf.estimator.ModeKeys.TRAIN:
        logging.log(logging.INFO, info["description"])
        logging.log(logging.INFO, "NN {} layers / {:,d} total weights".format(info["layers"], info["weights"]))

    box_c_sim = box_c[:,:,:,:,1]  # shape [batch, GRID_N,GRID_N,CELL_B]
    DETECTION_TRESHOLD = 0.5  # ship "detected" if predicted C>0.5
    detected_w = tf.where(tf.greater(box_c_sim, DETECTION_TRESHOLD), box_w, tf.zeros_like(box_w))
    detected_h = tf.where(tf.greater(box_c_sim, DETECTION_TRESHOLD), box_h, tf.zeros_like(box_w))
    
    # all rois with confidence factors
    predicted_rois = tf.stack([box_x, box_y, box_w, box_h], axis=-1)  # shape [batch, GRID_N, GRID_N, CELL_B, 4]
    predicted_rois = box.grid_cell_to_tile_coords(predicted_rois, grid_nn, 768) / 768
    predicted_rois = tf.reshape(predicted_rois, [-1, grid_nn*grid_nn*cell_n, 4])
    predicted_c = tf.reshape(box_c_sim, [-1, grid_nn*grid_nn*cell_n])
    # only the rois where a ship was detected
    detected_rois = tf.stack([box_x, box_y, detected_w, detected_h], axis=-1)  # shape [batch, GRID_N, GRID_N, CELL_B, 4]
    detected_rois = box.grid_cell_to_tile_coords(detected_rois, grid_nn, 768) / 768
    detected_rois = tf.reshape(detected_rois, [-1, grid_nn*grid_nn*cell_n, 4])
    detected_rois, detected_rois_overflow = box.remove_empty_rois(detected_rois, 50)

    loss = train_op = eval_metrics = None
    if mode != tf.estimator.ModeKeys.PREDICT:

        # Target labels
        # Ground truth boxes. Used to compute IOU accuracy and display debug ground truth boxes.
        target_rois = labels["target_rois"] # shape [batch, MAX_TARGET_ROIS_PER_TILE, x1y1x2y2]
        # Ground truth boxes assigned to YOLO grid cells. Used to compute loss.
        target_rois_yolo = labels["yolo_target_rois"]  # shape [4,4,3,3] = [batch, GRID_N, GRID_N, CEL_B, xywh]
        target_x, target_y, target_w, target_h = tf.unstack(target_rois_yolo, axis=-1) # shape 3 x [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]
        # target probability is 1 if there is a corresponding target box, 0 otherwise
        target_is_ship = tf.greater(target_w, 0.0001)
        target_is_ship_onehot = tf.one_hot(tf.cast(target_is_ship, tf.int32), 2, dtype=tf.float32)
        target_is_ship_float = tf.cast(target_is_ship, tf.float32) # shape [batch, 4,4,3] = [batch, GRID_N, GRID_N, CELL_B]

        # Mistakes and correct detections for visualisation and debugging.
        # This is computed against the ground truth boxes assigned to YOLO grid cells.
        mistakes, size_correct, position_correct, all_correct = box.compute_mistakes(box_x, box_y,
                                                                                     box_w, box_h, box_c_sim,
                                                                                     target_x, target_y,
                                                                                     target_w, target_h, target_is_ship, grid_nn)
        
        
        debug_img = imgdbg.debug_image(X, mistakes, target_rois, predicted_rois, predicted_c,
                                       size_correct, position_correct, all_correct,
                                       grid_nn, cell_n, 768)

        
        if mode == tf.estimator.ModeKeys.EVAL:
            iou_accuracy = box.compute_safe_IOU(target_rois, detected_rois, detected_rois_overflow, 768)

        
        # IOU (Intersection Over Union) accuracy
        # IOU computation removed from training mode because it used an op not yet supported with MirroredStrategy
        
        # Improvement ideas and experiment results
        # 1) YOLO trick: take square root of predicted size for loss so as not to drown errors on small boxes: tested, no benefit
        # 2) if only one ship in cell, teach all cell_n detectors to detect it: implemented in box.n_experimental_roi_selection_strategy, beneficial
        # 3) TODO: try two or more grids, shifted by 1/2 cell size: This could make it easier to have cells detect ships in their center, if that is an actual problem they have (no idea)
        # 4) try using TC instead of TC_ in position loss and size loss: tested, no benefit
        # 5) TODO: one run without batch norm for comparison
        # 6) TODO: add dropout, tested, weird resukts: eval accuracy goes up signicantly but model performs worse in real life. Probably not enough training data.
        # 7) TODO: idea, compute detection box loss agains all ROI, not just assigned ROIs: if neighboring cell detects something that aligns well with ground truth, no reason to penalise
        # 8) TODO: add tile rotations, tile color inversion (data augmentation)

        # Loss function
        logging.log(logging.INFO,Y)
        logging.log(logging.INFO,box_x)
        logging.log(logging.INFO,box_y)
        logging.log(logging.INFO,box_w)
        logging.log(logging.INFO,box_h)
        logging.log(logging.INFO,box_c)
        logging.log(logging.INFO,box_c_logits)
        position_loss = tf.reduce_mean(target_is_ship_float * (tf.square(box_x - target_x) + tf.square(box_y - target_y)))
        size_loss = tf.reduce_mean(target_is_ship_float * tf.square(box_w - target_w) * 2 + target_is_ship_float * tf.square(box_h - target_h) * 2)
        obj_loss = tf.losses.softmax_cross_entropy(target_is_ship_onehot, box_c_logits)

        # YOLO trick: weights the different losses differently
        loss_weight_total = (params['lw1'] + params['lw2'] + params['lw3']) * 1.0  # 1.0 to force conversion to float
        w_obj_loss = obj_loss*(params['lw1'] / loss_weight_total)
        w_position_loss = position_loss*(params['lw2'] / loss_weight_total)
        w_size_loss = size_loss*(params['lw3'] / loss_weight_total)
        loss = w_position_loss + w_size_loss + w_obj_loss
        nb_mistakes = tf.reduce_sum(mistakes)

        # average number of mistakes per image
        
        lr = learn_rate_decay(tf.train.get_or_create_global_step(), params)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = tf.contrib.training.create_train_op(loss, optimizer)

        if mode == tf.estimator.ModeKeys.EVAL:
            # metrics removed from training mode because they are not yet supported with MirroredStrategy
            eval_metrics = {"position_error": tf.metrics.mean(w_position_loss),
                            "size_error": tf.metrics.mean(w_size_loss),
                            "ship_cross_entropy_error": tf.metrics.mean(w_obj_loss),
                            "mistakes": tf.metrics.mean(nb_mistakes),
                            'IOU': tf.metrics.mean(iou_accuracy)
                            }
                            
        else:
            eval_metrics = None


        # Tensorboard summaries for debugging
        tf.summary.scalar("position_error", w_position_loss)
        tf.summary.scalar("size_error", w_size_loss)
        tf.summary.scalar("ship_cross_entropy_error", w_obj_loss)
        tf.summary.scalar("loss", loss)
        tf.summary.image("input_image", debug_img, max_outputs=20)
        tf.summary.scalar("learning_rate", lr)
        
        # a summary on iou_accuracy would be nice but it goes Out Of Memory

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"rois":predicted_rois, "rois_confidence": predicted_c},  # name these fields as you like
        loss=loss, train_op=train_op, eval_metric_ops=eval_metrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput({"rois": predicted_rois, 
                                                                      "rois_confidence": predicted_c})}  # TODO: remove legacy C
)


YOLOConfig = namedtuple('yolocfg', 'grid_nn cell_n cell_swarm cell_grow')
yolo_cfg = YOLOConfig(grid_nn =48, cell_n = 2, cell_swarm = True, cell_grow = 1.0)
tfrec_filelist='/home/bajaj94500_gmail_com/filesandjson'
train_data_input_fn = lambda: my_input_fn(tfrec_filelist,
                                          hparams["batch_size"],
                                          yolo_cfg)             

eval_yolo_cfg = YOLOConfig(hparams["grid_nn"], hparams["cell_n"], hparams["cell_swarm"], 1.0)
tfrec_filelist_eval='/home/bajaj94500_gmail_com/filesandjson'
eval_data_input_fn = lambda: eval_input_fn(tfrec_filelist_eval,
                                          hparams["eval_batch_size"],
                                          eval_yolo_cfg)   


hparams={'data_rnd_orientation': False,
         'lw2': 3,
         'decay_type': 'exponential',
         'cell_swarm': True,
         'decay_restarts': 3,
         'depth_increment': 5,
         'cell_grow': 1.3,
         'lw1': 1,
         'grid_nn': 48,
         'eval_iterations': 1,
         'eval_batch_size': 10,
         'first_layer_filter_stride': 2,
         'lr1': 0.0001,
         'decay_restart_height': 0.99,
         'layers': 12,
         'lr2': 3000,
         'dropout': 0.0,
         'spatial_dropout': True,
         'shuffle_buf': 10000,
         'first_layer_filter_size': 6,
         'iterations': 25000,
         'data_cache_n_epochs': 0,
         'lw3': 30,
         'data_rnd_hue': False,
         'data_rnd_orientation':True,
         'data_rnd_distmax': 2.0,
         'cell_n': 2,
         'bnexp': 0.993,
         'first_layer_filter_depth': 32,
         'lr0': 0.01,
         'batch_size': 10}


estimator = tf.estimator.Estimator(model_fn=model_fn,
                                    model_dir='first_1',
                                    params=hparams)


estimator.train(input_fn=train_data_input_fn,max_steps=hparams["iterations"])

a=estimator.evaluate(input_fn=eval_data_input_fn)

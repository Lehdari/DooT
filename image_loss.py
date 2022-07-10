import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.image import gaussian_filter2d


class ImageLoss:
    def __init__ (self, image, image_pred,
        weight_matrix=[
            # columns: lumacity, chromacity Cb, chromacity Cr, depth
            [1.0, 0.5, 0.5, 1.0], # original image
            [2.0, 1.0, 1.0, 4.0], # gradient 1
            [2.0, 1.0, 1.0, 4.0], # gradient 2
            [2.0, 1.0, 1.0, 4.0], # gradient 4
            [2.0, 1.0, 1.0, 4.0], # gradient 8
            [2.0, 1.0, 1.0, 4.0], # gradient 16
            [4.0, 2.0, 2.0, 2.0], # laplacian 1
            [4.0, 2.0, 2.0, 2.0], # laplacian 2
            [4.0, 2.0, 2.0, 2.0], # laplacian 4
            [4.0, 2.0, 2.0, 2.0], # laplacian 8
            [4.0, 2.0, 2.0, 2.0], # laplacian 16
        ],
        mask_hud=True):

        self.image = image
        self.image_pred = image_pred
        self.loss_mask = tf.ones_like(image)
        if mask_hud:
            self.loss_mask = tf.cast(tf.where(image[:,:,:,3:4] > 0.001,
                tf.ones_like(image[:,:,:,3:4]),
                tf.zeros_like(image[:,:,:,3:4])), tf.float32)

        self.weight_matrix = tf.constant(weight_matrix)

        self.n_levels = 5

        self.images_blurred = tf.stack([
            gaussian_filter2d(image, sigma=2.0**i, filter_shape=[int((2**i)*4+1), int((2**i)*4+1)],
                padding="REFLECT")
            for i in range(self.n_levels)
        ], axis=0)
        self.image_preds_blurred = tf.stack([
            gaussian_filter2d(image_pred, sigma=2.0**i, filter_shape=[int((2**i)*4+1), int((2**i)*4+1)],
                padding="REFLECT")
            for i in range(self.n_levels)
        ], axis=0)

        self.losses = tf.convert_to_tensor(
            [ImageLoss.image_loss(self.image, self.image_pred, self.loss_mask)] +\
            [self.loss_gradient_level(i) for i in range(5)] +\
            [self.loss_laplacian_level(i) for i in range(5)]
        )

    def loss_gradient_level(self, level):
        if level >= self.n_levels:
            return [0.0, 0.0, 0.0, 0.0]
        else:
            return ImageLoss.image_loss_gradient(
                self.images_blurred[level], self.image_preds_blurred[level], self.loss_mask, 2**level)
    
    def loss_laplacian_level(self, level):
        if level >= self.n_levels:
            return [0.0, 0.0, 0.0, 0.0]
        else:
            if level == 0:
                return ImageLoss.image_loss(
                    self.images_blurred[0]-self.image,
                    self.image_preds_blurred[0]-self.image_pred, self.loss_mask)
            else:
                return ImageLoss.image_loss(
                    self.images_blurred[level]-self.images_blurred[level-1],
                    self.image_preds_blurred[level]-self.image_preds_blurred[level-1], self.loss_mask)

    @staticmethod
    def image_loss(y_true, y_pred, mask):
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred)*mask, axis=[0,1,2])
        l2 = tf.reduce_mean(tf.square(y_true - y_pred)*mask, axis=[0,1,2])
        
        return l1 + 1.5*l2
    
    @staticmethod
    def image_loss_gradient(y_true, y_pred, mask, stride):
        return ImageLoss.image_loss(
            ImageLoss.image_x_gradient(y_true, stride),
            ImageLoss.image_x_gradient(y_pred, stride),
            ImageLoss.gradient_x_mask(mask, stride)) +\
        ImageLoss.image_loss(
            ImageLoss.image_y_gradient(y_true, stride),
            ImageLoss.image_y_gradient(y_pred, stride),
            ImageLoss.gradient_y_mask(mask, stride))

    def total_loss(self):
        return tf.reduce_sum(tf.math.multiply(self.losses, self.weight_matrix))
    
    @staticmethod
    def image_x_gradient(image, stride):
        x_grad = image[:,stride:,:,:] - image[:,:-stride,:,:]
        return x_grad

    @staticmethod
    def image_y_gradient(image, stride):
        y_grad = image[:,:,stride:,:] - image[:,:,:-stride,:]
        return y_grad

    @staticmethod
    def gradient_x_mask(mask, stride):
        x_mask_out = tf.math.minimum(mask[:,stride:,:,:], mask[:,:-stride,:,:])
        return x_mask_out

    @staticmethod
    def gradient_y_mask(mask, stride):
        y_mask_out = tf.math.minimum(mask[:,:,stride:,:], mask[:,:,:-stride,:])
        return y_mask_out

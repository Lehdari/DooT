import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.image import gaussian_filter2d


class ImageLoss:
    def __init__ (self, image, image_pred, loss_mask=None, depth_channel=3,
        depth_weight=1.0, laplacian_weight=5.0):
        self.loss_mask = loss_mask
        if self.loss_mask is None:
            self.loss_mask = tf.ones_like(image)

        self.depth_channel = depth_channel
        self.depth_weight = depth_weight
        depth_channel_mask_np = np.ones((image.shape[3]), dtype=np.float32)
        depth_channel_mask_np[depth_channel] = depth_weight
        self.depth_channel_mask = tf.convert_to_tensor(depth_channel_mask_np)
        self.laplacian_weight = laplacian_weight

        self.images_blurred = [image]
        self.images_laplacian = []
        self.image_preds_blurred = [image_pred]
        self.image_preds_laplacian = []

        self.losses_laplacian = []
        
        self.loss_image = self.image_loss(image, image_pred, loss_mask)

        for i in range(5):
            blur = 2.0**i
            kernel_size = int(blur*4+1)

            self.images_blurred.append(gaussian_filter2d(image,
                filter_shape=[kernel_size, kernel_size], sigma=blur, padding="REFLECT"))
            self.images_laplacian.append(self.images_blurred[-1]-self.images_blurred[-2])
            self.image_preds_blurred.append(gaussian_filter2d(image_pred,
                filter_shape=[kernel_size, kernel_size], sigma=blur, padding="REFLECT"))
            self.image_preds_laplacian.append(self.image_preds_blurred[-1]-self.image_preds_blurred[-2])

            self.losses_laplacian.append(self.laplacian_weight * self.image_loss(
                self.images_laplacian[-1], self.image_preds_laplacian[-1], self.loss_mask))
        
    def image_loss(self, y_true, y_pred, mask):
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred)*mask, axis=[0,1,2])
        l2 = tf.reduce_mean(tf.square(y_true - y_pred)*mask, axis=[0,1,2])
        
        return tf.reduce_mean((l1 + 1.5*l2)*self.depth_channel_mask)

    def total_loss(self):
        total_loss = self.loss_image
        for l in self.losses_laplacian:
            total_loss += l
        return total_loss
    
    @staticmethod
    def image_gradient(image):
        x_grad = image[:,1:,:,:] - image[:,:-1,:,:]
        y_grad = image[:,:,1:,:] - image[:,:,:-1,:]
        return x_grad, y_grad

    @staticmethod
    def gradient_masks(mask):
        x_mask_out = tf.math.minimum(mask[:,1:,:,:], mask[:,:-1,:,:])
        y_mask_out = tf.math.minimum(mask[:,:,1:,:], mask[:,:,:-1,:])
        return x_mask_out, y_mask_out

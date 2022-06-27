import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class ImageLoss:
    def __init__ (self, image, image_pred, loss_mask=None, depth_channel=3, depth_weight=1.0):
        if loss_mask is None:
            loss_mask = tf.ones_like(image)

        self.depth_channel = depth_channel
        self.depth_weight = depth_weight
        depth_channel_mask_np = np.ones((image.shape[3]), dtype=np.float32)
        depth_channel_mask_np[depth_channel] = depth_weight
        self.depth_channel_mask = tf.convert_to_tensor(depth_channel_mask_np)

        self.grad_x = []
        self.grad_y = []
        self.grad_x_pred = []
        self.grad_y_pred = []
        self.grad_x_mask = []
        self.grad_y_mask = []
        self.loss_grad = []

        grad_x, grad_y = ImageLoss.image_gradient(image)
        grad_x_pred, grad_y_pred = ImageLoss.image_gradient(image_pred)
        grad_x_mask, grad_y_mask = ImageLoss.gradient_masks(loss_mask)
        loss_grad =\
            self.image_loss(grad_x, grad_x_pred, grad_x_mask) +\
            self.image_loss(grad_y, grad_y_pred, grad_y_mask)
        
        self.grad_x.append(grad_x)
        self.grad_y.append(grad_y)
        self.grad_x_pred.append(grad_x_pred)
        self.grad_y_pred.append(grad_y_pred)
        self.grad_x_mask.append(grad_x_mask)
        self.grad_y_mask.append(grad_y_mask)
        self.loss_grad.append(loss_grad)

        self.loss_image = self.image_loss(image, image_pred, loss_mask)

        self.images = [image]
        self.image_preds = [image_pred]
        self.loss_masks = [loss_mask]
        for i in range(4):
            self.images.append(layers.AveragePooling2D((2,2))(self.images[-1]))
            self.image_preds.append(layers.AveragePooling2D((2,2))(self.image_preds[-1]))
            self.loss_masks.append(layers.AveragePooling2D((2,2))(self.loss_masks[-1]))
            grad_x, grad_y = ImageLoss.image_gradient(self.images[-1])
            grad_x_pred, grad_y_pred = ImageLoss.image_gradient(self.image_preds[-1])
            grad_x_mask, grad_y_mask = ImageLoss.gradient_masks(self.loss_masks[-1])
            loss_grad =\
                self.image_loss(grad_x, grad_x_pred, grad_x_mask) +\
                self.image_loss(grad_y, grad_y_pred, grad_y_mask)
            
            self.grad_x.append(grad_x)
            self.grad_y.append(grad_y)
            self.grad_x_pred.append(grad_x_pred)
            self.grad_y_pred.append(grad_y_pred)
            self.grad_x_mask.append(grad_x_mask)
            self.grad_y_mask.append(grad_y_mask)
            self.loss_grad.append(loss_grad)
        
    def image_loss(self, y_true, y_pred, mask):
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred)*mask, axis=[0,1,2])
        l2 = tf.reduce_mean(tf.square(y_true - y_pred)*mask, axis=[0,1,2])
        
        return tf.reduce_mean((l1 + 1.5*l2)*self.depth_channel_mask)

    def total_loss(self):
        total_loss = self.loss_image
        for l in self.loss_grad:
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ImageLoss:
    def __init__ (self, image, image_pred):
        self.grad_x = []
        self.grad_y = []
        self.grad_x_pred = []
        self.grad_y_pred = []
        self.loss_grad = []

        grad_x, grad_y = ImageLoss.image_gradient(image)
        grad_x_pred, grad_y_pred = ImageLoss.image_gradient(image_pred)
        loss_grad =\
            ImageLoss.image_loss(grad_x, grad_x_pred) +\
            ImageLoss.image_loss(grad_y, grad_y_pred)
        
        self.grad_x.append(grad_x)
        self.grad_y.append(grad_y)
        self.grad_x_pred.append(grad_x_pred)
        self.grad_y_pred.append(grad_y_pred)
        self.loss_grad.append(loss_grad)

        self.loss_image = ImageLoss.image_loss(image, image_pred)

        self.images = [image]
        self.image_preds = [image_pred]
        for i in range(4):
            self.images.append(layers.AveragePooling2D((2,2))(self.images[-1]))
            self.image_preds.append(layers.AveragePooling2D((2,2))(self.image_preds[-1]))
            grad_x, grad_y = ImageLoss.image_gradient(self.images[-1])
            grad_x_pred, grad_y_pred = ImageLoss.image_gradient(self.image_preds[-1])
            loss_grad =\
                ImageLoss.image_loss(grad_x, grad_x_pred) +\
                ImageLoss.image_loss(grad_y, grad_y_pred)
            
            self.grad_x.append(grad_x)
            self.grad_y.append(grad_y)
            self.grad_x_pred.append(grad_x_pred)
            self.grad_y_pred.append(grad_y_pred)
            self.loss_grad.append(loss_grad)
        
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
    def image_loss(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred)) +\
            1.5*tf.reduce_mean(tf.square(y_true - y_pred))
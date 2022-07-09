import numpy as np
import skimage
from tensorflow import image
from image_loss import ImageLoss
from tensorflow.keras import layers


def relative_error(target, prediction):
    err = np.abs(target.numpy()-prediction.numpy())
    return np.divide(err, target.numpy(), out=np.zeros_like(err), where=target.numpy() != 0)

# Encode + Decode
def edec(encoder, decoder, image_target, automap):
    image_enc = encoder([image_target, automap], training=False)
    image_pred, depth_pred = decoder(image_enc, training=False)
    return image_pred, depth_pred

def double_edec_loss(encoder, decoder, image_target, image_target_combined, automap):
    image_pred, _ = edec(encoder, decoder, image_target, automap)
    image_pred_2, depth_pred_2 = edec(encoder, decoder, image_pred, automap) # Use the same automap twice

    # Calculate error between image_pred_2 and image_target
    image_pred_combined = layers.Concatenate()([image_pred_2, depth_pred_2])
    image_loss = ImageLoss(image_target_combined, image_pred_combined)
    loss_total = image_loss.total_loss()
    return loss_total

# Assume that inputs are of the same shape, have channel order
# NHWC and C==3
def print_total_metrics(yuv_target, yuv_pred):
    print("==== Metrics ====")
    mae_yuv = np.concatenate(np.abs(yuv_target - yuv_pred)).mean()
    mae_yuv_chans = [np.concatenate(np.abs(yuv_target[:,:,k] - yuv_pred[:,:,0])).mean() for k in range(3)]
    mse_yuv = np.concatenate(np.square(yuv_target - yuv_pred)).mean()
    mse_yuv_chans = [np.concatenate(np.square(yuv_target[:,:,k] - yuv_pred[:,:,0])).mean() for k in range(3)]

    psnr = skimage.metrics.peak_signal_noise_ratio(yuv_target.numpy(), yuv_pred.numpy())

    ssim = image.ssim(yuv_target, yuv_pred, max_val=1.0, filter_size=11,
            filter_sigma=1.5, k1=0.01, k2=0.03)
    ssim_ms = image.ssim_multiscale(yuv_target, yuv_pred, max_val=1.0)

    nmi = skimage.metrics.normalized_mutual_information(yuv_target, yuv_pred) # output = 1...2 where 1=perfectly incorrelated,2=perfectly correlated

    print(f"MAE {mae_yuv:.3f} ({mae_yuv_chans[0]:.3f} {mae_yuv_chans[1]:.3f} {mae_yuv_chans[2]:.3f}) "
        f"MSE {mse_yuv:.3f} ({mse_yuv_chans[0]:.3f} {mse_yuv_chans[1]:.3f} {mse_yuv_chans[2]:.3f}) "
        f"PSNR: {psnr:.1f} SSIM: {ssim:.3f} MS-SSIM {ssim_ms:.3f} NMI {nmi:.1f}")

    
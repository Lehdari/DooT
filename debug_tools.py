import cv2
import numpy as np

def print_frame_diff(images, actions, id):
    print(actions[id,0,14].numpy())
    cv2.imshow("img_{}".format(id), cv2.cvtColor(images[id,0,:,:,0:3].numpy(), cv2.COLOR_RGB2BGR))
    cv2.imshow("img_{}".format(id+1), cv2.cvtColor(images[id+1,0,:,:,0:3].numpy(), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

def show_frame_comparison(target, prediction):
    f = np.zeros((240,320,3), dtype=np.float32)
    m = np.zeros((240,320,3), dtype=np.float32)
    g = np.zeros((240,320,3), dtype=np.float32)

    cv2.imshow("target / prediction", cv2.vconcat([cv2.hconcat([
        cv2.cvtColor(target.numpy(), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(prediction.numpy(), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(np.abs(prediction.numpy()-target.numpy()), cv2.COLOR_RGB2BGR)]),
        cv2.hconcat([f / 64 + 0.5, m, g])]))
    cv2.waitKey(1)
import cv2
import numpy as np
from metrics import relative_error

def print_frame_diff(images, actions, id):
    print(actions[id,0,14].numpy())
    cv2.imshow("img_{}".format(id), cv2.cvtColor(images[id,0,:,:,0:3].numpy(), cv2.COLOR_RGB2BGR))
    cv2.imshow("img_{}".format(id+1), cv2.cvtColor(images[id+1,0,:,:,0:3].numpy(), cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

def show_frame_comparison(target, prediction, windowname, delay=-1):
    cv2.imshow(windowname, cv2.hconcat([
        cv2.cvtColor(target.numpy(), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(prediction.numpy(), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(np.abs(prediction.numpy()-target.numpy()), cv2.COLOR_RGB2BGR)]))

    if delay>=0:
        cv2.waitKey(delay)

# Frame comparison with relative error instead of absolute error
def show_frame_comparison_relative(target, prediction, windowname, delay=-1):
    cv2.imshow(windowname, cv2.hconcat([
        cv2.cvtColor(target.numpy(), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(prediction.numpy(), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(relative_error(target, prediction), cv2.COLOR_RGB2BGR)]))

    if delay>=0:
        cv2.waitKey(delay)

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def show_frame_comparison_all(target, prediction, windowname, delay=-1):
    cv2.imshow(windowname, concat_tile([
       [cv2.cvtColor(target.numpy(), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(prediction.numpy(), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(relative_error(target, prediction), cv2.COLOR_RGB2BGR)],
       [cv2.cvtColor(target.numpy(), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(prediction.numpy(), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(np.abs(prediction.numpy()-target.numpy()), cv2.COLOR_RGB2BGR)], 
    ]))

    if delay>=0:
        cv2.waitKey(delay)
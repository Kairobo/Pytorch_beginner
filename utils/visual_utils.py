import matplotlib
import matplotlib.pyplot as plt
from torchvision import utils


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    images_batch, labels_batch = \
        sample_batched['image'], sample_batched['label']
    batch_size = len(images_batch)
    im_size = images_batch.shape
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()
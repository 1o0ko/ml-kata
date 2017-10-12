import matplotlib.pyplot as plt

from utils.data import Counter

def plot_sample(gen):
    sample, _ = gen.sample()
    plt.imshow(sample, cmap='gray')
    plt.title("Random Sample from Random Class")
    plt.axis('off')
    plt.show()


def plot_class_samples(gen):
    plot_number = Counter()
    for y in range(gen.num_classes):
        sample, mean = gen.sample(y)

        plt.subplot(gen.num_classes, 2, plot_number.value())
        plt.imshow(sample, cmap='gray')
        plt.axis('off')

        plt.subplot(gen.num_classes, 2, plot_number.value())
        plt.imshow(mean, cmap='gray')
        plt.axis('off')

    plt.show()

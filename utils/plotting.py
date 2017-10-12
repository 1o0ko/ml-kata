import matplotlib.pyplot as plt

from utils.data import Counter

def plot_sample(gen):
    sample, y = gen.sample()
    plt.imshow(sample, cmap='gray')
    plt.title("Random Sample from Random Class [%s]" % y)
    plt.axis('off')
    plt.show()


def plot_class_samples(gen):
    plot_number = Counter()
    for y in range(gen.num_classes):
        sample, _ = gen.sample(y)
        mean = gen.params[y]['mean'].reshape(28, 28)

        plt.subplot(gen.num_classes, 2, plot_number.value())
        plt.imshow(sample, cmap='gray')
        plt.axis('off')

        plt.subplot(gen.num_classes, 2, plot_number.value())
        plt.imshow(mean, cmap='gray')
        plt.axis('off')

    plt.show()

import matplotlib.pyplot as plt

def visulizeClassByName(img, label_name, prob=1, hold=False):
    """
    Visualize image and its class name.
    :param img: image.
    :param label_name: single label name
    :return: None.
    """
    # label = np.argmax(label) + 1
    class_name = label_name

    # Show image and its label.
    if not hold:
        fig, ax = plt.subplots(1)
    else:
        ax = plt.gca()

    ax.clear()
    ax.imshow(img)

    # Add the patch to the Axes
    text = 'Class: %s, prob: %f.' %(class_name, prob)
    ax.text(10, 10, text, color='r', bbox=dict(facecolor='green', alpha=0.5))

    plt.draw()
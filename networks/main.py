from .evlog_net import EVLOG_Net


def build_network(meta_data):
    """Builds the neural network."""

    net = EVLOG_Net(meta_data)

    return net



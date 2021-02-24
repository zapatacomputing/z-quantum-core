from _gates import CustomGate
from _matrices import x_matrix, rx_matrix

X = CustomGate("X", x_matrix, (), 1)


def RX(angle):
    return CustomGate("RX", rx_matrix, (angle,), 1)

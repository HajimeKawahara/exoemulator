"""opacity model training (opt).

    - spectrum training (spt)

"""

from exoemulator.utils.checkpackage import check_installed


class OptExoJAX:

    def __init__(self):
        check_installed("exojax")


if __name__ == "__main__":
    opt = OptExoJAX()

"""opacity model training (opt).

    - spectrum training (spt)

"""

from exoemulator.utils.checkpackage import check_installed


class OptExoJAX:
    """Opacity Training with ExoJAX"""

    def __init__(self, opa=None):
        check_installed("exojax")
        self.set_opa(opa)

    def set_opa(self, opa):
        if opa is None:
            print("opa has not been given yet")
        else:
            print("opa in opt: ", opa.__class__.__name__)
            self.opa = opa

    def generate_mini_batches():
        pass


if __name__ == "__main__":
    opt = OptExoJAX()

from flax import nnx


class EmuMlpDecoder(nnx.Module):
    """simple decoder type MLP neuralnet emulator model

    Args:
        nnx (_type_): nnx module

    """

    def __init__(self, *, rngs: nnx.Rngs, grid_length: int):
        self.dense_entrance = nnx.Linear(in_features=2, out_features=16, rngs=rngs)
        self.dense_1 = nnx.Linear(in_features=16, out_features=32, rngs=rngs)
        self.dense_2 = nnx.Linear(in_features=32, out_features=256, rngs=rngs)
        self.dense_3 = nnx.Linear(in_features=256, out_features=512, rngs=rngs)
        self.dense_out = nnx.Linear(
            in_features=512, out_features=grid_length, rngs=rngs
        )

    def __call__(self, input_parameter):
        x = nnx.gelu(self.dense_entrance(input_parameter))
        x = nnx.gelu(self.dense_1(x))
        x = nnx.gelu(self.dense_2(x))
        x = nnx.gelu(self.dense_3(x))
        return self.dense_out(x)


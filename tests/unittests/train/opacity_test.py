from exoemulator.utils.checkpackage import check_installed
from exoemulator.train.opacity import OptExoJAX


def test_OptExoJAX():
    opt = OptExoJAX()


def test_OptExoJAX_opa_premodit():
    if check_installed("exojax"):
        from exojax.test.emulate_mdb import mock_mdbExomol
        from exojax.test.emulate_mdb import mock_wavenumber_grid
        from exojax.spec.opacalc import OpaPremodit
        from jax import config

        config.update("jax_enable_x64", True)
        nu_grid, wav, res = mock_wavenumber_grid()
        mdb = mock_mdbExomol()
        opa = OpaPremodit(mdb, nu_grid, auto_trange=[490.0, 510.0])
        opt = OptExoJAX(opa=opa)

        assert opt.opa.method == "premodit"
    else:
        pass




if __name__ == "__main__":
    test_OptExoJAX()
    test_OptExoJAX_opa_premodit()

from zquantum.core.interfaces.optimizer import optimization_result


def test_optimization_result_contains_opt_value_and_opt_params():
    opt_value = 2.0
    opt_params = [-1, 0, 3.2]

    result = optimization_result(opt_value=opt_value, opt_params=opt_params)

    assert result.opt_value == opt_value
    assert result.opt_params == opt_params


def test_optimization_result_contains_other_attributes_passed_as_kwargs():
    opt_value = 0.0
    opt_params = [1, 2, 3]
    kwargs = {"bitstring": "01010", "foo": 3.0}

    result = optimization_result(opt_value=opt_value, opt_params=opt_params, **kwargs)

    assert all(getattr(result, key) == value for key, value in kwargs.items())

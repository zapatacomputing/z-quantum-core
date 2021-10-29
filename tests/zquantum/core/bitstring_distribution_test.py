import zquantum.core.bitstring_distribution as old_distribution_module
import zquantum.core.distribution as new_distribution_module


def test_both_modules_provide_same_api():
    old = set(dir(old_distribution_module))
    new = set(dir(new_distribution_module))

    assert "BitstringDistribution" in old
    old.remove("BitstringDistribution")
    assert new.issuperset(old)
    assert new.difference(old) == {
        "mmd",
        "__path__",
        "clipped_negative_log_likelihood",
        "math",
        "_measurement_outcome_distribution",
        "jensen_shannon_divergence",
    }

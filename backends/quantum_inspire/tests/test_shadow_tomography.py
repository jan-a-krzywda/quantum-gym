from ..shadow_tomography import expectation_from_counts, estimate_pauli_expectations_from_counts


def test_expectation_simple():
    counts = {'00': 500, '11': 500}
    exp, std = expectation_from_counts(counts, 'ZZ')
    # For counts with half 00 and half 11, expectation = -0.0? actually ZZ eigenvalues: 00 -> +1, 11 -> +1
    assert abs(exp - 1.0) < 1e-6


def test_estimate_multiple():
    counts = {'0': 300, '1': 700}
    res = estimate_pauli_expectations_from_counts(counts, ['Z'])
    assert 'Z' in res
    exp, std = res['Z']
    assert -1.0 <= exp <= 1.0

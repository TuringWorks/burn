use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_atanh_ops() {
    // atanh is defined for inputs in (-1, 1)
    let data = TensorData::from([[-0.5, 0.0, 0.5], [-0.9, 0.25, 0.9]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.atanh();
    // Expected values: atanh(-0.5)=-0.5493, atanh(0)=0, atanh(0.5)=0.5493
    //                  atanh(-0.9)=-1.4722, atanh(0.25)=0.2554, atanh(0.9)=1.4722
    let expected = TensorData::from([
        [-0.5493061, 0.0, 0.5493061],
        [-1.4722195, 0.2554128, 1.4722195],
    ]);

    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

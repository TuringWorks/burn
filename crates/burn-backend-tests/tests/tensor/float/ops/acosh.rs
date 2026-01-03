use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_acosh_ops() {
    // acosh is defined for inputs >= 1
    let data = TensorData::from([[1.0, 1.5, 2.0], [2.5, 3.0, 4.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.acosh();
    // Expected values: acosh(1)=0, acosh(1.5)=0.9624, acosh(2)=1.3170
    //                  acosh(2.5)=1.5668, acosh(3)=1.7627, acosh(4)=2.0634
    let expected = TensorData::from([
        [0.0, 0.9624237, 1.3169578],
        [1.5667992, 1.7627472, 2.0634370],
    ]);

    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

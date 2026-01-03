use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_asinh_ops() {
    let data = TensorData::from([[-1.0, 0.0, 1.0], [-2.0, 0.5, 2.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.asinh();
    // Expected values: asinh(-1)=-0.8814, asinh(0)=0, asinh(1)=0.8814
    //                  asinh(-2)=-1.4436, asinh(0.5)=0.4812, asinh(2)=1.4436
    let expected = TensorData::from([
        [-0.8813736, 0.0, 0.8813736],
        [-1.4436355, 0.48121184, 1.4436355],
    ]);

    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

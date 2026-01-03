use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_asin_ops() {
    // asin is defined for inputs in [-1, 1]
    let data = TensorData::from([[-0.5, 0.0, 0.5], [-1.0, 0.25, 1.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.asin();
    // Expected values: asin(-0.5)=-0.5236, asin(0)=0, asin(0.5)=0.5236
    //                  asin(-1)=-1.5708, asin(0.25)=0.2527, asin(1)=1.5708
    let expected = TensorData::from([
        [-0.5235988, 0.0, 0.5235988],
        [-1.5707964, 0.25268024, 1.5707964],
    ]);

    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

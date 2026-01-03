use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_acos_ops() {
    // acos is defined for inputs in [-1, 1]
    let data = TensorData::from([[-0.5, 0.0, 0.5], [-1.0, 0.25, 1.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.acos();
    // Expected values: acos(-0.5)=2.0944, acos(0)=1.5708, acos(0.5)=1.0472
    //                  acos(-1)=3.1416, acos(0.25)=1.3181, acos(1)=0
    let expected = TensorData::from([
        [2.0943952, 1.5707964, 1.0471976],
        [3.1415927, 1.3181161, 0.0],
    ]);

    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

use super::*;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;

#[test]
fn should_support_atan_ops() {
    let data = TensorData::from([[-1.0, 0.0, 1.0], [-2.0, 0.5, 2.0]]);
    let tensor = TestTensor::<2>::from_data(data, &Default::default());

    let output = tensor.atan();
    // Expected values: atan(-1)=-0.7854, atan(0)=0, atan(1)=0.7854
    //                  atan(-2)=-1.1071, atan(0.5)=0.4636, atan(2)=1.1071
    let expected = TensorData::from([
        [-0.7853982, 0.0, 0.7853982],
        [-1.1071488, 0.4636476, 1.1071488],
    ]);

    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

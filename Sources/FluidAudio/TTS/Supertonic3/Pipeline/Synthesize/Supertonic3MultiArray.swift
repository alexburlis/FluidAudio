@preconcurrency import CoreML
import Foundation

/// Tiny helpers for moving Float / Int32 buffers in and out of `MLMultiArray`
/// for the Supertonic-3 stages.
///
/// All factories produce row-major contiguous Float32 / Int32 arrays; strides
/// are inferred from the shape. `extractFloats` handles either Float32 or
/// Float16 backing storage so a CoreML graph using FP16 IO is transparent.
public enum Supertonic3MultiArray {

    public static func makeFloat32(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let expected = shape.reduce(1, *)
        guard values.count == expected else {
            throw Supertonic3Error.invalidTensorShape(
                stage: "makeFloat32",
                expected: "\(expected) (shape \(shape))",
                got: "\(values.count)")
        }
        let arr = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        let dst = arr.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        values.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: values.count)
        }
        return arr
    }

    public static func makeInt32(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let expected = shape.reduce(1, *)
        guard values.count == expected else {
            throw Supertonic3Error.invalidTensorShape(
                stage: "makeInt32",
                expected: "\(expected) (shape \(shape))",
                got: "\(values.count)")
        }
        let arr = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .int32)
        let dst = arr.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        values.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: values.count)
        }
        return arr
    }

    /// Read the logical elements of an `MLMultiArray` in row-major order,
    /// handling Float32 / Float16 / Double / Int32 backing stores.
    public static func extractFloats(_ arr: MLMultiArray) -> [Float] {
        arr.rowMajorFloats()
    }
}

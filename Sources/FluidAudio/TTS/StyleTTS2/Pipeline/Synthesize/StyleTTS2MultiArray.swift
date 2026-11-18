@preconcurrency import CoreML
import Foundation

/// Tiny helpers for moving Float / Int32 buffers in and out of `MLMultiArray`.
///
/// All factories produce row-major contiguous arrays — strides are
/// computed from the shape so the StyleTTS2 stages see the layout they were
/// traced with. `extractFloats` reads either Float32 or Float16 backing
/// storage and returns a Swift Float32 buffer.
public enum StyleTTS2MultiArray {

    // MARK: - Factories

    public static func makeFloat32(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        precondition(values.count == shape.reduce(1, *), "shape product must match buffer length")
        let arr = try MLMultiArray(
            shape: shape.map(NSNumber.init), dataType: .float32)
        let dst = arr.dataPointer.bindMemory(
            to: Float.self, capacity: values.count)
        values.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: values.count)
        }
        return arr
    }

    public static func makeInt32(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        precondition(values.count == shape.reduce(1, *), "shape product must match buffer length")
        let arr = try MLMultiArray(
            shape: shape.map(NSNumber.init), dataType: .int32)
        let dst = arr.dataPointer.bindMemory(
            to: Int32.self, capacity: values.count)
        values.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: values.count)
        }
        return arr
    }

    // MARK: - Extraction

    /// Read the logical elements of an `MLMultiArray` in row-major order,
    /// handling both `.float32` and `.float16` backing stores.
    public static func extractFloats(_ arr: MLMultiArray) -> [Float] {
        arr.rowMajorFloats()
    }

    /// Convenience: produce a flat `[Float]` of length `S0*S1*S2…` plus the
    /// shape as `[Int]` for ergonomic debugging.
    public static func shape(of arr: MLMultiArray) -> [Int] {
        return arr.shape.map { $0.intValue }
    }
}

import XCTest
@testable import FluidAudio

final class SpeechComponentEvidenceTests: XCTestCase {
    func testEmptyNearestPoolCannotCreateBiasOnlySpeaker() {
        XCTAssertNil(OfflineSpeechComponentExtractor.cleanPoolingWeights([0, 1, 1, 0], featureFrames: 8))
        XCTAssertEqual(OfflineSpeechComponentExtractor.cleanPoolingWeights([1, 0, 0, 1], featureFrames: 16), [1, 0])
        XCTAssertEqual(OfflineSpeechComponentExtractor.cleanPoolingWeights([0, 1, 1, 0], featureFrames: 16), [0, 1])
    }

    func testUnsupportedVBxMergeKeepsDistinctSpeakers() throws {
        let result = try refine(intervals: [[0..<100], [200..<300]])
        XCTAssertEqual(result, [[1, 0], [0, 1]])
    }

    func testSharedCleanAudioSupportsMergeAtTrainingFloor() throws {
        let result = try refine(intervals: [[0..<100], [50..<150]])
        XCTAssertEqual(result, [[0.5, 0.5]])
        XCTAssertEqual(try refine(intervals: [[0..<100], [51..<151]]), [[1, 0], [0, 1]])
    }

    func testAcousticSupportPreservesOriginalVBxCentroid() throws {
        let result = try OfflineSpeakerMergeSupport.refine(
            centroids: [[0.95, 0.05]], retainedColumns: [0], gamma: [[1], [1]],
            initialClusters: [0, 1], trainingEmbeddings: [[1, 0], [0.9, 0.1]],
            cleanIntervals: [[0..<100], [200..<300]], minimumSharedSamples: 50
        )
        XCTAssertEqual(result, [[0.95, 0.05]])
    }

    func testMissingPhysicalEvidenceCannotPretendToValidateMerge() {
        XCTAssertThrowsError(try refine(intervals: [[0..<100], []]))
    }

    func testOrderedParallelMapReturnsResultsInInputOrder() async throws {
        let values = try await OfflineSpeechComponentExtractor.orderedParallelMap(
            count: 200, maxConcurrency: 8
        ) { index in
            // Uneven work so completion order differs from index order.
            if index % 7 == 0 { Thread.sleep(forTimeInterval: 0.002) }
            return index * 3
        }
        XCTAssertEqual(values, (0..<200).map { $0 * 3 })
    }

    func testOrderedParallelMapRethrowsAFailure() async {
        struct Failure: Error {}
        do {
            _ = try await OfflineSpeechComponentExtractor.orderedParallelMap(
                count: 50, maxConcurrency: 4
            ) { index -> Int in
                if index == 31 { throw Failure() }
                return index
            }
            XCTFail("a failing element must fail the whole map")
        } catch {
            XCTAssertTrue(error is Failure)
        }
    }

    func testOrderedParallelMapWithNoWorkOrOneLane() async throws {
        let none = try await OfflineSpeechComponentExtractor.orderedParallelMap(
            count: 0, maxConcurrency: 4
        ) { $0 }
        XCTAssertEqual(none, [])
        let serial = try await OfflineSpeechComponentExtractor.orderedParallelMap(
            count: 5, maxConcurrency: 1
        ) { $0 + 1 }
        XCTAssertEqual(serial, [1, 2, 3, 4, 5])
    }

    private func refine(intervals: [[Range<Int>]]) throws -> [[Double]] {
        try OfflineSpeakerMergeSupport.refine(
            centroids: [[0.5, 0.5]], retainedColumns: [0], gamma: [[1], [1]],
            initialClusters: [0, 1], trainingEmbeddings: [[1, 0], [0, 1]],
            cleanIntervals: intervals, minimumSharedSamples: 50
        )
    }
}

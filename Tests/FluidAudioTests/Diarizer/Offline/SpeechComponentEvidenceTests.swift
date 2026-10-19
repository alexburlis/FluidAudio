import XCTest
import os
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

    func testOrderedStreamingMapReturnsResultsInEmissionOrder() async throws {
        let finished = OSAllocatedUnfairLock<[Int]>(initialState: [])
        let (_, values) = try await OfflineSpeechComponentExtractor.orderedStreamingMap(
            maxConcurrency: 4,
            produce: { emit in (0..<40).forEach(emit) },
            transform: { (item: Int) -> Int in
                // The first item finishes last, so completion order differs from emission order.
                if item == 0 { Thread.sleep(forTimeInterval: 0.05) }
                finished.withLock { $0.append(item) }
                return item * 3
            }
        )
        XCTAssertEqual(values, (0..<40).map { $0 * 3 })
        XCTAssertNotEqual(finished.withLock { $0.first }, 0, "transforms must finish out of order for this test")
    }

    func testOrderedStreamingMapKeepsOptionalResultsInPlace() async throws {
        // Component embeddings are `[Float]?`: a missing identity must stay at its position.
        let (_, values) = try await OfflineSpeechComponentExtractor.orderedStreamingMap(
            maxConcurrency: 3,
            produce: { emit in (0..<6).forEach(emit) },
            transform: { (item: Int) -> [Float]? in item.isMultiple(of: 2) ? [Float(item)] : nil }
        )
        XCTAssertEqual(values, [[0], nil, [2], nil, [4], nil])
    }

    func testOrderedStreamingMapStartsTransformsWhileProducing() async throws {
        let started = OSAllocatedUnfairLock(initialState: false)
        let (overlapped, values) = try await OfflineSpeechComponentExtractor.orderedStreamingMap(
            maxConcurrency: 2,
            produce: { emit -> Bool in
                emit(0)
                // Keep producing until the first item is being transformed, or give up.
                let deadline = ContinuousClock.now + .seconds(5)
                while !started.withLock({ $0 }), ContinuousClock.now < deadline {
                    try await Task.sleep(for: .milliseconds(1))
                }
                emit(1)
                return started.withLock { $0 }
            },
            transform: { (item: Int) -> Int in
                started.withLock { $0 = true }
                return item
            }
        )
        XCTAssertTrue(overlapped, "the first item must start transforming before the producer finishes")
        XCTAssertEqual(values, [0, 1])
    }

    func testOrderedStreamingMapRespectsTheConcurrencyCap() async throws {
        struct Lanes: Sendable {
            var active = 0
            var peak = 0
        }
        for cap in [1, 3] {
            let lanes = OSAllocatedUnfairLock(initialState: Lanes())
            let (_, values) = try await OfflineSpeechComponentExtractor.orderedStreamingMap(
                maxConcurrency: cap,
                produce: { emit in (0..<48).forEach(emit) },
                transform: { (item: Int) -> Int in
                    lanes.withLock {
                        $0.active += 1
                        $0.peak = max($0.peak, $0.active)
                    }
                    Thread.sleep(forTimeInterval: 0.001)
                    lanes.withLock { $0.active -= 1 }
                    return item
                }
            )
            XCTAssertEqual(values, Array(0..<48))
            XCTAssertLessThanOrEqual(lanes.withLock { $0.peak }, cap)
        }
    }

    func testOrderedStreamingMapRethrowsAProducerFailure() async {
        struct Failure: Error {}
        do {
            _ = try await OfflineSpeechComponentExtractor.orderedStreamingMap(
                maxConcurrency: 2,
                produce: { emit in
                    (0..<5).forEach(emit)
                    throw Failure()
                },
                transform: { (item: Int) -> Int in item }
            )
            XCTFail("a failing producer must fail the whole map")
        } catch {
            XCTAssertTrue(error is Failure)
        }
    }

    func testOrderedStreamingMapTransformFailureStopsTheProducer() async {
        struct Failure: Error {}
        struct Production: Sendable {
            var emitted = 0
            var finished = false
        }
        let production = OSAllocatedUnfairLock(initialState: Production())
        do {
            _ = try await OfflineSpeechComponentExtractor.orderedStreamingMap(
                maxConcurrency: 2,
                produce: { emit in
                    defer { production.withLock { $0.finished = true } }
                    for item in 0..<2_000 {
                        emit(item)
                        production.withLock { $0.emitted += 1 }
                        try await Task.sleep(for: .milliseconds(1))
                    }
                },
                transform: { (item: Int) -> Int in
                    if item == 3 { throw Failure() }
                    return item
                }
            )
            XCTFail("a failing transform must fail the whole map")
        } catch {
            XCTAssertTrue(error is Failure)
        }
        // The map returns only after its producer has stopped, and the failure stopped it early.
        let stopped = production.withLock { $0 }
        XCTAssertTrue(stopped.finished)
        XCTAssertLessThan(stopped.emitted, 2_000)
    }

    func testOrderedStreamingMapWithNothingProduced() async throws {
        let (produced, values) = try await OfflineSpeechComponentExtractor.orderedStreamingMap(
            maxConcurrency: 4,
            produce: { _ in "done" },
            transform: { (item: Int) -> Int in item }
        )
        XCTAssertEqual(produced, "done")
        XCTAssertEqual(values, [])
    }

    func testOrderedStreamingMapCancelledCallerGetsNoPartialResults() async {
        let emitted = OSAllocatedUnfairLock(initialState: 0)
        let map = Task {
            try await OfflineSpeechComponentExtractor.orderedStreamingMap(
                maxConcurrency: 2,
                produce: { emit in
                    // Ignores cancellation, so only the map itself can refuse to return partial results.
                    for item in 0..<20 {
                        emit(item)
                        emitted.withLock { $0 += 1 }
                        try? await Task.sleep(for: .milliseconds(5))
                    }
                },
                transform: { (item: Int) -> Int in item }
            )
        }
        let deadline = ContinuousClock.now + .seconds(5)
        while emitted.withLock({ $0 }) < 3, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(1))
        }
        map.cancel()
        do {
            let partial = try await map.value
            XCTFail("a cancelled map returned \(partial.results)")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
    }

    private func refine(intervals: [[Range<Int>]]) throws -> [[Double]] {
        try OfflineSpeakerMergeSupport.refine(
            centroids: [[0.5, 0.5]], retainedColumns: [0], gamma: [[1], [1]],
            initialClusters: [0, 1], trainingEmbeddings: [[1, 0], [0, 1]],
            cleanIntervals: intervals, minimumSharedSamples: 50
        )
    }
}

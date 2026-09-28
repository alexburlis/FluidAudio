import AVFoundation
import XCTest

@testable import FluidAudio

/// Real SDK boundary proof for the app's awaited, value-owned audio feed.
/// Failure modes frozen before implementation: unbounded queued PCM; an append
/// acknowledging before decoding; missing final words; finish hanging an update
/// consumer; repeated finish re-decoding; cancellation publishing late text;
/// concurrent/invalid input accepted silently. Existing final-window tests cover
/// transcript quality, but use the unacknowledged PCM queue and never cancel it.
final class SlidingWindowOrderedInputTests: XCTestCase {
    private func inputs() throws -> (AsrModels, [Float]) {
        let directory =
            ProcessInfo.processInfo.environment["FLUIDAUDIO_TEST_ASR_MODELS"]
            .map { URL(fileURLWithPath: $0) } ?? AsrModels.defaultCacheDirectory(for: .v3)
        try XCTSkipUnless(AsrModels.modelsExist(at: directory, version: .v3), "Real local v3 models required")
        let models = try AsrModels.loadLocal(from: directory, version: .v3)
        let fileName = "01-validation-request-21.4s.wav"
        guard
            let url = Bundle.module.url(forResource: "Fixtures/" + fileName, withExtension: nil)
                ?? Bundle.module.url(forResource: fileName, withExtension: nil)
        else {
            throw XCTSkip("Public issue-855 recording required")
        }
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        XCTAssertEqual(format.sampleRate, 16_000)
        XCTAssertEqual(format.channelCount, 1)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let channel = try XCTUnwrap(buffer.floatChannelData?[0])
        return (models, Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength))))
    }

    private func manager(_ models: AsrModels) async throws -> SlidingWindowAsrManager {
        let manager = SlidingWindowAsrManager(
            config: SlidingWindowAsrConfig(
                chunkSeconds: 3, leftContextSeconds: 11, rightContextSeconds: 1, language: .english))
        try await manager.loadModels(models)
        try await manager.startOrderedStreaming()
        return manager
    }

    func testAwaitedRealAudioHasEarlyPreviewCompleteTailAndOneTerminalResult() async throws {
        let (models, samples) = try inputs()
        let manager = try await manager(models)
        let updates = await manager.transcriptionUpdates
        let collected = Task { () -> [SlidingWindowTranscriptionUpdate] in
            var values: [SlidingWindowTranscriptionUpdate] = []
            for await value in updates { values.append(value) }
            return values
        }
        var offset = 0
        while offset < samples.count {
            let end = min(offset + 3_200, samples.count)
            try await manager.appendAudioSamples(Array(samples[offset..<end]))
            offset = end
            let progress = await manager.orderedAudioProgress
            XCTAssertEqual(progress.acceptedSamples, offset)
            XCTAssertEqual(progress.processedSamples, offset)
            XCTAssertEqual(progress.inFlightSamples, 0)
            XCTAssertLessThanOrEqual(progress.highWaterSamples, 3_200)
            if offset == 64_000 {
                let preview = await manager.volatileTranscript
                XCTAssertFalse(preview.isEmpty, "Four seconds of real audio must produce a pre-Stop hypothesis")
            }
        }
        let final = try await manager.finishOrderedStreaming()
        XCTAssertTrue(final.lowercased().contains("help them out"), "Real final words must survive: \(final)")
        let repeated = try await manager.finishOrderedStreaming()
        XCTAssertEqual(repeated, final)
        let values = await collected.value
        XCTAssertFalse(values.isEmpty)
        XCTAssertTrue(values.allSatisfy { $0.confirmedTranscript != nil && $0.volatileTranscript != nil })
        do {
            try await manager.appendAudioSamples(Array(samples.prefix(3_200)))
            XCTFail("A finished session cannot accept audio")
        } catch let error as SlidingWindowAsrError {
            guard case .invalidStreamState = error else { return XCTFail("Wrong terminal error: \(error)") }
        }
    }

    func testCancelWaitsForAcceptedRealAudioAndRejectsFinalSuccess() async throws {
        let (models, samples) = try inputs()
        let manager = try await manager(models)
        let append = Task { try await manager.appendAudioSamples(Array(samples.prefix(160_000))) }
        for _ in 0..<1_000 {
            if await manager.orderedAudioProgress.inFlightSamples > 0 { break }
            await Task.yield()
        }
        let started = await manager.orderedAudioProgress
        XCTAssertGreaterThan(started.inFlightSamples, 0, "The cancellation must reach accepted in-flight audio")
        do {
            try await manager.appendAudioSamples(Array(samples.prefix(3_200)))
            XCTFail("Concurrent input must not create an unbounded decoder queue")
        } catch let error as SlidingWindowAsrError {
            guard case .bufferOverflow = error else { return XCTFail("Wrong pressure error: \(error)") }
        }
        await manager.cancel()
        _ = await append.result
        let stopped = await manager.orderedAudioProgress
        XCTAssertEqual(stopped.inFlightSamples, 0)
        let preview = await manager.volatileTranscript
        XCTAssertEqual(preview, "")
        do {
            _ = try await manager.finishOrderedStreaming()
            XCTFail("Cancelled audio must never be returned as a complete transcript")
        } catch is CancellationError {}
    }

    func testRejectedInputDoesNotEnterTheDecoderOrDisplaceRealSamples() async throws {
        let (models, samples) = try inputs()
        let manager = try await manager(models)
        var invalid = Array(samples.prefix(3_200))
        invalid[0] = .nan
        do {
            try await manager.appendAudioSamples(invalid)
            XCTFail("Non-finite audio must fail before acceptance")
        } catch let error as SlidingWindowAsrError {
            guard case .invalidConfiguration = error else { return XCTFail("Wrong input error: \(error)") }
        }
        let rejected = await manager.orderedAudioProgress
        XCTAssertEqual(rejected.acceptedSamples, 0)
        try await manager.appendAudioSamples(Array(samples.prefix(160_000)))
        let final = try await manager.finishOrderedStreaming()
        XCTAssertFalse(final.isEmpty)
    }
}

import Accelerate
@preconcurrency import CoreML
import Foundation
import os

/// The Community-1 speech-component path. Segmentation activity is retained even
/// when a component has no usable acoustic identity (overlap or a padded tail).
@available(macOS 14.0, iOS 17.0, *)
struct OfflineSpeechComponentExtractor {
    private struct Component: Equatable {
        let chunk: Int
        let slot: Int
        let activeFrames: [Int]
        let cleanMask: [Float]
        let sampleRange: Range<Int>
        let cleanIntervals: [Range<Int>]
        let trainingEligible: Bool
    }

    static func prepare(
        audioSource: AudioSampleSource,
        audioLoadingSeconds: TimeInterval,
        models: OfflineDiarizerModels,
        config: OfflineDiarizerConfig,
        progressCallback: (@Sendable (Int, Int) -> Void)?
    ) async throws -> PreparedDiarization {
        guard let embeddingModel = models.speechComponentEmbeddingModel else {
            throw OfflineDiarizationError.modelNotLoaded("speech-component-embedding")
        }
        guard config.samplesPerWindow == 160_000, config.samplesPerStep == 16_000,
            config.sampleRate == 16_000, config.embedding.excludeOverlap,
            !config.zeroVoteReembed.enabled
        else {
            throw OfflineDiarizationError.invalidConfiguration(
                "Speech components require the Community-1 waveform geometry and clean masks")
        }
        guard models.pldaPsi.count == 128, models.pldaPsi.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw OfflineDiarizationError.invalidConfiguration("Invalid component PLDA scale")
        }

        let start = Date()
        let full = max(0, 1 + (audioSource.sampleCount - 160_000) / 16_000)
        let total =
            audioSource.sampleCount < 160_000
            ? 1 : full + (audioSource.sampleCount % 16_000 == 0 ? 0 : 1)
        // Segmentation runs mostly on the Neural Engine and each component embedding is an
        // independent CPU prediction, so a chunk's components start embedding as soon as the
        // chunk is segmented, with up to `componentEmbeddingConcurrency` predictions in flight.
        // Results come back in component order, so everything downstream sees the same sequence.
        let fbankModel = models.fbankModel
        let sampleCount = audioSource.sampleCount
        // The chunk handler cannot throw: it keeps the failure and stops emitting chunks.
        let splitFailure = OSAllocatedUnfairLock<Error?>(initialState: nil)
        let (produced, streamed) = try await orderedStreamingMap(
            maxConcurrency: componentEmbeddingConcurrency,
            produce: { emit in
                let segmentation = try await OfflineSegmentationProcessor().process(
                    audioSource: audioSource, segmentationModel: models.segmentationModel, config: config,
                    chunkHandler: { chunk in
                        progressCallback?(chunk.chunkIndex + 1, total)
                        do {
                            let components = try splitComponents(
                                chunk: chunk.chunkIndex, masks: chunk.speakerWeights,
                                offsetSeconds: chunk.chunkOffsetSeconds, frameDuration: chunk.frameDuration,
                                config: config)
                            for component in components where isEmbeddable(component, sampleCount: sampleCount) {
                                emit(component)
                            }
                            return .continue
                        } catch {
                            splitFailure.withLock { $0 = error }
                            return .stop
                        }
                    }
                )
                if let failure = splitFailure.withLock({ $0 }) { throw failure }
                return (segmentation: segmentation, end: Date())
            },
            transform: { (component: Component) in
                let raw = try autoreleasepool {
                    try embed(component, audioSource: audioSource, fbank: fbankModel, embedding: embeddingModel)
                }
                return (component: component, raw: raw)
            }
        )
        let segmentation = produced.segmentation
        guard segmentation.numFrames == 589, segmentation.numSpeakers == 3 else {
            throw OfflineDiarizationError.processingFailed("Unexpected Community-1 segmentation geometry")
        }
        let segmentationSeconds = produced.end.timeIntervalSince(start)
        let components = try splitComponents(segmentation, config: config)
        let slotCount = max(1, components.map { $0.slot + 1 }.max() ?? 0)
        var weights = Array(
            repeating: Array(repeating: Array(repeating: Float.zero, count: slotCount), count: 589),
            count: segmentation.numChunks
        )
        var embeddable: [Component] = []
        for component in components {
            try Task.checkCancellation()
            for frame in component.activeFrames { weights[component.chunk][frame][component.slot] = 1 }
            guard isEmbeddable(component, sampleCount: sampleCount) else { continue }
            embeddable.append(component)
        }
        // The streamed embeddings are valid only for exactly these components, in this order.
        guard streamed.map({ $0.component }) == embeddable else {
            throw OfflineDiarizationError.processingFailed(
                "Streamed speech components differ from the finished segmentation")
        }
        let embedded = streamed.map { $0.raw }
        var validComponents: [Component] = []
        var rawEmbeddings: [[Float]] = []
        for (component, raw) in zip(embeddable, embedded) {
            if let raw {
                validComponents.append(component)
                rawEmbeddings.append(raw)
            }
        }
        let plda = PLDATransform(pldaRhoModel: models.pldaRhoModel, psi: models.pldaPsi)
        let scaledRho = try await plda.transform(rawEmbeddings)
        var timed: [TimedEmbedding] = []
        var training: [Int] = []
        var intervals: [[Range<Int>]] = []
        for (index, component) in validComponents.enumerated() {
            guard let firstFrame = component.activeFrames.first,
                let lastFrame = component.activeFrames.last
            else {
                throw OfflineDiarizationError.processingFailed("Speech component has no activity")
            }
            let raw = rawEmbeddings[index]
            var sum: Float = 0
            vDSP_svesq(raw, 1, &sum, vDSP_Length(raw.count))
            let magnitude = max(sqrt(sum), 1e-4)
            let rho = zip(scaledRho[index], models.pldaPsi).map { $0 / sqrt($1) }
            guard rho.allSatisfy(\.isFinite) else {
                throw OfflineDiarizationError.processingFailed("Component PLDA returned non-finite features")
            }
            timed.append(
                TimedEmbedding(
                    chunkIndex: component.chunk, speakerIndex: component.slot,
                    startFrame: firstFrame, endFrame: lastFrame + 1,
                    frameWeights: component.cleanMask,
                    startTime: Double(component.sampleRange.lowerBound) / 16_000,
                    endTime: Double(component.sampleRange.upperBound) / 16_000,
                    embedding256: raw.map { $0 / magnitude }, rho128: rho
                ))
            if component.trainingEligible {
                training.append(index)
                intervals.append(component.cleanIntervals)
            }
        }
        // With no long clean turn, retain the existing manager's finite-vector
        // fallback instead of declaring real short speech to be silence.
        if training.isEmpty {
            training = Array(timed.indices)
            intervals = validComponents.map(\.cleanIntervals)
        }
        return PreparedDiarization(
            audioSource: audioSource,
            segmentation: SegmentationOutput(
                logProbs: [], speakerWeights: weights, numChunks: segmentation.numChunks,
                numFrames: 589, numSpeakers: slotCount, chunkOffsets: segmentation.chunkOffsets,
                frameDuration: segmentation.frameDuration
            ),
            timedEmbeddings: timed, audioLoadingSeconds: audioLoadingSeconds,
            segmentationSeconds: segmentationSeconds,
            // Embedding overlaps segmentation: this is the time it still ran after segmentation
            // ended, so the stages still add up to the wall time.
            embeddingExtractionSeconds: Date().timeIntervalSince(produced.end),
            prepareWallSeconds: Date().timeIntervalSince(start),
            componentTraining: SpeechComponentTraining(
                indices: training, cleanIntervals: intervals,
                minimumSharedSamples: Int(ceil(Double(config.embedding.minimumActiveRatio) * 589)) * 270
            )
        )
    }

    /// The component model runs on CPU with a variable-length input. Leave two cores for this
    /// file's segmentation, which overlaps embedding, and for the other track, ASR and the UI.
    static var componentEmbeddingConcurrency: Int {
        max(1, min(8, ProcessInfo.processInfo.activeProcessorCount - 2))
    }

    /// Transforms every item `produce` emits, starting while `produce` is still running, with at
    /// most `maxConcurrency` transforms in flight. The results come back in emission order, exactly
    /// as a sequential loop over the emitted items would return them. `emit` may only be called
    /// while `produce` runs. The first failure on either side cancels the other side.
    static func orderedStreamingMap<Item: Sendable, Produced: Sendable, T: Sendable>(
        maxConcurrency: Int,
        produce: @escaping @Sendable (_ emit: @escaping @Sendable (Item) -> Void) async throws -> Produced,
        transform: @escaping @Sendable (Item) throws -> T
    ) async throws -> (produced: Produced, results: [T]) {
        let (items, continuation) = AsyncStream.makeStream(of: Item.self)
        let outcome = try await withThrowingTaskGroup(of: StreamingWorkerResult<Produced, T>.self) { workers in
            workers.addTask {
                defer { continuation.finish() }
                return .produced(try await produce { continuation.yield($0) })
            }
            workers.addTask {
                .transformed(try await transformInOrder(items, maxConcurrency: maxConcurrency, transform))
            }
            var produced: Produced?
            var results: [T]?
            while let finished = try await workers.next() {
                switch finished {
                case .produced(let value):
                    produced = value
                case .transformed(let values):
                    results = values
                }
            }
            return (produced: produced, results: results)
        }
        // A cancelled caller stops reading the stream early: never return partial results.
        try Task.checkCancellation()
        guard let produced = outcome.produced, let results = outcome.results else {
            throw OfflineDiarizationError.processingFailed("Streamed work finished without a result")
        }
        return (produced, results)
    }

    private enum StreamingWorkerResult<Produced: Sendable, T: Sendable>: Sendable {
        case produced(Produced)
        case transformed([T])
    }

    /// Transforms streamed items with at most `maxConcurrency` in flight, each starting as soon as
    /// a lane is free, and returns the results in arrival order.
    private static func transformInOrder<Item: Sendable, T: Sendable>(
        _ items: AsyncStream<Item>, maxConcurrency: Int, _ transform: @escaping @Sendable (Item) throws -> T
    ) async throws -> [T] {
        let lanes = max(1, maxConcurrency)
        return try await withThrowingTaskGroup(of: (Int, T).self) { group in
            var results: [T?] = []
            var inFlight = 0
            for await item in items {
                if inFlight == lanes, let (index, value) = try await group.next() {
                    results[index] = value
                    inFlight -= 1
                }
                let index = results.count
                results.append(nil)
                group.addTask {
                    try Task.checkCancellation()
                    return (index, try transform(item))
                }
                inFlight += 1
            }
            while let (index, value) = try await group.next() {
                results[index] = value
            }
            let ordered = results.compactMap { $0 }
            guard ordered.count == results.count else {
                throw OfflineDiarizationError.processingFailed("A streamed transform returned no result")
            }
            return ordered
        }
    }

    /// Splits every chunk of a finished segmentation, in chunk order.
    private static func splitComponents(
        _ segmentation: SegmentationOutput, config: OfflineDiarizerConfig
    ) throws -> [Component] {
        var components: [Component] = []
        for chunk in 0..<segmentation.numChunks {
            components += try splitComponents(
                chunk: chunk, masks: segmentation.speakerWeights[chunk],
                offsetSeconds: segmentation.chunkOffsets[chunk], frameDuration: segmentation.frameDuration,
                config: config)
        }
        return components
    }

    /// Splits one segmentation chunk into its speech components, in slot order. Streaming and
    /// the finished segmentation both split through here, so they yield the same components.
    private static func splitComponents(
        chunk: Int, masks: [[Float]], offsetSeconds: Double, frameDuration: Double,
        config: OfflineDiarizerConfig
    ) throws -> [Component] {
        guard masks.count == 589, masks.allSatisfy({ $0.count == 3 }) else {
            throw OfflineDiarizationError.processingFailed("Unexpected Community-1 segmentation geometry")
        }
        let clean = masks.map { $0.reduce(0, +) == 1 }
        let offset = Int((offsetSeconds * 16_000).rounded())
        var components: [Component] = []
        var slot = 0
        for local in 0..<3 {
            var groups: [[Int]] = []
            for frame in 0..<589 where masks[frame][local] > 0 {
                if let previous = groups.last?.last,
                    Double(frame - previous - 1) * frameDuration <= config.minGapDuration
                {
                    groups[groups.count - 1].append(frame)
                } else {
                    groups.append([frame])
                }
            }
            for active in groups {
                guard let first = active.first, let last = active.last else {
                    throw OfflineDiarizationError.processingFailed("Speech component has no activity")
                }
                let mask: [Float] = (first...last).map { clean[$0] ? masks[$0][local] : 0 }
                var cleanRanges: [Range<Int>] = []
                for frame in first...last where clean[frame] && masks[frame][local] > 0 {
                    let range = (offset + frame * 270)..<(offset + (frame + 1) * 270)
                    if let previous = cleanRanges.last, previous.upperBound == range.lowerBound {
                        cleanRanges[cleanRanges.count - 1] = previous.lowerBound..<range.upperBound
                    } else {
                        cleanRanges.append(range)
                    }
                }
                components.append(
                    Component(
                        chunk: chunk, slot: slot, activeFrames: active, cleanMask: mask,
                        sampleRange: (offset + first * 270)..<(offset + (last + 1) * 270),
                        cleanIntervals: cleanRanges,
                        trainingEligible: mask.reduce(0, +) >= config.embedding.minimumActiveRatio * 589
                    ))
                slot += 1
            }
        }
        return components
    }

    /// No arbitrary minimum turn length: only complete real FBank frames and at least one clean
    /// pooling weight can produce an identity.
    private static func isEmbeddable(_ component: Component, sampleCount: Int) -> Bool {
        component.sampleRange.lowerBound >= 0 && component.sampleRange.upperBound <= sampleCount
            && component.sampleRange.count >= 560 && component.cleanMask.contains(1)
    }

    private static func embed(
        _ component: Component, audioSource: AudioSampleSource, fbank: MLModel, embedding: MLModel
    ) throws -> [Float]? {
        let count = component.sampleRange.count
        // Frame 997 starts at sample 159520. Both final FBank frames must be
        // wholly zero for the pinned FBank's additive-floor inversion below.
        guard count <= 159_030 else {
            throw OfflineDiarizationError.processingFailed("Speech component exceeds its segmentation window")
        }
        let frames = 1 + (count - 400) / 160
        guard let poolingWeights = cleanPoolingWeights(component.cleanMask, featureFrames: frames) else { return nil }
        let poolFrames = poolingWeights.count
        let input = try MLMultiArray(shape: [1, 1, 160_000], dataType: .float32)
        let audio = input.dataPointer.assumingMemoryBound(to: Float.self)
        audio.initialize(repeating: 0, count: 160_000)
        try audioSource.copySamples(into: audio, offset: component.sampleRange.lowerBound, count: count)
        let output = try fbank.prediction(from: MLDictionaryFeatureProvider(dictionary: ["audio": input]))
        guard let bank = output.featureValue(for: "fbank_features")?.multiArrayValue,
            bank.shape.map(\.intValue) == [1, 1, 80, 998], bank.dataType == .float32
        else { throw OfflineDiarizationError.processingFailed("Unexpected component FBank output") }
        let features = try MLMultiArray(shape: [1, NSNumber(value: frames), 80], dataType: .float32)
        let featureValues = features.dataPointer.assumingMemoryBound(to: Float.self)
        let bankValues = bank.dataPointer.assumingMemoryBound(to: Float.self)
        let bandStride = bank.strides[2].intValue
        let frameStride = bank.strides[3].intValue
        let epsilonNative = Double(Float(1e-6))
        let epsilonSource = Double(Float.ulpOfOne)
        for band in 0..<80 {
            let base = band * bandStride
            let anchor = Double(bankValues[base + 997 * frameStride])
            guard anchor.isFinite, Double(bankValues[base + 996 * frameStride]) == anchor else {
                throw OfflineDiarizationError.processingFailed("Component FBank has no zero-tail anchor")
            }
            var logs = [Double](repeating: 0, count: frames)
            var sum: Double = 0
            for frame in 0..<frames {
                let value = Double(bankValues[base + frame * frameStride])
                let mel = epsilonNative * expm1(value - anchor)
                guard mel.isFinite else { throw OfflineDiarizationError.processingFailed("Non-finite component FBank") }
                logs[frame] = log(max(mel, epsilonSource))
                sum += logs[frame]
            }
            let mean = sum / Double(frames)
            for frame in 0..<frames { featureValues[frame * 80 + band] = Float(logs[frame] - mean) }
        }
        let weights = try MLMultiArray(shape: [1, NSNumber(value: poolFrames)], dataType: .float32)
        let weightValues = weights.dataPointer.assumingMemoryBound(to: Float.self)
        for frame in 0..<poolFrames {
            weightValues[frame] = poolingWeights[frame]
        }
        let result = try embedding.prediction(
            from: MLDictionaryFeatureProvider(dictionary: ["features": features, "weights": weights]))
        guard let array = result.featureValue(for: "embedding")?.multiArrayValue,
            array.shape.map(\.intValue) == [1, 256]
        else { throw OfflineDiarizationError.processingFailed("Unexpected component embedding output") }
        let vector = (0..<256).map { array[$0].floatValue }
        // Degenerate real spans can have no identity. Their activity stays in
        // reconstruction; an unusable vector must never contaminate clustering.
        guard vector.allSatisfy(\.isFinite), vector.contains(where: { $0 != 0 }) else { return nil }
        return vector
    }

    static func cleanPoolingWeights(_ mask: [Float], featureFrames: Int) -> [Float]? {
        guard !mask.isEmpty, featureFrames >= 2 else { return nil }
        let count = (featureFrames + 7) / 8
        // Exact nearest interpolation. Zero pooled support can still produce a
        // finite bias vector, which is not evidence of a speaker's identity.
        let weights = (0..<count).map { mask[$0 * mask.count / count] }
        return weights.contains(where: { $0 > 0 }) ? weights : nil
    }
}

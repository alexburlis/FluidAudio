# Awaited Parakeet streaming input

`SlidingWindowAsrManager` has an additive input path for applications that already own an ordered 16 kHz mono audio feed. It bypasses the legacy PCM `AsyncStream` and accepts copied `[Float]` values.

```swift
let manager = SlidingWindowAsrManager(config: selectedConfiguration)
try await manager.loadModels(selectedPreparedModels)
try await manager.startOrderedStreaming(source: .microphone)
let updates = await manager.transcriptionUpdates
// Consume updates in one task owned by the recording session.
// Each actual update contains complete confirmedTranscript/volatileTranscript snapshots.

// In the sole ordered feed consumer, outside the audio callback:
try await manager.appendAudioSamples(ownedNormalizedSamples)

// After stopping capture and draining the application's bounded feed:
let finalText = try await manager.finishOrderedStreaming()
```

Every append waits for its ready decode windows. There is at most one accepted append in flight, capped at 240,000 samples; concurrent or oversized appends throw `bufferOverflow` before acceptance. Normal microphone adapters should send much smaller frames. Non-finite samples fail before acceptance. An application still needs a bounded queue between its non-blocking capture callback and its sole awaited consumer. It must surface overflow instead of dropping samples or switching recognition paths.

`orderedAudioProgress` reports accepted, processed, in-flight, and high-water sample counts. Decoder context overlap is not new accepted input. The transcript fields on each update are an atomic display snapshot, so clients need not read two actor properties separately. They remain optional for compatibility with manually created legacy update values.

`finishOrderedStreaming()` closes acceptance, waits for accepted decode work, flushes the true audio tail, and finishes the update stream after the final preview. Repeated finish calls share the same terminal task. Any model window failure throws `incompleteTranscription(failedWindows:)`; incomplete text is never returned as complete success. Existing `finish()` also routes to this completion contract after an ordered start.

`cancel()` closes updates immediately, cancels and awaits the owned decoder tasks, and then clears pending samples and transcript state. Subsequent finish throws `CancellationError`. Core ML may finish a current call before cancellation returns; callers must await this barrier before releasing their recording lease. A new recording uses a new manager. Do not call the legacy `streamAudio` API on an ordered session; that marks completion failed.

The decode schedule is still controlled by `chunkSeconds` plus `rightContextSeconds`. `hypothesisChunkSeconds` does not create a second early-decoding track. Early-preview configuration must be evaluated together with final-word and recognition-quality budgets on real speech. These APIs alone make no latency or accuracy claim.

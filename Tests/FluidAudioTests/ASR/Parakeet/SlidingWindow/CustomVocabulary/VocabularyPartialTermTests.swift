import XCTest

@testable import FluidAudio

/// A recognized ordinary word that is only the beginning or end of a vocabulary term
/// ("speech" for "Paraspeech") must not be replaced by the term; a span whose other words
/// resemble the rest of the term ("para speech", "super base") still can be.
final class VocabularyPartialTermTests: XCTestCase {
    private func isPartial(_ span: String, _ forms: [String]) -> Bool {
        VocabularyRescorer.isPartialTermSpan(
            spanWords: span.split(separator: " ").map(String.init),
            termForms: forms
        )
    }

    func testOrdinaryWordThatEndsTheTermIsPartial() {
        XCTAssertTrue(isPartial("speech", ["paraspeech"]))
        XCTAssertTrue(isPartial("audio", ["fluidaudio"]))
        XCTAssertTrue(isPartial("buddy", ["note buddy"]))
    }

    func testOrdinaryWordThatStartsTheTermIsPartial() {
        XCTAssertTrue(isPartial("post", ["posthog"]))
        XCTAssertTrue(isPartial("fluid", ["fluidaudio"]))
    }

    func testUnrelatedWordsBeforeTheTailAreStillPartial() {
        XCTAssertTrue(isPartial("use speech", ["paraspeech"]))
        XCTAssertTrue(isPartial("the speech", ["paraspeech"]))
        XCTAssertTrue(isPartial("to speech", ["paraspeech"]))
        XCTAssertTrue(isPartial("her speech", ["paraspeech"]))
        XCTAssertTrue(isPartial("a speech", ["paraspeech"]))
    }

    func testSplitTermWhoseOtherPartResemblesTheTermStaysACandidate() {
        XCTAssertFalse(isPartial("para speech", ["paraspeech"]))
        XCTAssertFalse(isPartial("pair speech", ["paraspeech"]))
        XCTAssertFalse(isPartial("super base", ["supabase"]))
        XCTAssertFalse(isPartial("post hog", ["posthog"]))
        XCTAssertFalse(isPartial("nope buddy", ["note buddy"]))
    }

    func testSplitTermWhoseOtherPartSoundsLikeTheTermStaysACandidate() {
        // A recorded call transcribed "dictating with Paraspeech" as "dictating with fair speech":
        // "fair" shares the consonant sounds and a vowel of "para", so acoustic scoring decides.
        XCTAssertFalse(isPartial("fair speech", ["paraspeech"]))
        XCTAssertFalse(isPartial("for a speech", ["paraspeech"]))
    }

    func testOrdinaryPhrasesWithSimilarConsonantsStayPartial() {
        XCTAssertTrue(isPartial("for speech", ["paraspeech"]))
        XCTAssertTrue(isPartial("free speech", ["paraspeech"]))
        XCTAssertTrue(isPartial("four speech", ["paraspeech"]))
    }

    func testMisrecognitionThatIsNotPartOfTheTermStaysACandidate() {
        XCTAssertFalse(isPartial("superbass", ["supabase"]))
        XCTAssertFalse(isPartial("million", ["dillion"]))
    }

    func testExactFormOrAliasIsNeverPartial() {
        XCTAssertFalse(isPartial("paraspeech", ["paraspeech"]))
        XCTAssertFalse(isPartial("speech", ["paraspeech", "speech"]))
        XCTAssertFalse(isPartial("note buddy", ["note buddy"]))
    }

    func testShortWordsAndInnerPiecesAreLeftToTheExistingRules() {
        XCTAssertFalse(isPartial("aud", ["fluidaudio"]))
        XCTAssertFalse(isPartial("spee", ["paraspeechpro"]))
    }

    func testNearCompleteTruncationStaysACandidate() {
        // Fewer than three missing characters is a clipped or misheard term, not another word:
        // acoustic scoring decides, as before.
        XCTAssertFalse(isPartial("supabas", ["supabase"]))
        XCTAssertFalse(isPartial("paraspeec", ["paraspeech"]))
        XCTAssertFalse(isPartial("entry", ["sentry"]))
        XCTAssertFalse(isPartial("linked", ["linkedin"]))
    }

    func testUnrelatedAliasDoesNotSwitchOffTheGuard() {
        // "paraspeak" shares no edge with "speech", so it must not count as a reason to keep it.
        XCTAssertTrue(isPartial("speech", ["paraspeech", "paraspeak"]))
        XCTAssertTrue(isPartial("use speech", ["paraspeech", "paraspeak"]))
    }

    func testCompoundThatResemblesAnyFormStaysACandidate() {
        XCTAssertFalse(isPartial("para speech", ["paraspeech", "paraspeak"]))
        XCTAssertFalse(isPartial("superbass", ["supabase", "supa base"]))
    }

    func testPartialForOneFormButCloseToAnotherAliasStaysACandidate() {
        // "speech" is the tail of the canonical form, but an alias the user saved as
        // "speech one" begins with it and the span matches that alias's start only.
        XCTAssertTrue(isPartial("speech", ["paraspeech", "speechone"]))
        XCTAssertFalse(isPartial("speech one", ["paraspeech", "speech one"]))
    }
}

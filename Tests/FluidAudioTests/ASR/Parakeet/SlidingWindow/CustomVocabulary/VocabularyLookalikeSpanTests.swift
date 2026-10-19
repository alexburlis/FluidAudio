import XCTest

@testable import FluidAudio

/// A recognized word that rhymes with a one-word term but starts with another sound
/// ("million" for "Dillion") is a different word and must not be replaced by the term.
final class VocabularyLookalikeSpanTests: XCTestCase {
    private func words(_ span: String) -> [String] {
        span.split(separator: " ").map(String.init)
    }

    private func isRhyming(_ span: String, _ forms: [String]) -> Bool {
        VocabularyRescorer.isRhymingWordSpan(spanWords: words(span), termForms: forms)
    }

    private func isLookalike(_ span: String, _ forms: [String]) -> Bool {
        VocabularyRescorer.isLookalikeSpan(spanWords: words(span), termForms: forms)
    }

    func testRhymeWithAnotherFirstSoundIsAnotherWord() {
        // Recorded calls turned "two million" and "around 3 million" into "Dillion".
        XCTAssertTrue(isRhyming("million", ["dillion"]))
        XCTAssertTrue(isRhyming("billion", ["dillion"]))
        XCTAssertTrue(isRhyming("trillion", ["dillion"]))
        XCTAssertTrue(isRhyming("ran", ["cran"]))
    }

    func testSpanHoldingARhymeWithAnotherFirstSoundIsAnotherPhrase() {
        // With "million" rejected, the next candidates were "two million" and "3 million".
        XCTAssertTrue(isRhyming("two million", ["dillion"]))
        XCTAssertTrue(isRhyming("3 million", ["dillion"]))
        XCTAssertTrue(isRhyming("million views", ["dillion"]))
    }

    func testSpellingOfTheSameFirstSoundStaysACandidate() {
        XCTAssertFalse(isRhyming("cubernetes", ["kubernetes"]))
        XCTAssertFalse(isRhyming("phone", ["fone"]))
    }

    func testMisrecognitionsThatDoNotRhymeStayCandidates() {
        XCTAssertFalse(isRhyming("dillon", ["dillion"]))
        XCTAssertFalse(isRhyming("superbass", ["supabase"]))
        XCTAssertFalse(isRhyming("dillions", ["dillion"]))
        XCTAssertFalse(isRhyming("dill ion", ["dillion"]))
        XCTAssertFalse(isRhyming("super base", ["supabase"]))
    }

    func testExactFormsAndMultiWordFormsAreLeftToOtherRules() {
        XCTAssertFalse(isRhyming("dillion", ["dillion"]))
        XCTAssertFalse(isRhyming("million", ["dillion", "million"]))
        XCTAssertFalse(isRhyming("bote", ["note buddy"]))
    }

    func testLookalikeCoversPartialAndRhymingSpans() {
        XCTAssertTrue(isLookalike("speech", ["paraspeech"]))
        XCTAssertTrue(isLookalike("million", ["dillion"]))
        XCTAssertFalse(isLookalike("fair speech", ["paraspeech"]))
        XCTAssertFalse(isLookalike("superbass", ["supabase"]))
    }
}

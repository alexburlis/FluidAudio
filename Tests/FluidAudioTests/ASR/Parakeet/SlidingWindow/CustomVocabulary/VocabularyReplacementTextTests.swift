import XCTest

@testable import FluidAudio

/// A vocabulary replacement keeps the punctuation around the replaced words and a possessive
/// "'s": "Superbass?" becomes "Supabase?", not "Supabase".
final class VocabularyReplacementTextTests: XCTestCase {
    private func text(_ replacement: String, _ first: String, _ last: String? = nil) -> String {
        VocabularyRescorer.replacementText(replacement, firstWord: first, lastWord: last ?? first)
    }

    func testPlainWordTakesTheReplacement() {
        XCTAssertEqual(text("Supabase", "Superbass"), "Supabase")
    }

    func testTrailingPunctuationStays() {
        XCTAssertEqual(text("Supabase", "Superbass?"), "Supabase?")
        XCTAssertEqual(text("Paraspeech", "Paraspeech,"), "Paraspeech,")
        XCTAssertEqual(text("Paraspeech", "fair", "speech."), "Paraspeech.")
        XCTAssertEqual(text("Supabase", "Superbass?\""), "Supabase?\"")
    }

    func testLeadingPunctuationStays() {
        XCTAssertEqual(text("Supabase", "\"Superbass", "base\""), "\"Supabase\"")
        XCTAssertEqual(text("Supabase", "(Superbass)"), "(Supabase)")
    }

    func testPossessiveStays() {
        XCTAssertEqual(text("Dillion", "Dillion's"), "Dillion's")
        XCTAssertEqual(text("Dillion", "Dilion’s,"), "Dillion’s,")
        XCTAssertEqual(text("Note Buddy", "nope", "buddy's"), "Note Buddy's")
    }

    func testPunctuationTheReplacementAlreadyHasIsNotRepeated() {
        XCTAssertEqual(text("Yahoo!", "Yahu!"), "Yahoo!")
    }
}

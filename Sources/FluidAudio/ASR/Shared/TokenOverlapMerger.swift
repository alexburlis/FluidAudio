/// Merges token streams from adjacent overlapping audio windows.
enum TokenOverlapMerger {

    /// Searches the prefix's tail and suffix's head for their longest common substring.
    /// Keeps the prefix and drops the suffix through the matched run; insufficient matches concatenate plainly.
    static func merge(
        prefix: [Int],
        suffix: [Int],
        windowTokens: Int = 32,
        minMatch: Int = 4
    ) -> [Int] {
        if prefix.isEmpty { return suffix }
        if suffix.isEmpty { return prefix }

        let pTail = Array(prefix.suffix(windowTokens))
        let sHead = Array(suffix.prefix(windowTokens))
        let m = pTail.count
        let n = sHead.count
        if m == 0 || n == 0 { return prefix + suffix }

        // Classic LCS-substring DP (O(m·n), m,n ≤ windowTokens).
        var dp = [Int](repeating: 0, count: n + 1)
        var bestLen = 0
        var bestSEnd = 0  // index in sHead (exclusive) where the match ends
        for i in 1...m {
            var prev = 0
            for j in 1...n {
                let temp = dp[j]
                if pTail[i - 1] == sHead[j - 1] {
                    dp[j] = prev + 1
                    if dp[j] > bestLen {
                        bestLen = dp[j]
                        bestSEnd = j
                    }
                } else {
                    dp[j] = 0
                }
                prev = temp
            }
        }

        guard bestLen >= minMatch else { return prefix + suffix }
        return prefix + Array(suffix.dropFirst(bestSEnd))
    }
}

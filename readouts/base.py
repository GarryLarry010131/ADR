"""Readout interfaces. Scoring readouts return one scalar per caption; `kind` is "signed" (canonical zero, decide by
the sign), "swap" (decide by score minus the subject-object swapped score; implement swap_score) or "score" (oracle
threshold on true/false sets). Chat readouts return a credit in [0, 1]. Coverage is decided by the protocol from
the set's caption parse, not by the readout."""


class ScoringReadout:
    kind = "score"

    def scores(self, img, captions):
        raise NotImplementedError

    def swap_score(self, img, caption):
        raise NotImplementedError


class ChatReadout:
    def choose(self, img, captions, correct):
        raise NotImplementedError

    def yesno(self, img, statement, label):
        raise NotImplementedError

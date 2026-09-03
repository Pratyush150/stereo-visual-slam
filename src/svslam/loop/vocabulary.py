"""A small binary bag-of-words vocabulary over ORB descriptors.

ORB descriptors are 256-bit strings, so the clustering that builds a visual
vocabulary has to work in Hamming space: distances are popcounts, and a cluster
centre is the **bitwise majority** of its members, not their arithmetic mean.
Averaging binary descriptors as floats and rounding is a common shortcut and it
produces centres that no real descriptor is close to.

The vocabulary is intentionally flat (one level of k-means) rather than the
hierarchical tree of DBoW2.  For a few hundred keyframes a flat vocabulary of a
few hundred words is fast enough and much easier to verify.  Words are weighted
by inverse document frequency, so a word that fires on every keyframe -- road
texture, sky boundary -- contributes almost nothing to the similarity score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["POPCOUNT8", "hamming_distances", "BagOfWords", "train_vocabulary"]

#: Number of set bits in each byte value, for vectorised Hamming distances.
POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_distances(descriptors: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances, shape ``(N, K)``.

    Computed as a popcount of the XOR through a 256-entry byte lookup table,
    which is a great deal faster than unpacking every descriptor to bits.
    """
    d = np.ascontiguousarray(descriptors, dtype=np.uint8).reshape(descriptors.shape[0], -1)
    c = np.ascontiguousarray(centres, dtype=np.uint8).reshape(centres.shape[0], -1)
    out = np.zeros((d.shape[0], c.shape[0]), dtype=np.int32)
    # Chunked so the (N, K, 32) intermediate never becomes enormous.
    chunk = max(1, int(4_000_000 / max(c.shape[0] * c.shape[1], 1)))
    for start in range(0, d.shape[0], chunk):
        block = d[start:start + chunk]
        xor = np.bitwise_xor(block[:, None, :], c[None, :, :])
        out[start:start + chunk] = POPCOUNT8[xor].sum(axis=2)
    return out


def _majority_bits(descriptors: np.ndarray) -> np.ndarray:
    """Bitwise majority vote over a set of binary descriptors."""
    bits = np.unpackbits(np.ascontiguousarray(descriptors, dtype=np.uint8), axis=1)
    majority = (bits.mean(axis=0) >= 0.5).astype(np.uint8)
    return np.packbits(majority)


@dataclass
class BagOfWords:
    """A trained vocabulary plus its inverse-document-frequency weights."""

    centres: np.ndarray  # (K, 32) uint8
    idf: np.ndarray  # (K,) float

    @property
    def size(self) -> int:
        return int(self.centres.shape[0])

    def words(self, descriptors: np.ndarray) -> np.ndarray:
        """Assign each descriptor to its nearest word."""
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(0, dtype=int)
        return np.argmin(hamming_distances(descriptors, self.centres), axis=1)

    def vector(self, descriptors: np.ndarray) -> np.ndarray:
        """TF-IDF bag-of-words vector, L1-normalised.

        L1 normalisation pairs with the L1 similarity score below; it also means
        a keyframe with 400 features and one with 800 are directly comparable.
        """
        v = np.zeros(self.size, dtype=float)
        words = self.words(descriptors)
        if words.size == 0:
            return v
        np.add.at(v, words, 1.0)
        v *= self.idf
        total = v.sum()
        return v / total if total > 0 else v

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """DBoW2's L1 score: ``1 - 0.5 * ||a - b||_1`` for L1-normalised vectors.

        Ranges from 0 (no words in common) to 1 (identical distributions).
        """
        return float(1.0 - 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())


def train_vocabulary(
    descriptors: np.ndarray,
    n_words: int = 400,
    iterations: int = 12,
    seed: int = 0,
    n_documents: int | None = None,
    document_words: list[np.ndarray] | None = None,
) -> BagOfWords:
    """k-means in Hamming space over a descriptor sample.

    Initialisation is k-means++ in Hamming distance, which matters: with random
    initialisation a good fraction of words end up empty and the vocabulary is
    effectively smaller than requested.

    ``idf`` is computed from ``document_words`` when given (one array of word
    ids per training keyframe), otherwise from the cluster sizes.
    """
    descriptors = np.ascontiguousarray(descriptors, dtype=np.uint8)
    n = descriptors.shape[0]
    n_words = int(min(max(n_words, 1), n))
    rng = np.random.default_rng(seed)

    # k-means++ seeding.
    centres = np.zeros((n_words, descriptors.shape[1]), dtype=np.uint8)
    first = int(rng.integers(n))
    centres[0] = descriptors[first]
    closest = hamming_distances(descriptors, centres[:1]).reshape(-1).astype(float)
    for k in range(1, n_words):
        weights = closest ** 2
        total = weights.sum()
        if total <= 0:
            centres[k] = descriptors[int(rng.integers(n))]
        else:
            centres[k] = descriptors[int(rng.choice(n, p=weights / total))]
        d_new = hamming_distances(descriptors, centres[k:k + 1]).reshape(-1).astype(float)
        closest = np.minimum(closest, d_new)

    assignment = np.zeros(n, dtype=int)
    for _ in range(max(int(iterations), 1)):
        assignment = np.argmin(hamming_distances(descriptors, centres), axis=1)
        changed = False
        for k in range(n_words):
            members = descriptors[assignment == k]
            if members.shape[0] == 0:
                continue
            new_centre = _majority_bits(members)
            if not np.array_equal(new_centre, centres[k]):
                centres[k] = new_centre
                changed = True
        if not changed:
            break

    if document_words is not None and document_words:
        n_docs = len(document_words)
        doc_freq = np.zeros(n_words, dtype=float)
        for words in document_words:
            doc_freq[np.unique(words)] += 1.0
    else:
        n_docs = int(n_documents or n)
        doc_freq = np.bincount(assignment, minlength=n_words).astype(float)

    # log(N / n_i); words present in every document get weight ~0.
    idf = np.log(np.maximum(n_docs, 1) / np.maximum(doc_freq, 1e-9))
    idf = np.clip(idf, 0.0, None)
    return BagOfWords(centres=centres, idf=idf)

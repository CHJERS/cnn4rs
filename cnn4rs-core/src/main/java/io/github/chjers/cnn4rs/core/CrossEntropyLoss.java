package io.github.chjers.cnn4rs.core;

import java.util.Objects;

/** Cross-entropy for a rank-one probability vector and one class label. */
public final class CrossEntropyLoss {
    private static final float MINIMUM_PROBABILITY = 1.0e-7f;

    public float calculate(Tensor probabilities, int label) {
        Objects.requireNonNull(probabilities, "probabilities must not be null");
        Conv2D.requireRank(probabilities.shape(), 1, "cross-entropy probabilities");
        int classCount = probabilities.shape().dimension(0);
        if (label < 0 || label >= classCount) {
            throw new IllegalArgumentException("label " + label
                    + " is outside class count " + classCount);
        }
        float probability = probabilities.get(label);
        if (!Float.isFinite(probability) || probability < 0f || probability > 1f) {
            throw new IllegalArgumentException("class probability must be finite and between 0 and 1"
                    + " but was " + probability);
        }
        return (float) -Math.log(Math.max(probability, MINIMUM_PROBABILITY));
    }
}

package io.github.chjers.cnn4rs.core;

import java.util.Objects;

/** One classifier input and its zero-based class label. */
public record TrainingExample(Tensor input, int label) {
    public TrainingExample {
        Objects.requireNonNull(input, "training input must not be null");
        if (label < 0) {
            throw new IllegalArgumentException("training label must not be negative: " + label);
        }
    }
}

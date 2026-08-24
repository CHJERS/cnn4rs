package io.github.chjers.cnn4rs.core;

import java.util.Objects;

/** Converts a tensor with the expected element count to a rank-one tensor. */
public final class Flatten implements Layer {
    private final int inputElementCount;

    public Flatten(int inputElementCount) {
        if (inputElementCount <= 0) {
            throw new InvalidTensorException("flatten input element count must be positive");
        }
        this.inputElementCount = inputElementCount;
    }

    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "input must not be null");
        if (input.shape().elementCount() != inputElementCount) {
            throw new InvalidTensorException("flatten expects " + inputElementCount
                    + " input elements but got " + input.shape().elementCount());
        }
        return Tensor.of(input.toArray(), inputElementCount);
    }

    int inputElementCount() {
        return inputElementCount;
    }
}

package io.github.chjers.cnn4rs.core;

import java.util.Objects;

/** Elementwise rectified linear unit activation. */
public final class ReLU implements Layer {
    @Override
    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "input must not be null");
        float[] values = input.toArray();
        for (int index = 0; index < values.length; index++) {
            values[index] = Math.max(0f, values[index]);
        }
        return Tensor.of(values, input.shape().dimensions());
    }
}

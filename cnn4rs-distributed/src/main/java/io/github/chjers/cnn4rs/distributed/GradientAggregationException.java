package io.github.chjers.cnn4rs.distributed;

/** Indicates incompatible or unsafe inputs to experimental gradient aggregation. */
public final class GradientAggregationException extends IllegalArgumentException {
    public GradientAggregationException(String message) {
        super(message);
    }
}

package io.github.chjers.cnn4rs.core;

/** Indicates invalid tensor dimensions, values, or indices. */
public final class InvalidTensorException extends IllegalArgumentException {
    public InvalidTensorException(String message) {
        super(message);
    }
}

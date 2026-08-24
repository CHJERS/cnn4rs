package io.github.chjers.cnn4rs.core;

/** A deterministic transformation from one tensor to another. */
public interface Layer {
    Tensor forward(Tensor input);
}

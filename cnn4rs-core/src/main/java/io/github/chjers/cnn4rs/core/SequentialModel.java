package io.github.chjers.cnn4rs.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** A deterministic sequence of tensor layers. */
public final class SequentialModel {
    private final ModelConfig config;
    private final List<Layer> layers;

    public SequentialModel(ModelConfig config, List<? extends Layer> layers) {
        this.config = Objects.requireNonNull(config, "config must not be null");
        Objects.requireNonNull(layers, "layers must not be null");
        if (layers.size() != config.layers().size()) {
            throw new IllegalArgumentException("model has " + layers.size()
                    + " layers but its configuration describes " + config.layers().size());
        }
        ArrayList<Layer> copy = new ArrayList<>(layers.size());
        for (int index = 0; index < layers.size(); index++) {
            copy.add(Objects.requireNonNull(layers.get(index),
                    "layer " + index + " must not be null"));
        }
        this.layers = List.copyOf(copy);
    }

    public Tensor predict(Tensor input) {
        Objects.requireNonNull(input, "input must not be null");
        Tensor output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    ModelConfig config() {
        return config;
    }

    List<Layer> layers() {
        return layers;
    }
}

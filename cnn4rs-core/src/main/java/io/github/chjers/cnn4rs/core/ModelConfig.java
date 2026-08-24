package io.github.chjers.cnn4rs.core;

import java.util.List;
import java.util.Objects;

/** Immutable layer configuration used to initialize and persist a model. */
public final class ModelConfig {
    /** A layer type and its ordered integer configuration values. */
    public record LayerDescriptor(String type, List<Integer> parameters) {
        public LayerDescriptor {
            Objects.requireNonNull(type, "layer type must not be null");
            if (type.isBlank()) {
                throw new IllegalArgumentException("layer type must not be blank");
            }
            parameters = List.copyOf(Objects.requireNonNull(parameters,
                    "layer parameters must not be null"));
        }
    }

    private final List<LayerDescriptor> layers;
    private final int classCount;
    private final long seed;

    public ModelConfig(List<LayerDescriptor> layers, int classCount, long seed) {
        this.layers = List.copyOf(Objects.requireNonNull(layers, "layers must not be null"));
        if (this.layers.isEmpty()) {
            throw new IllegalArgumentException("model configuration must contain at least one layer");
        }
        if (classCount <= 0) {
            throw new IllegalArgumentException("class count must be positive");
        }
        this.classCount = classCount;
        this.seed = seed;
    }

    public List<LayerDescriptor> layers() {
        return layers;
    }

    public int classCount() {
        return classCount;
    }

    public long seed() {
        return seed;
    }
}

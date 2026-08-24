package io.github.chjers.cnn4rs.core;

import java.util.List;
import java.util.Random;

/** Deterministic model factories. */
public final class Models {
    static final String DENSE = "dense";
    static final String CONVOLUTION = "conv2d";
    static final String MAX_POOL = "maxPool2d";
    static final String FLATTEN = "flatten";
    static final String RELU = "relu";
    static final String SOFTMAX = "softmax";

    private Models() {
    }

    public static SequentialModel denseClassifier(int inputSize, int classCount, long seed) {
        if (inputSize <= 0 || classCount <= 0) {
            throw new IllegalArgumentException("classifier input size and class count must be positive");
        }
        ModelConfig config = new ModelConfig(List.of(
                descriptor(DENSE, inputSize, classCount),
                descriptor(SOFTMAX)), classCount, seed);
        return new SequentialModel(config, List.of(
                new Dense(inputSize, classCount, new Random(seed)),
                new Softmax()));
    }

    static ModelConfig.LayerDescriptor descriptor(String type, int... parameters) {
        return new ModelConfig.LayerDescriptor(type,
                java.util.Arrays.stream(parameters).boxed().toList());
    }
}

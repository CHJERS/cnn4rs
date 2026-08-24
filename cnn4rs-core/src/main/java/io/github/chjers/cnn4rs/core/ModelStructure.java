package io.github.chjers.cnn4rs.core;

import java.util.List;

/** Shared validation for configured classifier layer sequences. */
final class ModelStructure {
    private ModelStructure() {
    }

    static void validate(ModelConfig config, List<? extends Layer> layers) {
        if (layers.size() != config.layers().size()) {
            throw new IllegalArgumentException("model has " + layers.size()
                    + " layers but its configuration describes " + config.layers().size());
        }

        for (int index = 0; index < layers.size(); index++) {
            Layer layer = layers.get(index);
            ModelConfig.LayerDescriptor descriptor = config.layers().get(index);
            requireMatchingLayer(index, descriptor, layer);
        }
        validateDescriptors(config);
    }

    static void validateDescriptors(ModelConfig config) {
        OutputShape output = OutputShape.unknown();
        for (int index = 0; index < config.layers().size(); index++) {
            ModelConfig.LayerDescriptor descriptor = config.layers().get(index);
            output = validateAdjacency(index, descriptor, output);
        }

        ModelConfig.LayerDescriptor last = config.layers().get(config.layers().size() - 1);
        if (!Models.SOFTMAX.equals(last.type())) {
            throw new IllegalArgumentException("final classifier layer must be softmax");
        }
        if (output.kind != OutputKind.VECTOR || output.size != config.classCount()) {
            throw new IllegalArgumentException("final classifier output width "
                    + describeSize(output) + " does not match class count " + config.classCount());
        }
    }

    private static void requireMatchingLayer(int index, ModelConfig.LayerDescriptor descriptor,
                                             Layer layer) {
        List<Integer> values = descriptor.parameters();
        boolean matches = switch (descriptor.type()) {
            case Models.DENSE -> layer instanceof Dense dense && values.size() == 2
                    && values.get(0) == dense.inputSize() && values.get(1) == dense.outputSize();
            case Models.CONVOLUTION -> layer instanceof Conv2D convolution && values.size() == 4
                    && values.get(0) == convolution.inputChannels()
                    && values.get(1) == convolution.outputChannels()
                    && values.get(2) == convolution.kernelSize()
                    && values.get(3) == convolution.stride();
            case Models.MAX_POOL -> layer instanceof MaxPool2D pool && values.size() == 2
                    && values.get(0) == pool.windowSize() && values.get(1) == pool.stride();
            case Models.FLATTEN -> layer instanceof Flatten flatten && values.size() == 1
                    && values.get(0) == flatten.inputElementCount();
            case Models.RELU -> layer instanceof ReLU && values.isEmpty();
            case Models.SOFTMAX -> layer instanceof Softmax && values.isEmpty();
            default -> false;
        };
        if (!matches) {
            throw new IllegalArgumentException("layer " + index + " does not match descriptor "
                    + descriptor.type() + descriptor.parameters());
        }
    }

    private static OutputShape validateAdjacency(int index,
                                                 ModelConfig.LayerDescriptor descriptor,
                                                 OutputShape input) {
        List<Integer> values = descriptor.parameters();
        return switch (descriptor.type()) {
            case Models.DENSE -> {
                int inputSize = values.get(0);
                if (input.kind == OutputKind.CHW) {
                    throw incompatibleRank(index, "dense", "rank one", input);
                }
                if (input.kind == OutputKind.VECTOR && input.size != inputSize) {
                    throw incompatibleSize(index, "dense input", inputSize, input.size);
                }
                yield OutputShape.vector(values.get(1));
            }
            case Models.CONVOLUTION -> {
                int inputChannels = values.get(0);
                if (input.kind == OutputKind.VECTOR) {
                    throw incompatibleRank(index, "convolution", "CHW rank three", input);
                }
                if (input.kind == OutputKind.CHW && input.size >= 0 && input.size != inputChannels) {
                    throw incompatibleSize(index, "convolution input channels", inputChannels,
                            input.size);
                }
                yield OutputShape.chw(values.get(1));
            }
            case Models.MAX_POOL -> {
                if (input.kind == OutputKind.VECTOR) {
                    throw incompatibleRank(index, "max pool", "CHW rank three", input);
                }
                yield OutputShape.chw(input.kind == OutputKind.CHW ? input.size : -1);
            }
            case Models.FLATTEN -> {
                int inputElementCount = values.get(0);
                if (input.kind == OutputKind.VECTOR && input.size != inputElementCount) {
                    throw incompatibleSize(index, "flatten input element count",
                            inputElementCount, input.size);
                }
                if (input.kind == OutputKind.CHW && input.size >= 0
                        && inputElementCount % input.size != 0) {
                    throw new IllegalArgumentException("layer " + index
                            + " flatten input element count " + inputElementCount
                            + " is incompatible with " + input.size + " channels");
                }
                yield OutputShape.vector(inputElementCount);
            }
            case Models.RELU -> input;
            case Models.SOFTMAX -> {
                if (input.kind == OutputKind.CHW) {
                    throw incompatibleRank(index, "softmax", "rank one", input);
                }
                yield input;
            }
            default -> throw new IllegalArgumentException("unsupported layer descriptor at layer "
                    + index + ": " + descriptor.type());
        };
    }

    private static IllegalArgumentException incompatibleRank(int index, String layer,
                                                             String expected,
                                                             OutputShape actual) {
        return new IllegalArgumentException("layer " + index + " " + layer + " requires "
                + expected + " input but the previous layer produces " + actual.kind.description);
    }

    private static IllegalArgumentException incompatibleSize(int index, String description,
                                                             int expected, int actual) {
        return new IllegalArgumentException("layer " + index + " " + description
                + " is " + expected + " but the previous layer produces " + actual);
    }

    private static String describeSize(OutputShape output) {
        return output.kind == OutputKind.VECTOR && output.size >= 0
                ? Integer.toString(output.size) : "is unknown";
    }

    private enum OutputKind {
        UNKNOWN("an unknown shape"), VECTOR("rank one"), CHW("CHW rank three");

        private final String description;

        OutputKind(String description) {
            this.description = description;
        }
    }

    private record OutputShape(OutputKind kind, int size) {
        private static OutputShape unknown() {
            return new OutputShape(OutputKind.UNKNOWN, -1);
        }

        private static OutputShape vector(int width) {
            return new OutputShape(OutputKind.VECTOR, width);
        }

        private static OutputShape chw(int channels) {
            return new OutputShape(OutputKind.CHW, channels);
        }
    }
}

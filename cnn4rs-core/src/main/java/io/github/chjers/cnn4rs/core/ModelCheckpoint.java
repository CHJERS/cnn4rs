package io.github.chjers.cnn4rs.core;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Reads and writes the version 1 CNN4RS model checkpoint format. */
public final class ModelCheckpoint {
    private static final byte[] MAGIC = "CNN4RS".getBytes(StandardCharsets.US_ASCII);
    private static final int FORMAT_VERSION = 1;
    private static final int MAXIMUM_LAYER_COUNT = 10_000;

    private static final int DENSE_TAG = 1;
    private static final int CONVOLUTION_TAG = 2;
    private static final int MAX_POOL_TAG = 3;
    private static final int RELU_TAG = 4;
    private static final int SOFTMAX_TAG = 5;
    private static final int FLATTEN_TAG = 6;

    private ModelCheckpoint() {
    }

    public static void write(SequentialModel model, Path path) throws IOException {
        Objects.requireNonNull(model, "model must not be null");
        Objects.requireNonNull(path, "checkpoint path must not be null");
        validateModelStructure(model);
        try (DataOutputStream output = new DataOutputStream(new BufferedOutputStream(
                Files.newOutputStream(path)))) {
            output.write(MAGIC);
            output.writeInt(FORMAT_VERSION);
            output.writeLong(model.config().seed());
            output.writeInt(model.config().classCount());
            output.writeInt(model.config().layers().size());
            for (ModelConfig.LayerDescriptor descriptor : model.config().layers()) {
                writeDescriptor(output, descriptor);
            }
            for (int index = 0; index < model.layers().size(); index++) {
                writeParameters(output, model.layers().get(index), index);
            }
        } catch (IOException exception) {
            throw new IOException("could not write CNN4RS checkpoint '" + path + "': "
                    + exception.getMessage(), exception);
        }
    }

    public static SequentialModel read(Path path) throws IOException {
        Objects.requireNonNull(path, "checkpoint path must not be null");
        try (CheckpointInput input = new CheckpointInput(path)) {
            for (int index = 0; index < MAGIC.length; index++) {
                if (input.readByte("magic byte " + index) != MAGIC[index]) {
                    throw input.invalid("invalid checkpoint magic; expected CNN4RS");
                }
            }
            int version = input.readInt("format version");
            if (version != FORMAT_VERSION) {
                throw input.invalid("unsupported checkpoint version " + version
                        + "; supported version is " + FORMAT_VERSION);
            }
            long seed = input.readLong("model seed");
            int classCount = input.readPositiveInt("class count");
            int layerCount = input.readPositiveInt("layer count");
            if (layerCount > MAXIMUM_LAYER_COUNT) {
                throw input.invalid("layer count " + layerCount + " exceeds limit "
                        + MAXIMUM_LAYER_COUNT);
            }
            List<ModelConfig.LayerDescriptor> descriptors = new ArrayList<>(layerCount);
            for (int index = 0; index < layerCount; index++) {
                descriptors.add(readDescriptor(input, index));
            }
            ModelConfig config = new ModelConfig(descriptors, classCount, seed);
            try {
                ModelStructure.validateDescriptors(config);
            } catch (IllegalArgumentException exception) {
                throw input.invalid(exception.getMessage());
            }
            List<Layer> layers = new ArrayList<>(layerCount);
            for (int index = 0; index < descriptors.size(); index++) {
                layers.add(readLayer(input, descriptors.get(index), index));
            }
            SequentialModel model = new SequentialModel(config, layers);
            if (input.hasTrailingContent()) {
                throw input.invalid("unexpected trailing content after model parameters");
            }
            return model;
        } catch (IOException exception) {
            throw exception;
        } catch (RuntimeException exception) {
            throw new IOException("invalid CNN4RS checkpoint '" + path + "': "
                    + exception.getMessage(), exception);
        }
    }

    private static void validateModelStructure(SequentialModel model) throws IOException {
        try {
            ModelStructure.validate(model.config(), model.layers());
        } catch (IllegalArgumentException exception) {
            throw new IOException("invalid model structure for checkpoint: "
                    + exception.getMessage(), exception);
        }
    }

    private static void writeDescriptor(DataOutputStream output,
                                        ModelConfig.LayerDescriptor descriptor) throws IOException {
        List<Integer> values = descriptor.parameters();
        switch (descriptor.type()) {
            case Models.DENSE -> {
                requireParameterCount(descriptor, 2);
                output.writeByte(DENSE_TAG);
                output.writeInt(values.get(0));
                output.writeInt(values.get(1));
            }
            case Models.CONVOLUTION -> {
                requireParameterCount(descriptor, 4);
                output.writeByte(CONVOLUTION_TAG);
                for (int value : values) {
                    output.writeInt(value);
                }
            }
            case Models.MAX_POOL -> {
                requireParameterCount(descriptor, 2);
                output.writeByte(MAX_POOL_TAG);
                output.writeInt(values.get(0));
                output.writeInt(values.get(1));
            }
            case Models.FLATTEN -> {
                requireParameterCount(descriptor, 1);
                output.writeByte(FLATTEN_TAG);
                output.writeInt(values.get(0));
            }
            case Models.RELU -> {
                requireParameterCount(descriptor, 0);
                output.writeByte(RELU_TAG);
            }
            case Models.SOFTMAX -> {
                requireParameterCount(descriptor, 0);
                output.writeByte(SOFTMAX_TAG);
            }
            default -> throw new IOException("unsupported checkpoint layer type '"
                    + descriptor.type() + "'");
        }
    }

    private static void requireParameterCount(ModelConfig.LayerDescriptor descriptor, int expected)
            throws IOException {
        if (descriptor.parameters().size() != expected) {
            throw new IOException("checkpoint layer type '" + descriptor.type() + "' requires "
                    + expected + " configuration values but had " + descriptor.parameters().size());
        }
    }

    private static ModelConfig.LayerDescriptor readDescriptor(CheckpointInput input, int index)
            throws IOException {
        int tag = Byte.toUnsignedInt(input.readByte("layer " + index + " type"));
        return switch (tag) {
            case DENSE_TAG -> Models.descriptor(Models.DENSE,
                    input.readPositiveInt("layer " + index + " dense input size"),
                    input.readPositiveInt("layer " + index + " dense output size"));
            case CONVOLUTION_TAG -> Models.descriptor(Models.CONVOLUTION,
                    input.readPositiveInt("layer " + index + " convolution input channels"),
                    input.readPositiveInt("layer " + index + " convolution output channels"),
                    input.readPositiveInt("layer " + index + " convolution kernel size"),
                    input.readPositiveInt("layer " + index + " convolution stride"));
            case MAX_POOL_TAG -> Models.descriptor(Models.MAX_POOL,
                    input.readPositiveInt("layer " + index + " pool window size"),
                    input.readPositiveInt("layer " + index + " pool stride"));
            case FLATTEN_TAG -> Models.descriptor(Models.FLATTEN,
                    input.readPositiveInt("layer " + index + " flatten input element count"));
            case RELU_TAG -> Models.descriptor(Models.RELU);
            case SOFTMAX_TAG -> Models.descriptor(Models.SOFTMAX);
            default -> throw input.invalid("unknown layer type tag " + tag + " at layer " + index);
        };
    }

    private static void writeParameters(DataOutputStream output, Layer layer, int layerIndex)
            throws IOException {
        if (layer instanceof Dense dense) {
            writeFiniteValues(output, dense.weights().toArray(), layerIndex, "dense weight");
            writeFiniteValues(output, dense.biases(), layerIndex, "dense bias");
        } else if (layer instanceof Conv2D convolution) {
            writeFiniteValues(output, convolution.weights().toArray(), layerIndex,
                    "convolution weight");
            writeFiniteValues(output, convolution.biases(), layerIndex, "convolution bias");
        }
    }

    private static void writeFiniteValues(DataOutputStream output, float[] values, int layerIndex,
                                          String description) throws IOException {
        for (int index = 0; index < values.length; index++) {
            if (!Float.isFinite(values[index])) {
                throw new IOException("layer " + layerIndex + " " + description + " " + index
                        + " must be finite but was " + values[index]);
            }
            output.writeFloat(values[index]);
        }
    }

    private static Layer readLayer(CheckpointInput input, ModelConfig.LayerDescriptor descriptor,
                                   int layerIndex) throws IOException {
        List<Integer> values = descriptor.parameters();
        return switch (descriptor.type()) {
            case Models.DENSE -> {
                int inputSize = values.get(0);
                int outputSize = values.get(1);
                int weightCount = checkedProduct(input, layerIndex, "dense weight count",
                        inputSize, outputSize);
                Tensor weights = Tensor.of(input.readFiniteFloats(weightCount,
                        "layer " + layerIndex + " dense weight"), outputSize, inputSize);
                float[] biases = input.readFiniteFloats(outputSize,
                        "layer " + layerIndex + " dense bias");
                yield Dense.withParameters(inputSize, outputSize, weights, biases);
            }
            case Models.CONVOLUTION -> {
                int inputChannels = values.get(0);
                int outputChannels = values.get(1);
                int kernelSize = values.get(2);
                int stride = values.get(3);
                int weightCount = checkedProduct(input, layerIndex, "convolution weight count",
                        inputChannels, outputChannels, kernelSize, kernelSize);
                Tensor weights = Tensor.of(input.readFiniteFloats(weightCount,
                                "layer " + layerIndex + " convolution weight"),
                        outputChannels, inputChannels, kernelSize, kernelSize);
                float[] biases = input.readFiniteFloats(outputChannels,
                        "layer " + layerIndex + " convolution bias");
                yield Conv2D.withParameters(inputChannels, outputChannels, kernelSize, stride,
                        weights, biases);
            }
            case Models.MAX_POOL -> new MaxPool2D(values.get(0), values.get(1));
            case Models.FLATTEN -> new Flatten(values.get(0));
            case Models.RELU -> new ReLU();
            case Models.SOFTMAX -> new Softmax();
            default -> throw input.invalid("unsupported layer type '" + descriptor.type()
                    + "' at layer " + layerIndex);
        };
    }

    private static int checkedProduct(CheckpointInput input, int layerIndex, String description,
                                      int... factors) throws IOException {
        int product = 1;
        try {
            for (int factor : factors) {
                product = Math.multiplyExact(product, factor);
            }
            return product;
        } catch (ArithmeticException exception) {
            throw input.invalid("layer " + layerIndex + " " + description + " overflows int");
        }
    }

    private static final class CheckpointInput implements AutoCloseable {
        private final Path path;
        private final long fileSize;
        private final DataInputStream input;

        private CheckpointInput(Path path) throws IOException {
            this.path = path;
            try {
                fileSize = Files.size(path);
                input = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)));
            } catch (IOException exception) {
                throw new IOException("could not open CNN4RS checkpoint '" + path + "': "
                        + exception.getMessage(), exception);
            }
        }

        private byte readByte(String context) throws IOException {
            try {
                return input.readByte();
            } catch (EOFException exception) {
                throw truncated(context, exception);
            }
        }

        private int readInt(String context) throws IOException {
            try {
                return input.readInt();
            } catch (EOFException exception) {
                throw truncated(context, exception);
            }
        }

        private long readLong(String context) throws IOException {
            try {
                return input.readLong();
            } catch (EOFException exception) {
                throw truncated(context, exception);
            }
        }

        private int readPositiveInt(String context) throws IOException {
            int value = readInt(context);
            if (value <= 0) {
                throw invalid(context + " must be positive but was " + value);
            }
            return value;
        }

        private float[] readFiniteFloats(int count, String context) throws IOException {
            if ((long) count * Float.BYTES > fileSize) {
                throw new IOException("CNN4RS checkpoint '" + path
                        + "' is truncated: declared " + count + " values for " + context
                        + " cannot fit in " + fileSize + " bytes");
            }
            float[] values = new float[count];
            for (int index = 0; index < count; index++) {
                try {
                    values[index] = input.readFloat();
                } catch (EOFException exception) {
                    throw truncated(context + " " + index, exception);
                }
                if (!Float.isFinite(values[index])) {
                    throw invalid(context + " " + index + " must be finite but was "
                            + values[index]);
                }
            }
            return values;
        }

        private boolean hasTrailingContent() throws IOException {
            return input.read() != -1;
        }

        private IOException truncated(String context, EOFException cause) {
            return new IOException("CNN4RS checkpoint '" + path + "' is truncated while reading "
                    + context, cause);
        }

        private IOException invalid(String message) {
            return new IOException("invalid CNN4RS checkpoint '" + path + "': " + message);
        }

        @Override
        public void close() throws IOException {
            input.close();
        }
    }
}

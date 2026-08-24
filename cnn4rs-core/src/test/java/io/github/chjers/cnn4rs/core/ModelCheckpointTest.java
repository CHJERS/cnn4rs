package io.github.chjers.cnn4rs.core;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelCheckpointTest {
    @Test
    void checkpointRoundTripPreservesPredictions(@TempDir Path directory) throws IOException {
        SequentialModel model = Models.denseClassifier(2, 2, 7L);
        Tensor input = Tensor.of(new float[] {1, 0}, 2);
        float[] initializedPrediction = model.predict(input).toArray();
        Trainer trainer = new Trainer(model, 0.1f);
        List<TrainingExample> samples = List.of(
                new TrainingExample(input, 0),
                new TrainingExample(Tensor.of(new float[] {0, 1}, 2), 1));
        for (int iteration = 0; iteration < 20; iteration++) {
            trainer.trainBatch(samples);
        }
        Path file = directory.resolve("model.cnn4rs");

        ModelCheckpoint.write(model, file);

        assertFalse(Arrays.equals(initializedPrediction, model.predict(input).toArray()));
        assertArrayEquals(model.predict(input).toArray(),
                ModelCheckpoint.read(file).predict(input).toArray());
    }

    @Test
    void checkpointStartsWithMagicAndVersionOne(@TempDir Path directory) throws IOException {
        byte[] bytes = Files.readAllBytes(writeValidCheckpoint(directory));

        assertArrayEquals("CNN4RS".getBytes(StandardCharsets.US_ASCII),
                Arrays.copyOf(bytes, 6));
        assertEquals(1, ByteBuffer.wrap(bytes, 6, Integer.BYTES)
                .order(ByteOrder.BIG_ENDIAN).getInt());
    }

    @Test
    void checkpointRejectsBadMagic(@TempDir Path directory) throws IOException {
        Path valid = writeValidCheckpoint(directory);
        byte[] bytes = Files.readAllBytes(valid);

        byte[] badMagic = bytes.clone();
        badMagic[0] = 'X';
        Path badMagicFile = directory.resolve("bad-magic.cnn4rs");
        Files.write(badMagicFile, badMagic);
        IOException magicFailure = assertThrows(IOException.class,
                () -> ModelCheckpoint.read(badMagicFile));

        assertTrue(magicFailure.getMessage().contains("magic"));
    }

    @Test
    void checkpointRejectsUnsupportedVersion(@TempDir Path directory) throws IOException {
        byte[] bytes = Files.readAllBytes(writeValidCheckpoint(directory));

        byte[] badVersion = bytes.clone();
        ByteBuffer.wrap(badVersion, 6, Integer.BYTES).order(ByteOrder.BIG_ENDIAN).putInt(2);
        Path badVersionFile = directory.resolve("bad-version.cnn4rs");
        Files.write(badVersionFile, badVersion);
        IOException versionFailure = assertThrows(IOException.class,
                () -> ModelCheckpoint.read(badVersionFile));

        assertTrue(versionFailure.getMessage().contains("version"));
    }

    @Test
    void checkpointRejectsTruncatedContent(@TempDir Path directory) throws IOException {
        Path valid = writeValidCheckpoint(directory);
        byte[] bytes = Files.readAllBytes(valid);

        Path truncated = directory.resolve("truncated.cnn4rs");
        Files.write(truncated, Arrays.copyOf(bytes, bytes.length - 1));
        IOException truncatedFailure = assertThrows(IOException.class,
                () -> ModelCheckpoint.read(truncated));

        assertTrue(truncatedFailure.getMessage().contains("truncated"));
    }

    @Test
    void checkpointRejectsNonFiniteParameters(@TempDir Path directory) throws IOException {
        byte[] bytes = Files.readAllBytes(writeValidCheckpoint(directory));

        byte[] nonFinite = bytes.clone();
        ByteBuffer.wrap(nonFinite, nonFinite.length - Float.BYTES, Float.BYTES)
                .order(ByteOrder.BIG_ENDIAN).putFloat(Float.NaN);
        Path nonFiniteFile = directory.resolve("non-finite.cnn4rs");
        Files.write(nonFiniteFile, nonFinite);
        IOException parameterFailure = assertThrows(IOException.class,
                () -> ModelCheckpoint.read(nonFiniteFile));

        assertTrue(parameterFailure.getMessage().contains("finite"));
    }

    @Test
    void checkpointRejectsClassCountDifferentFromFinalWidth(@TempDir Path directory)
            throws IOException {
        byte[] bytes = Files.readAllBytes(writeValidCheckpoint(directory));
        int classCountOffset = 6 + Integer.BYTES + Long.BYTES;
        ByteBuffer.wrap(bytes, classCountOffset, Integer.BYTES)
                .order(ByteOrder.BIG_ENDIAN).putInt(3);
        Path file = directory.resolve("bad-class-count.cnn4rs");
        Files.write(file, bytes);

        IOException failure = assertThrows(IOException.class, () -> ModelCheckpoint.read(file));

        assertTrue(failure.getMessage().contains("class count"));
    }

    @Test
    void checkpointRejectsAdjacentDenseWidthMismatch(@TempDir Path directory) throws IOException {
        ModelConfig config = new ModelConfig(List.of(
                Models.descriptor(Models.DENSE, 2, 3),
                Models.descriptor(Models.DENSE, 3, 2),
                Models.descriptor(Models.SOFTMAX)), 2, 7L);
        SequentialModel model = new SequentialModel(config, List.of(
                new Dense(2, 3, new java.util.Random(7L)),
                new Dense(3, 2, new java.util.Random(8L)),
                new Softmax()));
        Path valid = directory.resolve("two-dense.cnn4rs");
        ModelCheckpoint.write(model, valid);
        byte[] bytes = Files.readAllBytes(valid);
        int descriptorsOffset = 6 + Integer.BYTES + Long.BYTES
                + Integer.BYTES + Integer.BYTES;
        int secondDenseInputOffset = descriptorsOffset
                + 1 + Integer.BYTES + Integer.BYTES + 1;
        ByteBuffer.wrap(bytes, secondDenseInputOffset, Integer.BYTES)
                .order(ByteOrder.BIG_ENDIAN).putInt(2);
        Path corrupted = directory.resolve("bad-adjacency.cnn4rs");
        Files.write(corrupted, bytes);

        IOException failure = assertThrows(IOException.class,
                () -> ModelCheckpoint.read(corrupted));

        assertTrue(failure.getMessage().contains("layer 1"));
    }

    @Test
    void convolutionalCheckpointRoundTrips(
            @TempDir Path directory) throws IOException {
        Conv2D convolution = Conv2D.withParameters(1, 1, 1, 1,
                Tensor.of(new float[] {0.75f}, 1, 1, 1, 1), new float[] {0.25f});
        Dense dense = Dense.withParameters(2, 2,
                Tensor.of(new float[] {2, -1, -3, 4}, 2, 2), new float[] {0.5f, -0.5f});
        ModelConfig config = new ModelConfig(List.of(
                Models.descriptor(Models.CONVOLUTION, 1, 1, 1, 1),
                Models.descriptor(Models.FLATTEN, 2),
                Models.descriptor(Models.DENSE, 2, 2),
                Models.descriptor(Models.SOFTMAX)), 2, 99L);
        SequentialModel model = new SequentialModel(config, List.of(
                convolution, new Flatten(2), dense, new Softmax()));
        Tensor input = Tensor.of(new float[] {1, -2}, 1, 1, 2);
        Path file = directory.resolve("convolutional.cnn4rs");

        ModelCheckpoint.write(model, file);

        assertArrayEquals(model.predict(input).toArray(),
                ModelCheckpoint.read(file).predict(input).toArray());
    }

    private static Path writeValidCheckpoint(Path directory) throws IOException {
        Path file = directory.resolve("valid.cnn4rs");
        ModelCheckpoint.write(Models.denseClassifier(2, 2, 7L), file);
        return file;
    }
}

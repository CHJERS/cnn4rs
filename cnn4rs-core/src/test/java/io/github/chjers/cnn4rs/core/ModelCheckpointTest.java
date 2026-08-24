package io.github.chjers.cnn4rs.core;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelCheckpointTest {
    @Test
    void checkpointRoundTripPreservesPredictions(@TempDir Path directory) throws IOException {
        SequentialModel model = Models.denseClassifier(2, 2, 7L);
        Tensor input = Tensor.of(new float[] {1, 0}, 2);
        Path file = directory.resolve("model.cnn4rs");

        ModelCheckpoint.write(model, file);

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

    private static Path writeValidCheckpoint(Path directory) throws IOException {
        Path file = directory.resolve("valid.cnn4rs");
        ModelCheckpoint.write(Models.denseClassifier(2, 2, 7L), file);
        return file;
    }
}

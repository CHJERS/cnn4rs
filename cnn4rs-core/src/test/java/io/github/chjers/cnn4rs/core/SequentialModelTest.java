package io.github.chjers.cnn4rs.core;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SequentialModelTest {
    @Test
    void modelRejectsDescriptorThatDoesNotMatchItsLayer() {
        ModelConfig config = new ModelConfig(List.of(
                Models.descriptor(Models.DENSE, 2, 2),
                Models.descriptor(Models.SOFTMAX)), 2, 7L);

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> new SequentialModel(config, List.of(
                        new Dense(3, 2, new Random(7L)), new Softmax())));

        assertTrue(failure.getMessage().contains("descriptor"));
    }

    @Test
    void modelRejectsAdjacentDenseWidthMismatch() {
        ModelConfig config = new ModelConfig(List.of(
                Models.descriptor(Models.DENSE, 2, 3),
                Models.descriptor(Models.DENSE, 2, 2),
                Models.descriptor(Models.SOFTMAX)), 2, 7L);

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> new SequentialModel(config, List.of(
                        new Dense(2, 3, new Random(7L)),
                        new Dense(2, 2, new Random(8L)),
                        new Softmax())));

        assertTrue(failure.getMessage().contains("layer 1"));
    }

    @Test
    void modelRejectsFinalWidthDifferentFromClassCount() {
        ModelConfig config = new ModelConfig(List.of(
                Models.descriptor(Models.DENSE, 2, 3),
                Models.descriptor(Models.SOFTMAX)), 2, 7L);

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> new SequentialModel(config, List.of(
                        new Dense(2, 3, new Random(7L)), new Softmax())));

        assertTrue(failure.getMessage().contains("class count"));
    }
}

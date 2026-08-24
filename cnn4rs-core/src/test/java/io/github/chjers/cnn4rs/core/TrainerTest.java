package io.github.chjers.cnn4rs.core;

import java.util.List;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TrainerTest {
    @Test
    void oneHotTrainingReducesLossOnSimpleExamples() {
        SequentialModel model = Models.denseClassifier(2, 2, 7L);
        Trainer trainer = new Trainer(model, 0.1f);
        List<TrainingExample> samples = List.of(
                new TrainingExample(Tensor.of(new float[] {1, 0}, 2), 0),
                new TrainingExample(Tensor.of(new float[] {0, 1}, 2), 1));

        float before = trainer.loss(samples);
        TrainingResult last = trainer.trainBatch(samples);
        for (int iteration = 1; iteration < 80; iteration++) {
            last = trainer.trainBatch(samples);
        }

        assertTrue(trainer.loss(samples) < before);
        assertTrue(last.loss() >= 0f);
    }

    @Test
    void matchingSeedsProduceMatchingPredictions() {
        Tensor input = Tensor.of(new float[] {0.25f, -0.5f}, 2);

        Tensor first = Models.denseClassifier(2, 3, 29L).predict(input);
        Tensor second = Models.denseClassifier(2, 3, 29L).predict(input);

        assertArrayEquals(first.toArray(), second.toArray());
    }

    @Test
    void trainerRejectsEmptyBatchesAndOutOfRangeLabels() {
        Trainer trainer = new Trainer(Models.denseClassifier(2, 2, 7L), 0.1f);

        assertThrows(IllegalArgumentException.class, () -> trainer.trainBatch(List.of()));
        assertThrows(IllegalArgumentException.class, () -> trainer.loss(List.of(
                new TrainingExample(Tensor.of(new float[] {1, 0}, 2), 2))));
    }
}

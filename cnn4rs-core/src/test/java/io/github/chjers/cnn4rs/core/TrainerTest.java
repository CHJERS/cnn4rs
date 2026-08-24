package io.github.chjers.cnn4rs.core;

import java.util.List;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
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

    @Test
    void convolutionFlattenDenseClassifierTrainsEndToEnd() {
        Conv2D convolution = Conv2D.withParameters(1, 1, 1, 1,
                Tensor.of(new float[] {1}, 1, 1, 1, 1), new float[] {0});
        Dense dense = Dense.withParameters(2, 2,
                Tensor.of(new float[] {1, 0, 0, 1}, 2, 2), new float[] {0, 0});
        ModelConfig config = new ModelConfig(List.of(
                Models.descriptor(Models.CONVOLUTION, 1, 1, 1, 1),
                Models.descriptor(Models.FLATTEN, 2),
                Models.descriptor(Models.DENSE, 2, 2),
                Models.descriptor(Models.SOFTMAX)), 2, 7L);
        SequentialModel model = new SequentialModel(config, List.of(
                convolution, new Flatten(2), dense, new Softmax()));
        Trainer trainer = new Trainer(model, 0.1f);
        List<TrainingExample> samples = List.of(
                new TrainingExample(Tensor.of(new float[] {1, 0}, 1, 1, 2), 0),
                new TrainingExample(Tensor.of(new float[] {0, 1}, 1, 1, 2), 1));
        float beforeLoss = trainer.loss(samples);
        float beforeWeight = convolution.weights().get(0, 0, 0, 0);

        for (int iteration = 0; iteration < 20; iteration++) {
            trainer.trainBatch(samples);
        }

        assertTrue(trainer.loss(samples) < beforeLoss);
        assertNotEquals(beforeWeight, convolution.weights().get(0, 0, 0, 0));
    }
}

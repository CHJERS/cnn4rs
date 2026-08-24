package io.github.chjers.cnn4rs.distributed;

import io.github.chjers.cnn4rs.core.Tensor;
import io.github.chjers.cnn4rs.core.TrainingExample;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GradientAggregatorTest {
    @Test
    void averagesGradientsByWorkerSampleCount() {
        GradientResult one = result("v1", 1, 1f);
        GradientResult three = result("v1", 3, 5f);

        GradientResult average = new GradientAggregator().weightedAverage(List.of(one, three));

        assertEquals(4f, average.gradients().get(0).get(0));
        assertEquals(4, average.sampleCount());
        assertEquals("v1", average.modelVersion());
    }

    @Test
    void rejectsDifferentModelVersions() {
        assertThrows(GradientAggregationException.class,
                () -> new GradientAggregator().weightedAverage(List.of(
                        result("v1", 1, 1f), result("v2", 1, 1f))));
    }

    @Test
    void rejectsGradientsWithDifferentShapes() {
        GradientResult scalar = result("v1", 1, 1f);
        GradientResult vector = new GradientResult("v1", 1, List.of(Tensor.of(new float[] {1f, 2f}, 2)));

        assertThrows(GradientAggregationException.class,
                () -> new GradientAggregator().weightedAverage(List.of(scalar, vector)));
    }

    @Test
    void rejectsDifferentGradientTensorCounts() {
        GradientResult oneTensor = result("v1", 1, 1f);
        GradientResult twoTensors = new GradientResult("v1", 1,
                List.of(Tensor.of(new float[] {1f}, 1), Tensor.of(new float[] {2f}, 1)));

        assertThrows(GradientAggregationException.class,
                () -> new GradientAggregator().weightedAverage(List.of(oneTensor, twoTensors)));
    }

    @Test
    void taskAndResultDefensivelyCopyAndValidateTheirContents() {
        List<TrainingExample> examples = new ArrayList<>();
        examples.add(new TrainingExample(Tensor.of(new float[] {1f}, 1), 0));
        TrainingTask task = new TrainingTask("v1", examples);
        examples.clear();

        List<Tensor> gradients = new ArrayList<>();
        gradients.add(Tensor.of(new float[] {1f}, 1));
        GradientResult result = new GradientResult("v1", 1, gradients);
        gradients.clear();

        assertEquals(1, task.examples().size());
        assertEquals(1, result.gradients().size());
        assertThrows(IllegalArgumentException.class, () -> new TrainingTask(" ", task.examples()));
        assertThrows(IllegalArgumentException.class, () -> new TrainingTask("v1", List.of()));
        assertThrows(IllegalArgumentException.class, () -> new GradientResult(" ", 1, result.gradients()));
        assertThrows(IllegalArgumentException.class, () -> new GradientResult("v1", 0, result.gradients()));
        assertThrows(IllegalArgumentException.class, () -> new GradientResult("v1", 1, List.of()));
        assertThrows(IllegalArgumentException.class,
                () -> new GradientResult("v1", 1, List.of(Tensor.of(new float[] {Float.NaN}, 1))));
    }

    private static GradientResult result(String version, int sampleCount, float value) {
        return new GradientResult(version, sampleCount, List.of(Tensor.of(new float[] {value}, 1)));
    }
}

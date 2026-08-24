package io.github.chjers.cnn4rs.distributed;

import io.github.chjers.cnn4rs.core.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Performs local, sample-count weighted aggregation of compatible worker gradients.
 * This experimental class excludes networking, scheduling, retries, authentication,
 * and persistence.
 */
public final class GradientAggregator {
    public GradientResult weightedAverage(List<GradientResult> results) {
        Objects.requireNonNull(results, "gradient results must not be null");
        if (results.isEmpty()) {
            throw new GradientAggregationException("gradient results must not be empty");
        }

        GradientResult reference = requireResult(results.get(0), 0);
        int gradientCount = reference.gradients().size();
        List<double[]> sums = new ArrayList<>(gradientCount);
        for (Tensor gradient : reference.gradients()) {
            sums.add(new double[gradient.toArray().length]);
        }

        int totalSamples = 0;
        for (int resultIndex = 0; resultIndex < results.size(); resultIndex++) {
            GradientResult result = requireResult(results.get(resultIndex), resultIndex);
            validateCompatible(reference, result, resultIndex);
            try {
                totalSamples = Math.addExact(totalSamples, result.sampleCount());
            } catch (ArithmeticException exception) {
                throw new GradientAggregationException("total sample count exceeds integer range");
            }
            addWeighted(sums, result);
        }

        return new GradientResult(reference.modelVersion(), totalSamples,
                averages(reference.gradients(), sums, totalSamples));
    }

    private static GradientResult requireResult(GradientResult result, int index) {
        return Objects.requireNonNull(result, "gradient result " + index + " must not be null");
    }

    private static void validateCompatible(GradientResult reference, GradientResult candidate,
                                           int resultIndex) {
        if (!reference.modelVersion().equals(candidate.modelVersion())) {
            throw new GradientAggregationException("gradient result " + resultIndex
                    + " has model version " + candidate.modelVersion() + " instead of "
                    + reference.modelVersion());
        }
        if (reference.gradients().size() != candidate.gradients().size()) {
            throw new GradientAggregationException("gradient result " + resultIndex
                    + " has a different gradient tensor count");
        }
        for (int gradientIndex = 0; gradientIndex < reference.gradients().size(); gradientIndex++) {
            if (!reference.gradients().get(gradientIndex).shape()
                    .equals(candidate.gradients().get(gradientIndex).shape())) {
                throw new GradientAggregationException("gradient result " + resultIndex
                        + " has a different shape for gradient " + gradientIndex);
            }
        }
    }

    private static void addWeighted(List<double[]> sums, GradientResult result) {
        for (int gradientIndex = 0; gradientIndex < result.gradients().size(); gradientIndex++) {
            float[] values = result.gradients().get(gradientIndex).toArray();
            double[] sum = sums.get(gradientIndex);
            for (int valueIndex = 0; valueIndex < values.length; valueIndex++) {
                sum[valueIndex] += (double) values[valueIndex] * result.sampleCount();
            }
        }
    }

    private static List<Tensor> averages(List<Tensor> referenceGradients, List<double[]> sums,
                                         int totalSamples) {
        ArrayList<Tensor> averages = new ArrayList<>(referenceGradients.size());
        for (int gradientIndex = 0; gradientIndex < referenceGradients.size(); gradientIndex++) {
            double[] sum = sums.get(gradientIndex);
            float[] values = new float[sum.length];
            for (int valueIndex = 0; valueIndex < values.length; valueIndex++) {
                values[valueIndex] = (float) (sum[valueIndex] / totalSamples);
                if (!Float.isFinite(values[valueIndex])) {
                    throw new GradientAggregationException("weighted average is non-finite for gradient "
                            + gradientIndex + " at index " + valueIndex);
                }
            }
            averages.add(Tensor.of(values, referenceGradients.get(gradientIndex).shape().dimensions()));
        }
        return List.copyOf(averages);
    }
}

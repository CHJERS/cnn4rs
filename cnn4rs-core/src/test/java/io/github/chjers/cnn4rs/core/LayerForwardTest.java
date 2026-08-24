package io.github.chjers.cnn4rs.core;

import java.util.Random;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LayerForwardTest {
    @Test
    void convolutionCalculatesOneValidFeatureMap() {
        Conv2D layer = Conv2D.withParameters(1, 1, 2, 1,
                Tensor.of(new float[] {1, 0, 0, 1}, 1, 1, 2, 2), new float[] {0});

        Tensor output = layer.forward(Tensor.of(new float[] {1, 2, 3, 4}, 1, 2, 2));

        assertEquals(5f, output.get(0, 0, 0));
        assertArrayEquals(new int[] {1, 1, 1}, output.shape().dimensions());
    }

    @Test
    void convolutionAppliesAllChannelsBiasAndStride() {
        Conv2D layer = Conv2D.withParameters(2, 1, 2, 2,
                Tensor.of(new float[] {
                        1, 1, 1, 1,
                        2, 2, 2, 2
                }, 1, 2, 2, 2), new float[] {3});

        Tensor output = layer.forward(Tensor.of(new float[] {
                1, 1, 1, 1,
                2, 2, 2, 2
        }, 2, 2, 2));

        assertEquals(23f, output.get(0, 0, 0));
    }

    @Test
    void convolutionRandomConstructorIsDeterministicForMatchingRandoms() {
        Conv2D first = new Conv2D(1, 1, 2, 1, new Random(7));
        Conv2D second = new Conv2D(1, 1, 2, 1, new Random(7));
        Tensor input = Tensor.of(new float[] {1, 2, 3, 4}, 1, 2, 2);

        assertArrayEquals(first.forward(input).toArray(), second.forward(input).toArray());
    }

    @Test
    void maxPoolUsesChwWindows() {
        MaxPool2D pool = new MaxPool2D(2, 2);

        Tensor output = pool.forward(Tensor.of(new float[] {
                1, 4, 2, 3,
                9, 8, 7, 6
        }, 1, 2, 4));

        assertArrayEquals(new int[] {1, 1, 2}, output.shape().dimensions());
        assertArrayEquals(new float[] {9, 7}, output.toArray());
    }

    @Test
    void maxPoolRejectsIncompleteWindows() {
        MaxPool2D pool = new MaxPool2D(2, 2);

        assertThrows(InvalidTensorException.class,
                () -> pool.forward(Tensor.zeros(1, 3, 4)));
    }

    @Test
    void denseAcceptsRankOneInputAndAddsBias() {
        Dense dense = Dense.withParameters(2, 2,
                Tensor.of(new float[] {1, 2, 3, 4}, 2, 2), new float[] {1, -1});

        Tensor output = dense.forward(Tensor.of(new float[] {2, 3}, 2));

        assertArrayEquals(new float[] {9, 17}, output.toArray());
        assertThrows(InvalidTensorException.class, () -> dense.forward(Tensor.zeros(1, 2)));
    }

    @Test
    void reluClampsNegativeValues() {
        Tensor output = new ReLU().forward(Tensor.of(new float[] {-2, 0, 3}, 3));

        assertArrayEquals(new float[] {0, 0, 3}, output.toArray());
    }

    @Test
    void softmaxIsNormalizedAndStable() {
        Tensor output = new Softmax().forward(Tensor.of(new float[] {1000, 1001}, 2));

        assertEquals(1f, output.get(0) + output.get(1), 0.00001f);
        assertTrue(output.get(1) > output.get(0));
    }
}

package io.github.chjers.cnn4rs.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorTest {
    @Test
    void readsChannelRowColumnValues() {
        Tensor tensor = Tensor.of(new float[] {1, 2, 3, 4}, 1, 2, 2);

        assertEquals(4f, tensor.get(0, 1, 1));
        assertArrayEquals(new int[] {1, 2, 2}, tensor.shape().dimensions());
    }

    @Test
    void copiesInputAndOutputValues() {
        float[] source = {1, 2};
        Tensor tensor = Tensor.of(source, 2);
        source[0] = 99;
        float[] values = tensor.toArray();
        values[1] = 88;

        assertEquals(1f, tensor.get(0));
        assertEquals(2f, tensor.get(1));
    }

    @Test
    void withCreatesAnUpdatedTensorWithoutChangingOriginal() {
        Tensor original = Tensor.of(new float[] {1, 2}, 2);
        Tensor changed = original.with(new int[] {1}, 7);

        assertNotSame(original, changed);
        assertEquals(2f, original.get(1));
        assertEquals(7f, changed.get(1));
    }

    @Test
    void rejectsInvalidShapesValuesAndIndicesWithTensorErrors() {
        assertThrows(InvalidTensorException.class, () -> Shape.of(2, 0));
        assertThrows(InvalidTensorException.class, () -> Shape.of(Integer.MAX_VALUE, 2));
        assertThrows(InvalidTensorException.class, () -> Tensor.of(new float[] {1}, 2));
        assertThrows(InvalidTensorException.class, () -> Tensor.zeros(2).get(2));
        assertThrows(InvalidTensorException.class, () -> Tensor.zeros(2).get(0, 0));
        assertThrows(InvalidTensorException.class, () -> Shape.of(2).dimension(1));
    }
}

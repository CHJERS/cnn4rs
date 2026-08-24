# cnn4rs Library Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Convert the 2020 CNN research project into a documented, tested Java 17 multi-module library while retaining the original implementation and thesis.

**Architecture:** A pure-JDK core owns typed tensors, layers, models, training, and checkpoints. Optional GDAL access and experimental distributed gradient aggregation are separate modules; the 2020 source moves unchanged to legacy.

**Tech Stack:** Java 17, Maven reactor, JUnit Jupiter 5, Maven Surefire, GitHub Actions, runtime reflection for optional GDAL.

## Global Constraints

- License: MIT.
- Java baseline: JDK 17.
- New public package root: io.github.chjers.cnn4rs.
- The core module has only JDK dependencies.
- GDAL must be optional at runtime. Tests and the core build require no GDAL native library.
- Original source remains under legacy and outside the Maven reactor; do not alter its semantics.
- The distributed module is labelled experimental and makes no production claims.
- New library code throws contextual exceptions and never calls System.exit.
- The thesis PDF stays unchanged in docs and the README repeats its rights notice.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| pom.xml | Parent Maven reactor, compiler and test settings. |
| cnn4rs-core | Tensor values, layers, model, trainer, checkpoint. |
| cnn4rs-geospatial | Raster contracts and runtime-only GDAL adapter. |
| cnn4rs-distributed | Immutable task/result records and aggregation. |
| examples | In-memory classifier and opt-in GDAL tutorial. |
| legacy/src/main/java | Exact relocated 2020 code. |
| docs and .github | Documentation, provenance, CI and contributor templates. |

### Task 1: Create the Maven shell and preserve the legacy implementation

**Files:**

- Create: .gitignore, LICENSE, CHANGELOG.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md.
- Create: cnn4rs-core/pom.xml and cnn4rs-core/src/main/java/io/github/chjers/cnn4rs/core/Tensor.java.
- Create: cnn4rs-core/src/test/java/io/github/chjers/cnn4rs/core/BuildContractTest.java.
- Create: cnn4rs-geospatial/pom.xml, cnn4rs-distributed/pom.xml, examples/pom.xml.
- Modify: pom.xml and README.md.
- Move: src/main/java to legacy/src/main/java; cnn4rs.iml and .idea to legacy.

**Interfaces:**

- Consumes: original POM, README and source tree.
- Produces: reactor modules io.github.chjers:cnn4rs-core, cnn4rs-geospatial, cnn4rs-distributed and cnn4rs-examples at a shared version.

- [ ] **Step 1: Write the failing build-contract test**

~~~java
package io.github.chjers.cnn4rs.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class BuildContractTest {
    @Test void exposesJava17LibraryCoordinates() {
        assertEquals("io.github.chjers.cnn4rs.core", Tensor.class.getPackageName());
    }
}
~~~

- [ ] **Step 2: Run the test to verify it fails**

Run: mvn -pl cnn4rs-core -Dtest=BuildContractTest test

Expected: FAIL because neither the module nor Tensor exists.

- [ ] **Step 3: Build the minimal shell**

Use git mv for existing source and IDE files. Set the root POM packaging to pom; list all four modules; set maven.compiler.release to 17, JUnit Jupiter to 5.11.0 and Surefire to 3.5.2. Create:

~~~java
package io.github.chjers.cnn4rs.core;

/** Multidimensional numeric data used by cnn4rs layers. */
public final class Tensor { }
~~~

Write the MIT license, ignore target, .idea, iml/class files and local GDAL configuration, and state that legacy is reference-only and excluded from the Maven build.

- [ ] **Step 4: Verify the shell**

Run: mvn -pl cnn4rs-core -Dtest=BuildContractTest test && mvn test

Expected: both commands exit 0 and do not load GDAL.

- [ ] **Step 5: Commit**

~~~bash
git add pom.xml legacy .gitignore LICENSE CHANGELOG.md CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md README.md cnn4rs-core cnn4rs-geospatial cnn4rs-distributed examples
git commit -m "build: create modular cnn4rs project shell"
~~~

### Task 2: Implement an immutable tensor and deterministic forward layers

**Files:**

- Replace: cnn4rs-core/src/main/java/io/github/chjers/cnn4rs/core/Tensor.java.
- Create: Shape.java, InvalidTensorException.java, Layer.java, TrainableLayer.java, Conv2D.java, MaxPool2D.java, Dense.java, ReLU.java and Softmax.java in the same package.
- Test: TensorTest.java and LayerForwardTest.java in cnn4rs-core/src/test/java/io/github/chjers/cnn4rs/core.

**Interfaces:**

- Consumes: the core build from Task 1.
- Produces: Tensor.of(float[] values, int... shape), Tensor.zeros(int... shape), Shape shape(), float get(int... indices), Tensor with(int[] indices, float value), float[] toArray(), and Tensor forward(Tensor input).

- [ ] **Step 1: Write failing tensor and layer tests**

~~~java
@Test void readsChannelRowColumnValues() {
    Tensor tensor = Tensor.of(new float[] {1, 2, 3, 4}, 1, 2, 2);
    assertEquals(4f, tensor.get(0, 1, 1));
    assertArrayEquals(new int[] {1, 2, 2}, tensor.shape().dimensions());
}

@Test void convolutionCalculatesOneValidFeatureMap() {
    Conv2D layer = Conv2D.withParameters(1, 1, 2, 1,
        Tensor.of(new float[] {1, 0, 0, 1}, 1, 1, 2, 2), new float[] {0});
    Tensor output = layer.forward(Tensor.of(new float[] {1, 2, 3, 4}, 1, 2, 2));
    assertEquals(5f, output.get(0, 0, 0));
}

@Test void softmaxIsNormalizedAndStable() {
    Tensor output = new Softmax().forward(Tensor.of(new float[] {1000, 1001}, 2));
    assertEquals(1f, output.get(0) + output.get(1), 0.00001f);
    assertTrue(output.get(1) > output.get(0));
}
~~~

- [ ] **Step 2: Run tests to verify they fail**

Run: mvn -pl cnn4rs-core -Dtest=TensorTest,LayerForwardTest test

Expected: FAIL because Tensor APIs and layer classes are absent.

- [ ] **Step 3: Implement only these contracts**

Store defensive copies and calculate row-major offsets in one private method. Shape rejects zero/negative dimensions, overflow and bad axes. Tensor rejects wrong element counts and bad indices with context; with returns a new value. Use CHW tensors for convolution/pooling. Conv2D supports valid convolution and positive stride, MaxPool2D rejects incomplete windows, Dense accepts rank 1, ReLU computes max(0,x), and Softmax subtracts the maximum logit. Random constructors accept java.util.Random; parameter factories copy arguments.

- [ ] **Step 4: Verify**

Run: mvn -pl cnn4rs-core test

Expected: shape, immutability, convolution, pooling, Dense, ReLU and stable Softmax assertions pass.

- [ ] **Step 5: Commit**

~~~bash
git add cnn4rs-core/src/main cnn4rs-core/src/test
git commit -m "feat(core): add validated tensor and CNN forward layers"
~~~

### Task 3: Add model training and a versioned checkpoint format

**Files:**

- Create: ModelConfig.java, SequentialModel.java, TrainingExample.java, CrossEntropyLoss.java, Trainer.java, TrainingResult.java, ModelCheckpoint.java and Models.java in cnn4rs-core.
- Test: TrainerTest.java and ModelCheckpointTest.java in the core test package.

**Interfaces:**

- Consumes: Tensor, Layer and TrainableLayer from Task 2.
- Produces: SequentialModel.predict(Tensor), Trainer.trainBatch(List<TrainingExample>), TrainingResult.loss(), Models.denseClassifier(int,int,long), ModelCheckpoint.write(SequentialModel,Path), and ModelCheckpoint.read(Path).

- [ ] **Step 1: Write failing trainer/checkpoint tests**

~~~java
@Test void oneHotTrainingReducesLossOnSimpleExamples() {
    SequentialModel model = Models.denseClassifier(2, 2, 7L);
    Trainer trainer = new Trainer(model, 0.1f);
    List<TrainingExample> samples = List.of(
        new TrainingExample(Tensor.of(new float[] {1, 0}, 2), 0),
        new TrainingExample(Tensor.of(new float[] {0, 1}, 2), 1));
    float before = trainer.loss(samples);
    for (int i = 0; i < 80; i++) trainer.trainBatch(samples);
    assertTrue(trainer.loss(samples) < before);
}

@Test void checkpointRoundTripPreservesPredictions(@TempDir Path directory) throws IOException {
    SequentialModel model = Models.denseClassifier(2, 2, 7L);
    Tensor input = Tensor.of(new float[] {1, 0}, 2);
    Path file = directory.resolve("model.cnn4rs");
    ModelCheckpoint.write(model, file);
    assertArrayEquals(model.predict(input).toArray(),
        ModelCheckpoint.read(file).predict(input).toArray());
}
~~~

- [ ] **Step 2: Run tests to verify they fail**

Run: mvn -pl cnn4rs-core -Dtest=TrainerTest,ModelCheckpointTest test

Expected: FAIL because the model, trainer and checkpoint types are absent.

- [ ] **Step 3: Implement minimal sequential learning**

Cache forward values only for a training call. Backpropagate Softmax-plus-cross-entropy, accumulate Dense and Conv2D gradients and apply one learning-rate update per batch. ModelConfig stores layer descriptors and seed. Checkpoint bytes begin with magic CNN4RS and format version 1, then configuration and parameters; reject bad magic/version, truncated content and non-finite parameters using contextual IOException.

- [ ] **Step 4: Verify**

Run: mvn -pl cnn4rs-core test

Expected: the fixed loop lowers loss and checkpoint predictions match exactly.

- [ ] **Step 5: Commit**

~~~bash
git add cnn4rs-core/src/main cnn4rs-core/src/test
git commit -m "feat(core): add sequential training and checkpoints"
~~~

### Task 4: Add experimental aggregation and optional GDAL APIs

**Files:**

- Create: TrainingTask.java, GradientResult.java, GradientAggregator.java and GradientAggregationException.java in cnn4rs-distributed.
- Test: GradientAggregatorTest.java in cnn4rs-distributed/src/test/java/io/github/chjers/cnn4rs/distributed.
- Create: RasterReader.java, RasterWriter.java, RasterTile.java, GdalRasterReader.java and GdalUnavailableException.java in cnn4rs-geospatial.
- Test: GdalAvailabilityTest.java in cnn4rs-geospatial/src/test/java/io/github/chjers/cnn4rs/geospatial.

**Interfaces:**

- Consumes: Tensor and TrainingExample from core.
- Produces: TrainingTask(String,List<TrainingExample>), GradientResult(String,int,List<Tensor>), GradientResult GradientAggregator.weightedAverage(List<GradientResult>), RasterTile(Tensor,int,int), and RasterReader.read(Path).

- [ ] **Step 1: Write failing aggregation and GDAL absence tests**

~~~java
@Test void averagesGradientsByWorkerSampleCount() {
    GradientResult one = result("v1", 1, 1f);
    GradientResult three = result("v1", 3, 5f);
    GradientResult average = new GradientAggregator().weightedAverage(List.of(one, three));
    assertEquals(4f, average.gradients().get(0).get(0));
    assertEquals(4, average.sampleCount());
}

@Test void rejectsDifferentModelVersions() {
    assertThrows(GradientAggregationException.class,
        () -> new GradientAggregator().weightedAverage(List.of(result("v1", 1, 1f), result("v2", 1, 1f))));
}

@Test void explainsHowToEnableGdalWhenBindingsAreAbsent() {
    GdalUnavailableException error = assertThrows(GdalUnavailableException.class,
        () -> new GdalRasterReader().read(Path.of("missing.tif")));
    assertTrue(error.getMessage().contains("org.gdal"));
}
~~~

- [ ] **Step 2: Run tests to verify they fail**

Run: mvn -pl cnn4rs-distributed,cnn4rs-geospatial -am -Dtest=GradientAggregatorTest,GdalAvailabilityTest test

Expected: FAIL because protocol and GDAL API classes are absent.

- [ ] **Step 3: Implement safe optional boundaries**

Task and result records copy lists and reject blank versions, empty batches, empty gradients and non-finite values. Aggregation requires identical version, tensor count and shape, then returns sample-count weighted averages. Its Javadoc explicitly excludes networking, scheduling, retries, authentication and persistence.

The GDAL reader first calls Class.forName for org.gdal.gdal.gdal. If unavailable, it throws GdalUnavailableException explaining that Java bindings and matching native libraries are both required. Once available it invokes only read-only APIs reflectively and destroys datasets in finally. RasterTile uses CHW tensors and validates offsets.

- [ ] **Step 4: Verify**

Run: mvn -pl cnn4rs-distributed,cnn4rs-geospatial -am test && mvn test

Expected: aggregation and absent-GDAL diagnostics pass without GDAL installed.

- [ ] **Step 5: Commit**

~~~bash
git add cnn4rs-distributed cnn4rs-geospatial
git commit -m "feat: add experimental aggregation and optional GDAL APIs"
~~~

### Task 5: Add runnable examples, release documentation and GitHub automation

**Files:**

- Create: examples/src/main/java/io/github/chjers/cnn4rs/examples/InMemoryClassifierExample.java and GdalClassificationExample.java.
- Test: InMemoryClassifierExampleTest.java and ReadmeCommandTest.java in the examples test package.
- Replace: README.md.
- Create: docs/architecture.md, docs/migration-from-legacy.md, docs/gdal-setup.md and docs/citation.md.
- Create: .github/workflows/ci.yml, .github/ISSUE_TEMPLATE/bug_report.yml, .github/ISSUE_TEMPLATE/feature_request.yml and .github/pull_request_template.md.
- Modify: CHANGELOG.md, CONTRIBUTING.md and SECURITY.md.

**Interfaces:**

- Consumes: all public APIs completed by Tasks 1 to 4.
- Produces: an executable core-only quick start, optional GDAL tutorial, migration map, provenance/citation docs and Java 17 CI.

- [ ] **Step 1: Write failing example/README smoke tests**

~~~java
@Test void inMemoryExampleReturnsAClassIndex() {
    assertEquals(0, InMemoryClassifierExample.classifyFirstSample());
}

@Test void readmeExamplePrintsPrediction() {
    String output = captureStandardOut(() -> InMemoryClassifierExample.main(new String[0]));
    assertTrue(output.startsWith("predictedClass="));
}
~~~

- [ ] **Step 2: Run tests to verify they fail**

Run: mvn -pl examples -am -Dtest=InMemoryClassifierExampleTest,ReadmeCommandTest test

Expected: FAIL because the public examples do not exist.

- [ ] **Step 3: Implement examples and documentation**

Use a deterministic in-memory classifier and make its main method emit predictedClass=<integer>. Make the GDAL example accept input/output paths and never contain a drive letter. README includes purpose/non-goals, module table, JDK 17 prerequisite, core-only quick start, optional GDAL link, experimental warning, examples, legacy mapping, test command, MIT license, contribution/security links, thesis citation and unchanged-PDF rights notice. CI runs mvn -B -ntp test on Ubuntu Temurin 17, uploads Surefire reports on failure and never installs GDAL.

- [ ] **Step 4: Run release checks**

Run: mvn -B -ntp test && git diff --check && rg -n 'H:\\|C:\\|System\.exit' README.md cnn4rs-core cnn4rs-geospatial cnn4rs-distributed examples

Expected: Maven exits 0, whitespace check is silent, and the search finds no forbidden hard-coded path or JVM exit in public modules/docs.

- [ ] **Step 5: Commit**

~~~bash
git add README.md docs .github CHANGELOG.md CONTRIBUTING.md SECURITY.md examples
git commit -m "docs: prepare cnn4rs open-source release"
~~~

## Plan self-review

| Specification requirement | Implementing task |
| --- | --- |
| MIT license, JDK 17 Maven layout, untouched thesis | Tasks 1 and 5 |
| Pure-JDK tensor and typed CNN core | Tasks 2 and 3 |
| Determinism, training, checkpoint compatibility | Task 3 |
| Optional GDAL with useful diagnostics | Task 4 |
| Experimental, transport-free distributed behavior | Task 4 |
| Legacy traceability and migration | Tasks 1 and 5 |
| In-memory examples, CI and open-source docs | Task 5 |

The named APIs are produced before downstream tasks consume them. Verification commands are for CI or a user-provided JDK/Maven environment; this work does not install those tools.

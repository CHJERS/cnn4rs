# Final fix report

## Scope

- Added the root-versioned, test-scope `org.junit.jupiter:junit-jupiter` dependency to
  `cnn4rs-geospatial` and `cnn4rs-distributed`. The existing `examples` declaration already
  uses the same dependency, version property, and scope.
- Added focused GDAL boundary tests for a missing constants binding and native-linkage failures
  raised by `AllRegister` and `Open`. The existing nonzero `ReadRaster` status test now also
  proves that this meaningful raster error is not reclassified as `GdalUnavailableException`.
- `GdalRasterReader` now maps unavailable or incompatible constants bindings, missing static
  GDAL methods, and linkage/native-binding failures unwrapped from `AllRegister`/`Open` to
  `GdalUnavailableException`, preserving the original cause. Non-linkage invocation failures
  and raster status errors retain contextual `IllegalStateException` behavior.

## Validation

- Attempted the focused regression command before implementation:
  `mvn -pl cnn4rs-geospatial -am -Dtest=GdalAvailabilityTest test`.
  It could not start because this environment has no `mvn` executable (`/bin/bash: mvn: command not found`).
- Attempted the reactor command after implementation:
  `mvn -B -ntp test`.
  It is blocked by the same missing Maven executable. `java` is also unavailable; neither tool
  was installed.
- Static checks passed:
  - `git diff --check`
  - Every module with JUnit test imports (`cnn4rs-geospatial`, `cnn4rs-distributed`, and
    `examples`) has a JUnit Jupiter dependency with `${junit.jupiter.version}` and `test` scope.
  - Source-level checks confirm constants class-loading failures and static `AllRegister`/`Open`
    linkage failures are handled at the GDAL availability boundary.

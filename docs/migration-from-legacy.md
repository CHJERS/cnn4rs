# 从历史实现迁移

`legacy/` 保存 2020 年研究实现，供论文溯源和概念对照。它不参加 Maven 构建、不是公共 API，且不承诺维持未经测试的历史数值行为。新功能应直接使用 `io.github.chjers.cnn4rs` 下的模块化 API。

| 历史概念或类 | 新 API | 迁移提示 |
| --- | --- | --- |
| `InputLayer` | `Tensor`、`RasterReader` | 用显式 shape 的不可变张量替代裸数组。 |
| `HiddenLayer` | `Conv2D` 加 `ReLU` | 将卷积与激活作为独立层组合。 |
| `PoolMat` | `MaxPool2D` | 显式设置窗口和步长。 |
| `FC`、`OutputLayer` | `Dense`、`Softmax` | 使用 `SequentialModel` 组织分类器。 |
| `bp*` | `Trainer` | 通过 `TrainingExample` 和 `TrainingResult` 驱动单批训练。 |
| 历史 socket master/slave | `TrainingTask`、`GradientResult`、`GradientAggregator` | 仅迁移数据契约；传输与调度由应用自行实现。 |
| 历史 GDAL 工具类 | `RasterReader`、`RasterTile`、`GdalRasterReader` | 仅在部署 GDAL 后使用可选适配器。 |

历史实现含有本机路径、原始 `HashMap` 参数和特定并发假设。不要复制这些细节；请改用 `Path`、类型化配置、不可变 `Tensor`，并在应用层明确管理 I/O 与并发。

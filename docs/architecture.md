# 架构说明

`cnn4rs` 通过 Maven reactor 划分四个边界：`cnn4rs-core` 提供仅依赖 JDK 的张量、层、模型、训练和 checkpoint；`cnn4rs-geospatial` 在其上定义栅格边界，并在调用 `GdalRasterReader` 时才加载可选 GDAL；`cnn4rs-distributed` 提供实验性的任务与梯度聚合值对象；`examples` 只消费公开 API。

## 数据流

```text
内存特征或栅格数据
        ↓
Tensor / RasterReader
        ↓
SequentialModel.predict
        ↓
类别概率与 class index
```

训练时，`Trainer` 使用 `TrainingExample` 更新可训练层，并返回 `TrainingResult`。`ModelCheckpoint` 是版本化的持久化边界。固定配置和随机种子用于获得可重复的初始化；调用者仍应自行记录数据、训练次序和运行环境。

## 可选与实验性边界

GDAL Java 绑定和对应的本机库不是 Maven 构建依赖。缺失或不兼容时，`GdalRasterReader` 会抛出带配置建议的 `GdalUnavailableException`，库不会终止 JVM。

分布式模块只表达 `TrainingTask`、`GradientResult` 和 `GradientAggregator` 的聚合契约，不实现网络传输、线程生命周期或容错系统。它是研究用途的实验性 API，不应被描述为生产级分布式训练方案。

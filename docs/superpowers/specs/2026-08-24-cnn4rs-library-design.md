# cnn4rs 开源类库重构设计

## 目标

将 2020 年研究生项目重组为可阅读、可使用、可扩展的 Java 类库，同时保留论文实现的可追溯性。项目面向遥感影像分类研究和 CNN 基础原理学习，不宣称替代成熟深度学习框架或生产级分布式训练平台。

## 背景与边界

论文《基于分布式 Java 环境的卷积神经网络类库实现——以深圳市高分二号卫星影像数据分类为例》作为研究背景材料原样发布在 `docs/`。README 将标注作者、年份、论文授权页中的权利归属说明，并提示使用者自行确认再利用条件。

旧项目包含手写 CNN 前向传播、Forward-BP、GDAL 栅格处理和 Socket 分布式实验。它目前缺少公开 API、测试、许可、文档和可移植的配置；GDAL、模型计算和训练状态还存在耦合。本次重构不安装或要求开发者机器上的 Maven、JDK、GDAL，也不以复现实验精度作为验收条件。

## 发布决策

- 许可证：MIT。
- Java 基线：JDK 17。
- 构建布局：Maven 多模块项目。
- 文档：以中文为主、提供英文快速开始摘要。
- 旧代码：完整存于 `legacy/`，不再属于新公共 API，且不承诺维持其未测试的历史数值行为。
- 分布式能力：仅作为 `experimental` 模块，不得宣称生产可用。

## 结构

```text
cnn4rs/
├── cnn4rs-core/
│   └── src/main/java/io/github/chjers/cnn4rs/core/
├── cnn4rs-geospatial/
│   └── src/main/java/io/github/chjers/cnn4rs/geospatial/
├── cnn4rs-distributed/
│   └── src/main/java/io/github/chjers/cnn4rs/distributed/
├── examples/
├── legacy/
├── docs/
├── .github/
├── README.md
├── LICENSE
└── pom.xml
```

`cnn4rs-core` 只能依赖 JDK。`cnn4rs-geospatial` 依赖 `core`，并将 GDAL 隔离为可选适配；没有 GDAL 的使用者仍可构建和使用核心模型。`cnn4rs-distributed` 依赖 `core`，只处理可序列化的训练任务、梯度结果和聚合，不允许网络连接、线程生命周期或文件系统路径泄漏至模型 API。`examples` 只依赖公开 API，首个示例使用内存张量，避免样例被外部数据和本机配置阻断。

## 核心 API

### 数据与模型

- `Tensor`：具有显式 shape 的不可变值对象；所有构造、卷积、池化、展平操作先校验维度，杜绝裸 `float[][][]` 跨模块传递。
- `Layer`：统一的前向与反向计算契约；首批实现 `Conv2D`、`MaxPool2D`、`Dense`、`ReLU`、`Softmax`。
- `SequentialModel`：按顺序持有层，提供 `predict` 和训练所需的中间状态，不在构造函数中隐式随机计算。
- `ModelConfig`：显式保存层配置、类别数与随机种子；同一配置和种子必须有确定性初始化。
- `Trainer`：使用可选学习率和损失函数执行单批训练，返回损失和参数更新摘要。
- `ModelCheckpoint`：定义模型参数的版本化读写边界；初版格式仅承诺同一主版本可读取。

### 遥感与分布式

- `RasterReader`、`RasterTile` 和 `RasterWriter`：负责波段读取、归一化、切片与预测栅格写出。无法加载本机 GDAL 时返回可解释的配置错误，不调用 `System.exit`。
- `TrainingTask`、`GradientResult` 和 `GradientAggregator`：表达“分发一个批次、计算梯度、校验并平均”的流程。传输机制是可替换端口；旧 Socket 实现只作为迁移参考，不被当作默认实现。

## 数据流与错误处理

```text
内存训练样本或 GeoTIFF
        ↓
Tensor / RasterReader → Tile
        ↓
SequentialModel.forward
        ↓
Loss + backward → gradients
        ↓
Trainer 更新参数 → 可选 ModelCheckpoint
        ↓
预测类别或 RasterWriter 输出
```

分布式实验数据流为：任务创建者切分训练批次，worker 只返回带模型版本与样本数的梯度结果，聚合器拒绝 shape、版本或非有限数不匹配的结果，随后按样本数加权平均，最后由单一 `Trainer` 更新参数。任何不支持的层、非法 shape、空批次、缺失原生 GDAL 库或错误 checkpoint 都抛出带上下文的异常；库代码不得终止 JVM。

## 兼容性与迁移

保留论文使用的概念映射：`InputLayer` 对应 `Tensor`/`RasterReader`，`HiddenLayer` 对应 `Conv2D + activation`，`PoolMat` 对应 `MaxPool2D`，`FC`/`OutputLayer` 对应 `Dense`/`Softmax`，`bp*` 对应 `Trainer` 的反向传播实现。README 的迁移表将说明这些映射以及旧实现中硬编码路径、原始 `HashMap` 参数和缺乏线程安全保证的限制。

## 测试与验收

测试完全使用小型内存张量并覆盖：

1. 每个层的输入/输出 shape 与非法参数错误；
2. 卷积、池化、Dense、ReLU、Softmax 的已知前向结果；
3. 单批训练使简单样本的损失下降；
4. 固定随机种子时模型初始化一致；
5. checkpoint 往返后的预测一致；
6. 聚合器的加权平均、版本拒绝和 shape 拒绝；
7. 没有 GDAL 时 geospatial 模块给出明确诊断，而 core 测试不受影响；
8. README 中的内存数据案例可按文档命令执行。

发布前需完成 Maven 测试、文档命令检查、格式/链接检查、许可证和文件归属检查。由于本轮不安装运行环境，任何需要 JDK、Maven 或 GDAL 的验证都将在 CI 或用户已有环境中执行，并如实注明。

## 非目标

- 不复刻 TensorFlow、PyTorch 或 Deeplearning4j 的算子覆盖面、性能和设备支持。
- 不承诺在任意机器上自动配置 GDAL 原生库。
- 不将原始 Socket 通信代码宣传为容错、安全或生产级分布式系统。
- 不重新生成、修改或从论文中抽取受版权保护的内容；论文 PDF 只按原文件保存。

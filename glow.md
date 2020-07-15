# Glow编译器学习

## 1.Glow Graph

阅读论文，High-Level IR是计算图。

glow图被构建为一个module，module包含多个Function，每个Function包含多个节点。

存储节点（Storage Node），由module拥有，并且可被该module的所有function访问，类似于C程序中的全局变量。

其他节点由function拥有，表示神经网络的不同操作，例如卷积，池化等。这种节点可访问存储节点。

存储节点为Constant节点和Placeholder节点的基类。

Constant节点可以进行一些优化，因为编译器可以看到其中的内容。例如可以删掉不用的Constant节点，转置，量化，执行Constant Propagation.

一个module一般有inferance function和gradient function。

gradient function训练完后复制给inferance function，inferance function存储权重的Placeholder节点被转换为Constant节点，可进行只有Constant节点才能进行的优化。

### 流程

构造ExecutionEngine -> 引用EE中的Module -> 构造Function -> 构造需要的节点

## 2.Lowering

是High-Level IR的一个优化过程。

将卷积，回归等操作分解为一些矩阵乘法、加法等线性代数操作。

Lowering在differentiated（生成反向图）之后。

### 代码

Optimizer文件夹下

GraphOptimizer.h    lower()

lower.h   lowerNode()

lower.cpp    每种节点的lower。


## 3.differentiate流程

clone Function。

Generate the gradient nodes for each one of the node in the Function.

为需要训练的PH节点创建SGDNode和SaveNode。

将Vector中的节点加入图中。

### differentiate调试过程

for循环节点顺序：

SaveNode -> PH(not train) -> SoftMaxNode -> PH（not train) -> FullyConectedNode -> PH -> PH -> ReshapeNode -> MaxpoolNode -> ReluNode -> ConvolutionNode -> PH -> PH -> MaxpoolNode -> ReluNode -> ConvolutionNode -> PH -> PH -> PH(not train)

添加顺序：

SplatNode -> SoftMaxGradNode -> FullyConectedGradNode -> ReshapeNode -> SplatNode -> MaxpoolGradNode -> ReluGradNode -> ConvolutionGradNode -> SplatNode ->MaxpoolGradNode -> ReluGradNode -> ConvolutionGradNode -> SGDNode -> SavaNode -> SGDNode -> SavaNode -> SGDNode -> SavaNode -> SGDNode -> SavaNode -> SGDNode -> SavaNode -> SGDNode -> SavaNode

## 4.runBatch流程

（PlaceholderBinding 将一个Ph节点和一个支持Tensor绑定，input Tensor 将切片复制给支持Tensor）。

Update the input Placeholder。

Run network。

(batchSize = 8， 每一个epoch只迭代了30次，即每一个epoch只用了240张图的数据，总共用了14400张图)。

### run（）

用unique_ptr创建ExecutionContext，执行run然后释放指针。

### runInternal()

图的一次执行。

### runNetwork()

调用HostManger对象执行。

（The HostManager serves as an entry point into the Runtime environment. It provides an interface to add, run, and evict networks from the host. It handles DeviceManager initialization, houses the Executor, and calls into the Partitioner and Provisioner for network initialization.)

## 5. 类

`Named` ：表示节点名字

`Kinded` ：表示节点类型

`Type` ：表示Tensor类型，包括维度，数据的类型，对齐方式

`NodeValue` ：指明节点的哪一个结果被使用。

`NodeHandle` ：继承NodeValue，必须定义在Node的数据域中，有一个Parent指针指向所在Node。我的理解：NodeHandle表示该节点使用哪一个Node的结果，即使用哪一个NodeValue。

//NodeHandle有两个Node * ，其中一个为NodeValue中的指针，指向被使用的节点，另一个为NodeHandle中定义的指针，指向使用的节点。

`NodeUse` ：只包含NodeHandle指针。我的理解：表示一个节点使用另一个节点的值。

`UseDef<typename UserTY, typename Use>` ：包含一个Use的列表。Node继承UseDef<Node, NodeUse>，即Node中有一个NodeUse的列表。我的理解：该Use的列表即表示该节点使用了哪些节点的值。

使用这个节点的列表。

## CompilationContext

`bindings : PlaceholderBinding`

`partitionConfig : PartitionConfig`

`prepartitionedConfig : PrepartitionedConfig`

`loweredInfoMap : LoweredInfoMap`

`CompilationMode{Train. Infer, NumCompilationMode}`

`backendOpt : backendOpts`

...

## Pipeline

/include/glow/Optimizer/GraphOptimiezerPipeline/Pipiline.h

DCE 死码消除

枚举类 `FunctionPassID` 

//定义一个枚举以标识FunctionPass，用于管线中的FunctionPassConfig内部声明。

枚举类 `ConvergenceMode`

枚举类 `DCERequiedMode`

类 `FunctionPassConfig`

// FunctionPass 的配置参数，private继承`llvm::SmallVector<FunctionPassConfig, 64>`，包括FunctionPassID，convergenceMode，enableCompileMode(编译方式),dceMode.

类 `FunctionPassPipeline`

//需要运行的Pass。由FunctionPassConfig组成。

## PassManger

类 `FunctionPassManger`

//运行一系列FunctionPass的管理器，给定一些Function, CompileContext, 和提供的Pipeline，将在Function上运行所有的pass。

我的理解：简单来说，就是管理Function执行哪些Pass。

## HostManger

Runtime环境的入口，从主机添加，运行，退出计算图的接口。处理DeviceManger的初始化，保存Executor，并调用Partitioner和Provisioner初始化network.

结构体 `NetwordData`

//保存network的数据，包括有向无环图，Moudle的指针，引用的计数。

结构体 `InferRequest`

成员函数

`addNetwork()`

//添加计算图到主机中，并完成一些必要的初始化。包括分割图，预配，编译，初始化后端。为每个Function创建一个有向无环图，存储在network_中.

//会执行在lower之前的优化。

流程：

确认是否已经添加 -> 如果从命令行中加载一个后端选项并且cctx已经包含的话，抛出一个warning -> 加载devices信息 -> 如果包含后端信息，跳过优化过程 -> Function的优化 -> 分割图 -> 分配device -> 

### optimizeBeforeLowering

若从外部加载模型，节点可能已经Lower，此时需要折叠回去 -> 优化

优化过程：创建一个FuncitonPassManger，且创建一个DefaultGraphOptimizationPassPipeline，运行这个Pipeline。

## DAGNode

`children : vector<DAGNode *>`

`parents : vector<DAGNode *>`

`lock : mutex`

`deviceRunTimeInfos : map<DeviceIDTy, DeviceRuntimeInfo>`

`backendName : string`

`logicalDevices : vector<DeviceIDTy>`

`currentDeviceIdx : atomic<unsigned>`

`name : string` // 子图名称

`runtimeBundle : unique_ptr<RuntimeBundle>` // 包含所有子图在运行时所需的信息

`size : uint_64` // 子图包含的Constant和PlaceHolder节点个数

`backendHints : BackendHints` // 后端提示信息对象，由分割器填充，用于传递信息给编译器，比如SRAM的固定和资源的保留。

`backendSpecificOpts : BackendSpecificOptions`

`module : Module *`

## partition

### 类 NodeToFunctionMap

根据这个类的信息doPartitioning().

以FunctionList形式存在的新创建的partitions.

节点到新Functions(partitions)的映射。

数据域：

`functions_ : FunctionList`

`nodeToFunction_ : Map` // 在原来function中的节点到新function的映射

`functionToBackendName_ : FunctionToBackendNameMap` //分割后的子图到后端的映射

`partitionCost_ : PartitionCostMap` //子图到内存消耗的映射

`backendHints_ : BackendHintsMap` //子图的backendHint

`backendSpecificOpts_ : BackendSpecificOptMap`

`logicalDeviceIDMap_ : map<Function *, vector<DeviceIDTy>>`

成员函数：

`Function *operator[](Node *n) { return nodeToFunction_[n] }`

### PartitionConfig

数据域：

`funcName: string`

`numOfPatitions: size_t`

`backendNames: vector<string>`(for each patition)
 
`partitionNames: vector<string>`
  
`backendHints: vector<BackendHints>`

`logicalIDs: vector<vector<unsigned>>`

`nodeToPatition: llvm::StringMap<size_t>`//The mapping between node's name to Partition IDs. 

### PartitionerBase

doPartitioning()函数：

根据NodeToFunctionMap这个映射，将原来的Function分割为几个小Function（子图），每个Function用DAGNode表示信息，相连的Function之间添加SaveNode和PlaceHolder节点，表示上一个子图的输出和下一个子图的输入。

### Partitioner : public PartitionerBase

//根据CompilationContext将Function分解为多个DAG。

//根据内存大小，最小化信息传递的开销。

数据域：

`moudle : Module*`

`F_ : Function*` // 选择内存占用最大的Function表示

`multiBackendNames : bool`

`deviceInfo_ : vector<DeviceInfo>`

`backendHolder_ : vector<unique_ptr<Backend>>`//用于function的优化

`backends_ : vector<Backend* >`

`backendMap_ : map<string, BackendInfo>`

`logicalIDMap_ : map<Function *, vector<DeviceIDTy> >`//the map between partitions and the logicalDeviceID.

// DeviceIDTy = size_t

`logicalDeviceID_ : DeviceIDTy` // 逻辑设备的数量，也就是所需物理设备的数量

`memSize_ : uint_64`

`saturateHost_ : bool` // 是否使用所有可用的设备

`optimized_ : bool` 

`patitionConfig_ : PartitionConfig` // 包含用户定义的partition信息

## ExecutionEngine

## IR

`Instruction`定义指令，`IRFunction`定义转为LIR的指令流，`IRbuilder`负责将高级图中的节点转为指令。

## 添加算子

### 添加HIR算子

*  在 `tools/ClassGen/NodeGen.cpp` 中添加节点相关代码
*  在 `lib/Graph/Nodes.cpp` 中添加验证函数`verify()`
*  在 `lib/Graph/Graph.cpp` 中添加节点创建方法(不要忘记在头文件中添加声明)
*  在 `lib/Graph/Grad.cpp` 中的`differentied()`函数中添加取的梯度节点的代码
*  在 `lib/Optimizer/Lower/Lower.cpp` 中添加相应代码
*  在 `lib/Exporter/ONNXModelWriter.cpp` 中添加相应代码

*  如果添加一个自定义的算子：
   * Tensor布局的要求(比如元素的类型、大小，对齐方式)
   * 如果需要从Caffe2 或者 ONNX加载一个模型，需要在`Importer/Caffe2ModelLoader.cpp` 或者 `Importer/ONNXModelLoader.cpp` 或者 `Importer/CommonOperatorLoader.h` 中实现相应的操作

### 添加LIR指令

*  在`tools/ClassGen/NodeGen.cpp` 中添加指令的相关代码
*  如果自定义从HIR节点到LIR指令，则在`lib/IR/IRGen.cpp`中添加相应代码，否则，在`tools/ClassGen/InstrGen.cpp`中调用`Builder`类的`autoGen("name")`方法，其中，若节点与指令的名字相同，`name`为空
*  在`lib/backends/Interpreter/InterpreterNodes.cpp`中添加相应代码
*  定义在`lib/backends/CPU/Transform.cpp`文件下的`transformPostLowering()`执行在lower之后的优化，将一些节点转化为CPU后端的节点。
*  特定后端的节点和指令定义在`backends`文件夹下
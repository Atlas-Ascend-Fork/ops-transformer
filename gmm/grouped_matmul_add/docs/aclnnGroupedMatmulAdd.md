# aclnnGroupedMatmulAdd

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：沿K轴对x和weight分组，将每组矩阵乘的结果累加到yRef的对应组。

  - 按transposeX处理输入后，第$i$组参与矩阵乘的形状为$[M,k_i]$和$[k_i,N]$。各组的$k_i$可以不同，输出均为$[M,N]$。
- 计算公式：

  $$
  yRef_i^{out}=x_i\times weight_i + yRef_i^{in}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnGroupedMatmulAddGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnGroupedMatmulAdd”接口执行计算。

```c++

aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *groupList,
    aclTensor       *yRef,
    bool             transposeX,
    bool             transposeWeight,
    int64_t          groupType,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnGroupedMatmulAdd(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream
)
```

## aclnnGroupedMatmulAddGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 300px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
      <col style="width: 145px">
      </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>公式中的输入x。</td>
        <td>transposeX为true时，x的shape为[K, M]；为false时，x的shape为[M, K]。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>公式中的weight。</td>
        <td>shape为[K, N]。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
    <tr>
      <td>groupList</td>
      <td>输入</td>
      <td>表示输入和输出分组轴方向的matmul大小分布，Device侧的aclTensor类型。</td>
      <td>仅支持累积和（cumsum模式）。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
        <td>yRef</td>
        <td>输入/输出</td>
        <td>原地累加的输入输出张量。</td>
        <td>shape为[E, M, N]，E为分组数。Atlas A2/A3也支持二维布局[E × M, N]。仅Atlas A2/A3支持FLOAT16、BFLOAT16，此时数据类型须与x一致。</td>
        <td>FLOAT32、FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2、3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>transposeX</td>
        <td>属性</td>
        <td>表示x矩阵是否转置。</td>
        <td>支持true和false。</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeWeight</td>
        <td>属性</td>
        <td>表示weight矩阵是否转置。</td>
        <td>仅支持false。</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupType</td>
        <td>属性</td>
        <td>表示分组类型。</td>
        <td>仅支持2（K轴分组）。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：

    - yRef的shape不支持2维。

  <!-- end id7 -->

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：
    <table style="undefined;table-layout: fixed; width: 1150px"> <colgroup>
    <col style="width: 280px">
    <col style="width: 100px">
    <col style="width: 900px">
      </colgroup><thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>传入的x、weight、groupList、yRef是空指针。</td>
      </tr>
      <tr>
        <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="5">161002</td>
        <td>x、weight、groupList、yRef的数据类型或数据格式不在支持的范围之内。</td>
      </tr>
      <tr>
        <td>x与weight的数据类型不一致。</td>
      </tr>
      <tr>
        <td>x、weight、yRef的shape不满足矩阵乘限制要求。</td>
      </tr>
      <tr>
        <td>transposeWeight为true，或groupType不为2。</td>
      </tr>
      <tr>
        <td>yRef非连续，或16位yRef的视图偏移量、N维不满足对齐要求。</td>
      </tr>
    </tbody>
    </table>

## aclnnGroupedMatmulAdd

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1150px"> <colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 900px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnGroupedMatmulAddGetWorkspaceSize</code>获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
    </tr>
    </tbody></table>

- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- aclnnGroupedMatmulAdd默认确定性实现。
- x和weight中每一组tensor的每一维大小在32字节对齐后都应小于INT32的最大值2147483647。
- 支持的输入类型为：
  - x为FLOAT16、weight为FLOAT16、y为FLOAT32。
  - x为BFLOAT16、weight为BFLOAT16、y为FLOAT32。
  - Atlas A2/A3还支持x、weight和y同为FLOAT16，或同为BFLOAT16。
- 在Ascend 950PR/950DT上，transposeX为false时可能需要额外的workspace调整x的布局，请以第一段接口返回的workspaceSize为准。
- yRef须为连续Tensor。当yRef为FLOAT16或BFLOAT16时，视图偏移量须为非负数且为16个元素的倍数，N须为16的倍数。矩阵乘结果与yRef在FLOAT32中相加，再转换为yRef的数据类型。
- groupList须为非负单调非递减数列（累积和），最后一个值不大于x中tensor的K维。以K=256、E=4（各组大小依次为64、0、128、64）为例：`[64, 64, 192, 256]`。
- groupList第1维最大支持1024，即最多支持1024个group。

<!-- npu="A3,910b" id10 -->
### Atlas A3系列产品、Atlas A2系列产品

- x和weight的最后一维须小于65536。x的最后一维在transposeX为true时为M，为false时为K。weight的最后一维为N。
- 当transposeX为false或yRef为16位时，M、K、N及分组数均须在int32的非负范围内，输出元素数（M × N × 分组数）须在int64范围内。
<!-- end id10 -->

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。
<!-- npu="A3,910b" id8 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：

    ```c++
    #include <iostream>
    #include <vector>
    #include "acl/acl.h"
    #include "aclnnop/aclnn_grouped_matmul_add.h"

    #define CHECK_RET(cond, return_expr) \
      do {                               \
        if (!(cond)) {                   \
          return_expr;                   \
        }                                \
      } while (0)

    #define LOG_PRINT(message, ...)     \
      do {                              \
        printf(message, ##__VA_ARGS__); \
      } while (0)

    int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
        shapeSize *= i;
      }
      return shapeSize;
    }

    int Init(int32_t deviceId, aclrtStream* stream) {
      // 固定写法，资源初始化
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
    }

    template <typename T>
    int CreateAclTensor_New(const std::vector<int64_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请Device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请Device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
      std::vector<T> hostData(GetShapeSize(shape), 0);
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
    }



    int main() {
      // 1.（固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // check根据自己的需要处理
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> xShape = {64, 32};
      std::vector<int64_t> weightShape= {64, 32};
      std::vector<int64_t> yShape = {64, 32};
      std::vector<int64_t> groupListShape = {2};
      std::vector<int64_t> groupListData = {32, 64};
      void* xDeviceAddr;
      void* weightDeviceAddr;
      void* yDeviceAddr;

      void* groupListDeviceAddr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* groupedList = nullptr;
      aclTensor* y = nullptr;
      aclTensor* yRef = nullptr;

      bool transpose_x = true;
      bool transpose_weight = false;
      int group_type = 2;

      // 创建x aclTensor
      ret = CreateAclTensor<uint16_t>(xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weight aclTensor
      ret = CreateAclTensor<uint16_t>(weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建y aclTensor
      ret = CreateAclTensor<float>(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建group_list aclTensor
      ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      yRef = y;

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // 3. 调用CANN算子库API
      // 调用aclnnGroupedMatmulAdd第一段接口
      ret = aclnnGroupedMatmulAddGetWorkspaceSize(x, weight, groupedList, yRef, transpose_x, transpose_weight, group_type, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用aclnnGroupedMatmulAdd第二段接口
      ret = aclnnGroupedMatmulAdd(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulAdd failed. ERROR: %d\n", ret); return ret);

      // 4.（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(yShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr,
                            size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t j = 0; j < size; j++) {
          LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
      }


      // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
      aclDestroyTensor(x);
      aclDestroyTensor(weight);
      aclDestroyTensor(y);

      // 7. 释放device资源，需要根据具体API的接口定义修改
      aclrtFree(xDeviceAddr);
      aclrtFree(weightDeviceAddr);
      aclrtFree(yDeviceAddr);

      if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
    }
    ```

<!-- end id8 -->

<!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：

    ```c++
    #include <iostream>
    #include <vector>
    #include "acl/acl.h"
    #include "aclnnop/aclnn_grouped_matmul_add.h"

    #define CHECK_RET(cond, return_expr) \
      do {                               \
        if (!(cond)) {                   \
          return_expr;                   \
        }                                \
      } while (0)

    #define LOG_PRINT(message, ...)     \
      do {                              \
        printf(message, ##__VA_ARGS__); \
      } while (0)

    int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
        shapeSize *= i;
      }
      return shapeSize;
    }

    int Init(int32_t deviceId, aclrtStream* stream) {
      // 固定写法，资源初始化
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
    }

    template <typename T>
    int CreateAclTensor_New(const std::vector<int64_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请Device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请Device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
      std::vector<T> hostData(GetShapeSize(shape), 0);
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
    }



    int main() {
      // 1.（固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // check根据自己的需要处理
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> xShape = {64, 32};
      std::vector<int64_t> weightShape= {64, 32};
      std::vector<int64_t> yShape = {2, 32, 32};
      std::vector<int64_t> groupListShape = {2};
      std::vector<int64_t> groupListData = {32, 64};
      void* xDeviceAddr;
      void* weightDeviceAddr;
      void* yDeviceAddr;

      void* groupListDeviceAddr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* groupedList = nullptr;
      aclTensor* y = nullptr;
      aclTensor* yRef = nullptr;

      bool transpose_x = true;
      bool transpose_weight = false;
      int group_type = 2;

      // 创建x aclTensor
      ret = CreateAclTensor<uint16_t>(xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weight aclTensor
      ret = CreateAclTensor<uint16_t>(weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建y aclTensor
      ret = CreateAclTensor<float>(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建group_list aclTensor
      ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      yRef = y;

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // 3. 调用CANN算子库API
      // 调用aclnnGroupedMatmulAdd第一段接口
      ret = aclnnGroupedMatmulAddGetWorkspaceSize(x, weight, groupedList, yRef, transpose_x, transpose_weight, group_type, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用aclnnGroupedMatmulAdd第二段接口
      ret = aclnnGroupedMatmulAdd(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulAdd failed. ERROR: %d\n", ret); return ret);

      // 4.（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(yShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr,
                            size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t j = 0; j < size; j++) {
          LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
      }


      // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
      aclDestroyTensor(x);
      aclDestroyTensor(weight);
      aclDestroyTensor(y);

      // 7. 释放device资源，需要根据具体API的接口定义修改
      aclrtFree(xDeviceAddr);
      aclrtFree(weightDeviceAddr);
      aclrtFree(yDeviceAddr);

      if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
    }
    ```
<!-- end id9 -->

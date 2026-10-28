/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/graph.h"
#define private public
#include "platform/platform_info.h"
#undef private
#include "register/register_custom_pass.h"

#include "gmm/grouped_matmul_finalize_routing/op_graph/fusion_pass/grouped_matmul_finalize_routing_transpose_fusion_pass.h"
#include "gmm_csv_ge_parse_utils.h"

#ifndef GMMFR_TRANSPOSE_PASS_CSV_PATH
#define GMMFR_TRANSPOSE_PASS_CSV_PATH ""
#endif

using namespace ge;

namespace {
constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_INDEX = 1;
constexpr int64_t WEIGHT_SCALE_INDEX = 2;

struct TensorWithDesc {
    ge::es::EsTensorHolder tensor;
    TensorDesc desc;
};

struct FusionCase {
    std::string caseName;
    DataType xDtype;
    DataType weightDtype;
    DataType weightScaleDtype;
    bool hasBitcast;
    bool useWeightReshape;
    bool validWeightPattern;
    bool hasMxWeightScaleTranspose;
    Status expectedStatus;
    int expectedTransposeCount;
    int expectedReshapeCount;
    int expectedBitcastCount;
    bool expectedTransposeWeight;
    bool nonUnitWeightReshape = false;
    bool specialFp4StaticChain = false;
    std::string socVersion;
    bool scaleTwoReshapes = false;
    bool validScalePerm = true;
    bool validScaleShape = true;
    bool scaleBitcast = false;
};

static FusionCase ParseCase(const std::vector<std::string>& fields)
{
    constexpr size_t kColumnCount = 20;
    if (fields.size() != kColumnCount) {
        throw std::runtime_error("Expected 20 CSV columns, got " + std::to_string(fields.size()));
    }
    FusionCase param{};
    param.caseName = ops::ut::Trim(fields[0]);
    if (param.caseName.empty()) {
        throw std::runtime_error("CSV case name must not be empty");
    }
    auto parseDtype = [](const std::string& value) {
        const auto dtype = ops::ut::ParseGeDtype(ops::ut::Trim(value));
        if (dtype == DT_UNDEFINED) {
            throw std::runtime_error("Unsupported CSV dtype: " + value);
        }
        return dtype;
    };
    auto parseBool = [](const std::string& value) {
        const auto token = ops::ut::Trim(value);
        if (token != "0" && token != "1") {
            throw std::runtime_error("Expected boolean 0 or 1, got " + token);
        }
        return token == "1";
    };
    param.xDtype = parseDtype(fields[1]);
    param.weightDtype = parseDtype(fields[2]);
    param.weightScaleDtype = parseDtype(fields[3]);
    param.hasBitcast = parseBool(fields[4]);
    param.useWeightReshape = parseBool(fields[5]);
    param.validWeightPattern = parseBool(fields[6]);
    param.hasMxWeightScaleTranspose = parseBool(fields[7]);
    const auto status = ops::ut::Trim(fields[8]);
    if (status == "SUCCESS") {
        param.expectedStatus = SUCCESS;
    } else if (status == "GRAPH_NOT_CHANGED") {
        param.expectedStatus = GRAPH_NOT_CHANGED;
    } else {
        throw std::runtime_error("Unsupported expected status: " + status);
    }
    param.expectedTransposeCount = std::stoi(fields[9]);
    param.expectedReshapeCount = std::stoi(fields[10]);
    param.expectedBitcastCount = std::stoi(fields[11]);
    param.expectedTransposeWeight = parseBool(fields[12]);
    param.nonUnitWeightReshape = parseBool(fields[13]);
    param.specialFp4StaticChain = parseBool(fields[14]);
    param.socVersion = ops::ut::Trim(fields[15]);
    param.scaleTwoReshapes = parseBool(fields[16]);
    param.validScalePerm = parseBool(fields[17]);
    param.validScaleShape = parseBool(fields[18]);
    param.scaleBitcast = parseBool(fields[19]);
    if (param.socVersion != "Ascend950" && param.socVersion != "Ascend350") {
        throw std::runtime_error("Unsupported CSV platform: " + param.socVersion);
    }
    return param;
}

static std::vector<FusionCase> LoadCases()
{
    std::string csvPath = GMMFR_TRANSPOSE_PASS_CSV_PATH;
    if (csvPath.empty()) {
        csvPath = ops::ut::ResolveCsvPath("test_grouped_matmul_finalize_routing_transpose_fusion_pass.csv",
                                          "gmm/grouped_matmul_finalize_routing/tests/ut/op_graph", __FILE__);
    }
    std::ifstream csv(csvPath);
    if (!csv.is_open()) {
        throw std::runtime_error("Failed to open CSV: " + csvPath);
    }
    std::vector<FusionCase> cases;
    std::string line;
    size_t lineNo = 0;
    while (std::getline(csv, line)) {
        ++lineNo;
        if (ops::ut::Trim(line).empty()) {
            continue;
        }
        std::vector<std::string> fields;
        ops::ut::SplitStr2Vec(line, ",", fields);
        if (lineNo == 1 && ops::ut::Trim(fields[0]) == "case_name") {
            continue;
        }
        try {
            cases.emplace_back(ParseCase(fields));
        } catch (const std::exception& error) {
            throw std::runtime_error(ops::ut::BuildCsvParseErrorMessage(csvPath, lineNo, fields[0], error));
        }
    }
    if (cases.empty()) {
        throw std::runtime_error("No test cases in CSV: " + csvPath);
    }
    return cases;
}

class TestGroupedMatmulFinalizeRoutingTransposeFusionPass
    : public ops::GroupedMatmulFinalizeRoutingTransposeFusionPass {
public:
    Status RunForTest(GraphPtr& graph, CustomPassContext& passContext)
    {
        return Run(graph, passContext);
    }
};

static std::vector<std::pair<int64_t, int64_t>> MakeShapeRange(const std::vector<int64_t>& shape)
{
    std::vector<std::pair<int64_t, int64_t>> range;
    range.reserve(shape.size());
    for (auto dim : shape) {
        range.emplace_back(dim, dim);
    }
    return range;
}

static void SetPlatformSupport(const std::string& socVersion = "Ascend950")
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.str_info.short_soc_version = socVersion;
    optionalInfo.soc_version = socVersion;
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    fe::PlatformInfoManager::Instance().platform_info_map_[optionalInfo.soc_version] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

static void ClearPlatformSupport()
{
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
}

static TensorDesc MakeTensorDesc(const std::vector<int64_t>& shape, DataType dtype, Format format = FORMAT_ND)
{
    TensorDesc desc(Shape(shape), format, dtype);
    desc.SetOriginFormat(format);
    desc.SetOriginShape(Shape(shape));
    desc.SetShapeRange(MakeShapeRange(shape));
    return desc;
}

static std::vector<int64_t> TransposeLastTwoDims(const std::vector<int64_t>& shape)
{
    auto transposedShape = shape;
    if (transposedShape.size() >= 2) {
        std::swap(transposedShape[transposedShape.size() - 1], transposedShape[transposedShape.size() - 2]);
    }
    return transposedShape;
}

static GNode BuildUnaryNode(Graph* graph, const char* opType, const char* nodeName, const TensorDesc& outputDesc)
{
    return ge::es::CompliantNodeBuilder(graph)
        .OpType(opType)
        .Name(nodeName)
        .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("y", outputDesc.GetDataType())
        .InstanceOutputShape("y", outputDesc.GetShape().GetDims())
        .InstanceOutputFormat("y", FORMAT_ND)
        .Build();
}

static void LinkUnaryNode(Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& input, GNode& node,
                          const TensorDesc& outputDesc)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(), node, 0);
    node.UpdateInputDesc(0, input.desc);
    node.UpdateOutputDesc(0, outputDesc);
    input.tensor = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
    input.desc = outputDesc;
}

static void AddTranspose(Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& input,
                         const char* nodeName, bool validPerm = true)
{
    auto outputDims = input.desc.GetShape().GetDims();
    std::vector<int64_t> perm;
    if (input.desc.GetDataType() == DT_FLOAT8_E8M0) {
        std::swap(outputDims[1], outputDims[2]);
        perm = {0, 2, 1, 3};
    } else {
        outputDims = TransposeLastTwoDims(outputDims);
        perm = {0, 2, 1};
    }
    auto outputDesc = MakeTensorDesc(outputDims, input.desc.GetDataType());
    auto transposeNode = BuildUnaryNode(graph, "TransposeD", nodeName, outputDesc);
    if (!validPerm) {
        perm = {0, 1, 2};
    }
    transposeNode.SetAttr("perm", perm);
    LinkUnaryNode(graph, graphBuilder, input, transposeNode, outputDesc);
}

static void AddReshape(Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& input, const char* nodeName,
                       bool validShape)
{
    auto outputDims = TransposeLastTwoDims(input.desc.GetShape().GetDims());
    if (!validShape) {
        outputDims = {2, 16, 128};
    }
    auto outputDesc = MakeTensorDesc(outputDims, input.desc.GetDataType());
    auto reshapeNode = BuildUnaryNode(graph, "Reshape", nodeName, outputDesc);
    LinkUnaryNode(graph, graphBuilder, input, reshapeNode, outputDesc);
}

static void AddBitcast(Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& input, const char* nodeName)
{
    auto outputDesc = input.desc;
    outputDesc.SetDataType(DT_HIFLOAT8);
    auto bitcastNode = BuildUnaryNode(graph, "Bitcast", nodeName, outputDesc);
    LinkUnaryNode(graph, graphBuilder, input, bitcastNode, outputDesc);
    DataType bitcastType = DT_HIFLOAT8;
    bitcastNode.SetAttr("type", bitcastType);
}

static void AddStaticFp4WeightChain(Graph* graph, ge::es::EsGraphBuilder& graphBuilder, TensorWithDesc& input)
{
    // This is the static graph emitted by TorchAir for a single-group FP4
    // transposed weight: carrier Reshape -> 2-D Transpose -> Reshape ->
    // Bitcast(FP4) -> packing Reshape.  The pass must remove only the three
    // view nodes before the Bitcast and keep the Bitcast itself.
    const auto sourceDesc = input.desc;
    auto topReshapeDesc = MakeTensorDesc({7168, 1024}, sourceDesc.GetDataType());
    auto topReshape = BuildUnaryNode(graph, "Reshape", "reshape_fp4_top", topReshapeDesc);
    LinkUnaryNode(graph, graphBuilder, input, topReshape, topReshapeDesc);

    auto transposeDesc = MakeTensorDesc({1024, 7168}, sourceDesc.GetDataType());
    auto transpose = BuildUnaryNode(graph, "TransposeD", "transpose_fp4", transposeDesc);
    std::vector<int64_t> transposePerm = {1, 0};
    transpose.SetAttr("perm", transposePerm);
    LinkUnaryNode(graph, graphBuilder, input, transpose, transposeDesc);

    auto bottomReshapeDesc = MakeTensorDesc({1, 1024, 7168}, sourceDesc.GetDataType());
    auto bottomReshape = BuildUnaryNode(graph, "Reshape", "reshape_fp4_bottom", bottomReshapeDesc);
    LinkUnaryNode(graph, graphBuilder, input, bottomReshape, bottomReshapeDesc);

    auto bitcastDesc = MakeTensorDesc({1, 1024, 7168}, DT_FLOAT4_E2M1);
    auto bitcast = BuildUnaryNode(graph, "Bitcast", "bitcast_fp4", bitcastDesc);
    DataType bitcastType = DT_FLOAT4_E2M1;
    bitcast.SetAttr("type", bitcastType);
    LinkUnaryNode(graph, graphBuilder, input, bitcast, bitcastDesc);

    auto tailDesc = MakeTensorDesc({1, 1024, 14336}, DT_FLOAT4_E2M1);
    auto tail = BuildUnaryNode(graph, "Reshape", "reshape_fp4_tail", tailDesc);
    LinkUnaryNode(graph, graphBuilder, input, tail, tailDesc);
}

static void AddScaleReshapeChain(Graph* graph, ge::es::EsGraphBuilder& builder, TensorWithDesc& input, bool validPerm,
                                 bool validShape)
{
    // Keep the group axis and swap only the two layout axes of an FP32 scale.
    // The identity Reshapes retain the legacy three-node pattern explicitly.
    auto topDesc = input.desc;
    auto top = BuildUnaryNode(graph, "Reshape", "reshape_scale_top", topDesc);
    LinkUnaryNode(graph, builder, input, top, topDesc);
    AddTranspose(graph, builder, input, "transpose_weight_scale", validPerm);
    auto bottomDims = input.desc.GetShape().GetDims();
    if (!validShape) {
        bottomDims = {2, 2, 16};
    }
    auto bottomDesc = MakeTensorDesc(bottomDims, input.desc.GetDataType());
    auto bottom = BuildUnaryNode(graph, "Reshape", "reshape_scale_bottom", bottomDesc);
    LinkUnaryNode(graph, builder, input, bottom, bottomDesc);
}

static void LinkGmmInput(Graph* graph, GNode& gmmNode, const TensorWithDesc& input, int64_t inputIndex)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(), gmmNode,
                                     inputIndex);
    gmmNode.UpdateInputDesc(inputIndex, input.desc);
}

static GNode BuildGmmNode(Graph* graph)
{
    return ge::es::CompliantNodeBuilder(graph)
        .OpType("GroupedMatmulFinalizeRouting")
        .Name("GroupedMatmulFinalizeRouting")
        .IrDefInputs({
            {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"w", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"bias", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"pertoken_scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"group_list", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"shared_input", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"logit", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"row_index", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"offset", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
        })
        .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .InstanceOutputDataType("y", DT_BF16)
        .InstanceOutputShape("y", {4, 32})
        .InstanceOutputFormat("y", FORMAT_ND)
        .Build();
}

static GraphPtr BuildGraph(const FusionCase& param)
{
    auto graphBuilder = ge::es::EsGraphBuilder(param.caseName.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto xDesc = MakeTensorDesc({4, 64}, param.xDtype);
    const auto weightShape = param.specialFp4StaticChain ?
                                 std::vector<int64_t>{1, 7168, 1024} :
                                 (param.useWeightReshape && param.validWeightPattern && !param.nonUnitWeightReshape ?
                                      std::vector<int64_t>{2, 1, 64} :
                                      std::vector<int64_t>{2, 32, 64});
    auto weightDesc = MakeTensorDesc(weightShape, param.specialFp4StaticChain ? DT_INT8 : param.weightDtype);
    const auto weightScaleShape =
        param.weightScaleDtype == DT_FLOAT8_E8M0 ?
            std::vector<int64_t>{2, 32, 1, 2} :
            (param.scaleTwoReshapes || param.hasMxWeightScaleTranspose ? std::vector<int64_t>{2, 32, 1} :
                                                                         std::vector<int64_t>{2, 1});
    auto weightScaleDesc = MakeTensorDesc(weightScaleShape, param.weightScaleDtype);

    auto dataX = graphBuilder.CreateInput(0, "dataX", param.xDtype, FORMAT_ND, xDesc.GetShape().GetDims());
    auto dataWeight =
        graphBuilder.CreateInput(1, "dataWeight", param.weightDtype, FORMAT_ND, weightDesc.GetShape().GetDims());
    auto dataWeightScale = graphBuilder.CreateInput(2, "dataWeightScale", param.weightScaleDtype, FORMAT_ND,
                                                    weightScaleDesc.GetShape().GetDims());
    dataX.GetProducer()->UpdateOutputDesc(0, xDesc);
    dataWeight.GetProducer()->UpdateOutputDesc(0, weightDesc);
    dataWeightScale.GetProducer()->UpdateOutputDesc(0, weightScaleDesc);

    TensorWithDesc xInput = {dataX, xDesc};
    TensorWithDesc weightInput = {dataWeight, weightDesc};
    TensorWithDesc weightScaleInput = {dataWeightScale, weightScaleDesc};

    if (param.specialFp4StaticChain) {
        AddStaticFp4WeightChain(graph, graphBuilder, weightInput);
    } else if (param.useWeightReshape) {
        AddReshape(graph, graphBuilder, weightInput, "reshape_weight", param.validWeightPattern);
    } else {
        AddTranspose(graph, graphBuilder, weightInput, "transpose_weight", param.validWeightPattern);
    }
    if (param.hasBitcast) {
        AddBitcast(graph, graphBuilder, weightInput, "bitcast_weight");
    }
    if (param.scaleTwoReshapes) {
        AddScaleReshapeChain(graph, graphBuilder, weightScaleInput, param.validScalePerm, param.validScaleShape);
    } else if (param.hasMxWeightScaleTranspose) {
        AddTranspose(graph, graphBuilder, weightScaleInput, "transpose_weight_scale");
    }
    if (param.scaleBitcast) {
        auto desc = weightScaleInput.desc;
        auto bitcast = BuildUnaryNode(graph, "Bitcast", "bitcast_weight_scale", desc);
        DataType dtype = param.weightScaleDtype;
        bitcast.SetAttr("type", dtype);
        LinkUnaryNode(graph, graphBuilder, weightScaleInput, bitcast, desc);
    }

    auto gmmNode = BuildGmmNode(graph);
    LinkGmmInput(graph, gmmNode, xInput, X_INDEX);
    LinkGmmInput(graph, gmmNode, weightInput, WEIGHT_INDEX);
    LinkGmmInput(graph, gmmNode, weightScaleInput, WEIGHT_SCALE_INDEX);

    auto outputDesc = MakeTensorDesc({4, 32}, DT_BF16);
    gmmNode.UpdateOutputDesc(0, outputDesc);
    bool transposeWeight = false;
    gmmNode.SetAttr("transpose_w", transposeWeight);

    auto output = ge::es::EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(gmmNode, 0));
    return graphBuilder.BuildAndReset({output});
}

static int CountNodes(const GraphPtr& graph, const char* nodeType)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type == nodeType) {
            ++count;
        }
    }
    return count;
}

static GNode FindGmmNode(const GraphPtr& graph)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type == "GroupedMatmulFinalizeRouting") {
            return node;
        }
    }
    ADD_FAILURE() << "GroupedMatmulFinalizeRouting node is not found.";
    return GNode();
}

class GroupedMatmulFinalizeRoutingTransposeFusionPassTest : public testing::TestWithParam<FusionCase> {};

TEST_P(GroupedMatmulFinalizeRoutingTransposeFusionPassTest, RunFusionPass)
{
    const auto& param = GetParam();
    auto graph = BuildGraph(param);

    CustomPassContext passContext;
    TestGroupedMatmulFinalizeRoutingTransposeFusionPass pass;
    SetPlatformSupport(param.socVersion);
    auto status = pass.RunForTest(graph, passContext);
    ClearPlatformSupport();

    EXPECT_EQ(status, param.expectedStatus);
    EXPECT_EQ(CountNodes(graph, "TransposeD"), param.expectedTransposeCount);
    EXPECT_EQ(CountNodes(graph, "Reshape"), param.expectedReshapeCount);
    EXPECT_EQ(CountNodes(graph, "Bitcast"), param.expectedBitcastCount);

    auto gmmNode = FindGmmNode(graph);
    bool transposeWeight = false;
    ASSERT_EQ(gmmNode.GetAttr("transpose_w", transposeWeight), GRAPH_SUCCESS);
    EXPECT_EQ(transposeWeight, param.expectedTransposeWeight);
    if (param.scaleTwoReshapes && param.expectedStatus == SUCCESS) {
        TensorDesc scaleDesc;
        ASSERT_EQ(gmmNode.GetInputDesc(WEIGHT_SCALE_INDEX, scaleDesc), GRAPH_SUCCESS);
        EXPECT_EQ(scaleDesc.GetDataType(), param.weightScaleDtype);
        EXPECT_EQ(scaleDesc.GetShape().GetDims(), (std::vector<int64_t>{2, 32, 1}));
        auto producer = gmmNode.GetInDataNodesAndPortIndexs(WEIGHT_SCALE_INDEX).first;
        ASSERT_NE(producer, nullptr);
        if (param.scaleBitcast) {
            producer = producer->GetInDataNodesAndPortIndexs(0).first;
            ASSERT_NE(producer, nullptr);
        }
        AscendString name;
        ASSERT_EQ(producer->GetName(name), GRAPH_SUCCESS);
        EXPECT_STREQ(name.GetString(), "dataWeightScale");
    }
}

std::string TestName(const testing::TestParamInfo<FusionCase>& info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

INSTANTIATE_TEST_CASE_P(GroupedMatmulFinalizeRoutingTransposeFusionPass,
                        GroupedMatmulFinalizeRoutingTransposeFusionPassTest, testing::ValuesIn(LoadCases()), TestName);
} // namespace

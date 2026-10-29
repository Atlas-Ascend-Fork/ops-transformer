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
#include "register/register_custom_pass.h"

#include "gmm/grouped_matmul_swiglu_quant_v2/op_graph/fusion_pass/grouped_matmul_swiglu_quant_transpose_fusion_pass.h"
#include "gmm_csv_ge_parse_utils.h"

#ifndef GMMSWIGLU_TRANSPOSE_PASS_CSV_PATH
#define GMMSWIGLU_TRANSPOSE_PASS_CSV_PATH ""
#endif

using namespace ge;

namespace {
constexpr int64_t X_INDEX = 0;
constexpr int64_t X_SCALE_INDEX = 1;
constexpr int64_t GROUP_LIST_INDEX = 2;
constexpr int64_t WEIGHT_INDEX = 3;
constexpr int64_t WEIGHT_SCALE_INDEX = 4;

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
    bool specialTail;
    bool shapeUser;
    bool twoReshapes;
    bool dynamicShape;
    std::string shapeCase;
    DataType outputDtype;
    bool scaleReshape;
};

FusionCase ParseCase(const std::vector<std::string>& fields)
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
    auto parseDtype = [](const std::string& field) {
        auto dtype = ops::ut::ParseGeDtype(field);
        if (dtype == DT_UNDEFINED) {
            throw std::runtime_error("Unsupported dtype: " + field);
        }
        return dtype;
    };
    auto parseBool = [](const std::string& field) {
        const auto token = ops::ut::Trim(field);
        if (token != "0" && token != "1") {
            throw std::runtime_error("Expected boolean 0 or 1, got " + token);
        }
        return token == "1";
    };
    auto parseCount = [](const std::string& field) {
        const auto token = ops::ut::Trim(field);
        size_t parsed = 0;
        const auto count = std::stoi(token, &parsed);
        if (parsed != token.size() || count < 0) {
            throw std::runtime_error("Expected non-negative node count, got " + token);
        }
        return count;
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
    param.expectedTransposeCount = parseCount(fields[9]);
    param.expectedReshapeCount = parseCount(fields[10]);
    param.expectedBitcastCount = parseCount(fields[11]);
    param.expectedTransposeWeight = parseBool(fields[12]);
    param.specialTail = parseBool(fields[13]);
    param.shapeUser = parseBool(fields[14]);
    param.twoReshapes = parseBool(fields[15]);
    param.dynamicShape = parseBool(fields[16]);
    param.shapeCase = ops::ut::Trim(fields[17]);
    if (param.shapeCase != "normal" && param.shapeCase != "k1" && param.shapeCase != "m1" && param.shapeCase != "e1") {
        throw std::runtime_error("Unsupported shape case: " + param.shapeCase);
    }
    param.outputDtype = parseDtype(fields[18]);
    param.scaleReshape = parseBool(fields[19]);
    return param;
}

std::vector<FusionCase> LoadCases()
{
    std::string csvPath = GMMSWIGLU_TRANSPOSE_PASS_CSV_PATH;
    if (csvPath.empty()) {
        csvPath = ops::ut::ResolveCsvPath("test_grouped_matmul_swiglu_quant_transpose_fusion_pass.csv",
                                          "gmm/grouped_matmul_swiglu_quant_v2/tests/ut/op_graph", __FILE__);
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

class TestGroupedMatmulSwigluQuantTransposeFusionPass : public ops::GroupedMatmulSwigluQuantTransposeFusionPass {
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

static void LinkGmmInput(Graph* graph, GNode& gmmNode, const TensorWithDesc& input, int64_t inputIndex)
{
    ge::es::AddEdgeAndUpdatePeerDesc(*graph, *input.tensor.GetProducer(), input.tensor.GetProducerOutIndex(), gmmNode,
                                     inputIndex);
    gmmNode.UpdateInputDesc(inputIndex, input.desc);
}

static void AddPackedWeightChain(Graph* graph, ge::es::EsGraphBuilder& builder, TensorWithDesc& weight,
                                 const FusionCase& param)
{
    const auto sourceDims = weight.desc.GetShape().GetDims();
    if (param.twoReshapes) {
        auto topDesc = MakeTensorDesc({sourceDims[0] * sourceDims[1], sourceDims[2]}, weight.desc.GetDataType());
        auto top = BuildUnaryNode(graph, "Reshape", "top_weight_reshape", topDesc);
        LinkUnaryNode(graph, builder, weight, top, topDesc);
        auto transDesc = MakeTensorDesc({sourceDims[2], sourceDims[0] * sourceDims[1]}, weight.desc.GetDataType());
        auto trans = BuildUnaryNode(graph, "TransposeD", "transpose_weight", transDesc);
        std::vector<int64_t> perm = {1, 0};
        trans.SetAttr("perm", perm);
        LinkUnaryNode(graph, builder, weight, trans, transDesc);
        auto bottomDesc = MakeTensorDesc({sourceDims[0], sourceDims[2], sourceDims[1]}, weight.desc.GetDataType());
        auto bottom = BuildUnaryNode(graph, "Reshape", "bottom_weight_reshape", bottomDesc);
        LinkUnaryNode(graph, builder, weight, bottom, bottomDesc);
    } else {
        AddTranspose(graph, builder, weight, "transpose_weight");
    }
    const auto carrierDims = weight.desc.GetShape().GetDims();
    TensorWithDesc shapeInput = weight;
    if (param.shapeUser) {
        auto shapeDesc = MakeTensorDesc({3}, DT_INT32);
        auto shape = BuildUnaryNode(graph, "Shape", "weight_shape", shapeDesc);
        LinkUnaryNode(graph, builder, shapeInput, shape, shapeDesc);
        if (param.hasBitcast) {
            auto factors = builder.CreateInput(5, "packing_factors", DT_INT32, FORMAT_ND, {3});
            TensorWithDesc factorInput{factors, shapeDesc};
            auto mul = ge::es::CompliantNodeBuilder(graph)
                           .OpType("Mul")
                           .Name("packed_weight_shape")
                           .IrDefInputs({{"x1", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"x2", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .InstanceOutputDataType("y", DT_INT32)
                           .InstanceOutputShape("y", {3})
                           .InstanceOutputFormat("y", FORMAT_ND)
                           .Build();
            LinkGmmInput(graph, mul, shapeInput, 0);
            LinkGmmInput(graph, mul, factorInput, 1);
            shapeInput = {ge::es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(mul, 0)),
                          shapeDesc};
        }
    }
    auto logicalDims = carrierDims;
    if (param.hasBitcast) {
        auto bitcastDims = carrierDims;
        bitcastDims.emplace_back(2);
        auto bitcastDesc = MakeTensorDesc(bitcastDims, DT_FLOAT4_E2M1);
        auto bitcast = BuildUnaryNode(graph, "Bitcast", "bitcast_weight", bitcastDesc);
        DataType dtype = DT_FLOAT4_E2M1;
        bitcast.SetAttr("type", dtype);
        LinkUnaryNode(graph, builder, weight, bitcast, bitcastDesc);
        logicalDims.back() *= 2;
    }
    auto tailDesc = MakeTensorDesc(logicalDims, DT_FLOAT4_E2M1);
    auto tail = ge::es::CompliantNodeBuilder(graph)
                    .OpType("Reshape")
                    .Name("tail_weight_reshape")
                    .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"shape", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                    .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", DT_FLOAT4_E2M1)
                    .InstanceOutputShape("y", logicalDims)
                    .InstanceOutputFormat("y", FORMAT_ND)
                    .Build();
    LinkUnaryNode(graph, builder, weight, tail, tailDesc);
    if (param.shapeUser) {
        LinkGmmInput(graph, tail, shapeInput, 1);
    }
    if (param.dynamicShape) {
        weight.desc = MakeTensorDesc({-1, -1, -1}, DT_FLOAT4_E2M1);
        tail.UpdateOutputDesc(0, weight.desc);
    }
}

static GNode BuildGmmNode(Graph* graph)
{
    return ge::es::CompliantNodeBuilder(graph)
        .OpType("GroupedMatmulSwigluQuantV2")
        .Name("GroupedMatmulSwigluQuantV2")
        .IrDefInputs({
            {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x_scale", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"group_list", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"weight", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
            {"weight_scale", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
            {"weight_assist_matrix", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
            {"bias", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
            {"smooth_scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
        })
        .IrDefOutputs({
            {"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"y_scale", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
        })
        .InstanceOutputDataType("y", DT_INT8)
        .InstanceOutputShape("y", {4, 32})
        .InstanceOutputFormat("y", FORMAT_ND)
        .InstanceOutputDataType("y_scale", DT_FLOAT)
        .InstanceOutputShape("y_scale", {4, 32})
        .InstanceOutputFormat("y_scale", FORMAT_ND)
        .Build();
}

static GraphPtr BuildGraph(const FusionCase& param)
{
    auto graphBuilder = ge::es::EsGraphBuilder(param.caseName.c_str());
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto xDesc = MakeTensorDesc({param.shapeCase == "m1" ? 1 : 4, param.shapeCase == "k1" ? 1 : 64}, param.xDtype);
    auto xScaleDesc = MakeTensorDesc({4, 1}, DT_FLOAT);
    auto groupListDesc = MakeTensorDesc({2}, DT_INT64);
    auto weightShape = param.useWeightReshape && param.validWeightPattern ? std::vector<int64_t>{2, 1, 64} :
                                                                            std::vector<int64_t>{2, 32, 64};
    if (param.shapeCase == "k1") {
        weightShape = {param.dynamicShape ? -1 : 2, 32, 1};
    } else if (param.shapeCase == "e1") {
        weightShape = {1, 32, 64};
    }
    auto weightDesc =
        MakeTensorDesc(param.specialTail ? std::vector<int64_t>{2, 4736, param.hasBitcast ? 391 : 782} : weightShape,
                       param.weightDtype);
    const auto weightScaleShape =
        param.weightScaleDtype == DT_FLOAT8_E8M0 ? std::vector<int64_t>{2, 32, 1, 2} : std::vector<int64_t>{2, 1};
    auto weightScaleDesc = MakeTensorDesc(param.specialTail ? std::vector<int64_t>{2, 4736, 13, 2} : weightScaleShape,
                                          param.weightScaleDtype);

    auto dataX = graphBuilder.CreateInput(0, "dataX", param.xDtype, FORMAT_ND, xDesc.GetShape().GetDims());
    auto dataXScale = graphBuilder.CreateInput(1, "dataXScale", DT_FLOAT, FORMAT_ND, xScaleDesc.GetShape().GetDims());
    auto dataGroupList =
        graphBuilder.CreateInput(2, "dataGroupList", DT_INT64, FORMAT_ND, groupListDesc.GetShape().GetDims());
    auto dataWeight =
        graphBuilder.CreateInput(3, "dataWeight", param.weightDtype, FORMAT_ND, weightDesc.GetShape().GetDims());
    auto dataWeightScale = graphBuilder.CreateInput(4, "dataWeightScale", param.weightScaleDtype, FORMAT_ND,
                                                    weightScaleDesc.GetShape().GetDims());
    dataX.GetProducer()->UpdateOutputDesc(0, xDesc);
    dataXScale.GetProducer()->UpdateOutputDesc(0, xScaleDesc);
    dataGroupList.GetProducer()->UpdateOutputDesc(0, groupListDesc);
    dataWeight.GetProducer()->UpdateOutputDesc(0, weightDesc);
    dataWeightScale.GetProducer()->UpdateOutputDesc(0, weightScaleDesc);

    TensorWithDesc xInput = {dataX, xDesc};
    TensorWithDesc xScaleInput = {dataXScale, xScaleDesc};
    TensorWithDesc groupListInput = {dataGroupList, groupListDesc};
    TensorWithDesc weightInput = {dataWeight, weightDesc};
    TensorWithDesc weightScaleInput = {dataWeightScale, weightScaleDesc};

    if (param.specialTail) {
        AddPackedWeightChain(graph, graphBuilder, weightInput, param);
    } else if (param.useWeightReshape) {
        AddReshape(graph, graphBuilder, weightInput, "reshape_weight", param.validWeightPattern);
    } else {
        AddTranspose(graph, graphBuilder, weightInput, "transpose_weight", param.validWeightPattern);
    }
    if (param.hasBitcast && !param.specialTail) {
        AddBitcast(graph, graphBuilder, weightInput, "bitcast_weight");
    }
    if (param.scaleReshape) {
        auto desc = MakeTensorDesc({2, 1, 32, 2}, param.weightScaleDtype);
        auto reshape = BuildUnaryNode(graph, "Reshape", "reshape_weight_scale", desc);
        LinkUnaryNode(graph, graphBuilder, weightScaleInput, reshape, desc);
    } else if (param.hasMxWeightScaleTranspose) {
        AddTranspose(graph, graphBuilder, weightScaleInput, "transpose_weight_scale");
    }

    auto gmmNode = BuildGmmNode(graph);
    LinkGmmInput(graph, gmmNode, xInput, X_INDEX);
    LinkGmmInput(graph, gmmNode, xScaleInput, X_SCALE_INDEX);
    LinkGmmInput(graph, gmmNode, groupListInput, GROUP_LIST_INDEX);
    LinkGmmInput(graph, gmmNode, weightInput, WEIGHT_INDEX);
    LinkGmmInput(graph, gmmNode, weightScaleInput, WEIGHT_SCALE_INDEX);

    auto outputDesc = MakeTensorDesc({param.shapeCase == "m1" ? 1 : 4, 16}, param.outputDtype);
    auto outputScaleDesc = MakeTensorDesc({4, 32}, DT_FLOAT);
    gmmNode.UpdateOutputDesc(0, outputDesc);
    gmmNode.UpdateOutputDesc(1, outputScaleDesc);
    bool transposeWeight = false;
    gmmNode.SetAttr("transpose_weight", transposeWeight);

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
        if (node.GetType(type) == GRAPH_SUCCESS && type == "GroupedMatmulSwigluQuantV2") {
            return node;
        }
    }
    ADD_FAILURE() << "GroupedMatmulSwigluQuantV2 node is not found.";
    return GNode();
}

class GroupedMatmulSwigluQuantTransposeFusionPassTest : public testing::TestWithParam<FusionCase> {};

TEST_P(GroupedMatmulSwigluQuantTransposeFusionPassTest, RunFusionPass)
{
    const auto& param = GetParam();
    auto graph = BuildGraph(param);
    TensorDesc originalScaleDesc;
    auto originalGmm = FindGmmNode(graph);
    ASSERT_EQ(originalGmm.GetInputDesc(WEIGHT_SCALE_INDEX, originalScaleDesc), GRAPH_SUCCESS);
    auto originalScaleSource = originalGmm.GetInDataNodesAndPortIndexs(WEIGHT_SCALE_INDEX).first;
    ASSERT_NE(originalScaleSource, nullptr);
    AscendString originalScaleName;
    ASSERT_EQ(originalScaleSource->GetName(originalScaleName), GRAPH_SUCCESS);

    CustomPassContext passContext;
    TestGroupedMatmulSwigluQuantTransposeFusionPass pass;
    auto status = pass.RunForTest(graph, passContext);

    EXPECT_EQ(status, param.expectedStatus);
    EXPECT_EQ(CountNodes(graph, "TransposeD"), param.expectedTransposeCount);
    EXPECT_EQ(CountNodes(graph, "Reshape"), param.expectedReshapeCount);
    EXPECT_EQ(CountNodes(graph, "Bitcast"), param.expectedBitcastCount);

    auto gmmNode = FindGmmNode(graph);
    bool transposeWeight = false;
    ASSERT_EQ(gmmNode.GetAttr("transpose_weight", transposeWeight), GRAPH_SUCCESS);
    EXPECT_EQ(transposeWeight, param.expectedTransposeWeight);
    if (param.weightScaleDtype != DT_FLOAT8_E8M0) {
        TensorDesc scaleDesc;
        ASSERT_EQ(gmmNode.GetInputDesc(WEIGHT_SCALE_INDEX, scaleDesc), GRAPH_SUCCESS);
        EXPECT_EQ(scaleDesc.GetDataType(), originalScaleDesc.GetDataType());
        EXPECT_EQ(scaleDesc.GetShape().GetDims(), originalScaleDesc.GetShape().GetDims());
        auto source = gmmNode.GetInDataNodesAndPortIndexs(WEIGHT_SCALE_INDEX).first;
        ASSERT_NE(source, nullptr);
        AscendString name;
        ASSERT_EQ(source->GetName(name), GRAPH_SUCCESS);
        EXPECT_EQ(name, originalScaleName);
    }
    if (param.expectedStatus == SUCCESS && !param.specialTail) {
        TensorDesc desc;
        ASSERT_EQ(gmmNode.GetInputDesc(WEIGHT_INDEX, desc), GRAPH_SUCCESS);
        EXPECT_EQ(desc.GetDataType(), param.hasBitcast ? DT_HIFLOAT8 : param.weightDtype);
        auto [source, port] = gmmNode.GetInDataNodesAndPortIndexs(WEIGHT_INDEX);
        ASSERT_NE(source, nullptr);
        if (param.hasBitcast) {
            source = source->GetInDataNodesAndPortIndexs(0).first;
            ASSERT_NE(source, nullptr);
        }
        AscendString sourceName;
        ASSERT_EQ(source->GetName(sourceName), GRAPH_SUCCESS);
        EXPECT_STREQ(sourceName.GetString(), "dataWeight");
    }
    if (param.specialTail && param.expectedStatus == SUCCESS) {
        TensorDesc weightDesc;
        ASSERT_EQ(gmmNode.GetInputDesc(WEIGHT_INDEX, weightDesc), GRAPH_SUCCESS);
        EXPECT_EQ(weightDesc.GetShape().GetDims(), (std::vector<int64_t>{2, 4736, 782}));
        EXPECT_EQ(weightDesc.GetDataType(), DT_FLOAT4_E2M1);
        auto [tail, tailPort] = gmmNode.GetInDataNodesAndPortIndexs(WEIGHT_INDEX);
        ASSERT_NE(tail, nullptr);
        AscendString name;
        ASSERT_EQ(tail->GetName(name), GRAPH_SUCCESS);
        EXPECT_EQ(std::string(name.GetString()), "tail_weight_reshape");
        auto [tailInput, sourcePort] = tail->GetInDataNodesAndPortIndexs(0);
        ASSERT_NE(tailInput, nullptr);
        ASSERT_EQ(tailInput->GetName(name), GRAPH_SUCCESS);
        EXPECT_EQ(std::string(name.GetString()), param.hasBitcast ? "bitcast_weight" : "dataWeight");
        if (param.shapeUser) {
            for (auto node : graph->GetAllNodes()) {
                ASSERT_EQ(node.GetName(name), GRAPH_SUCCESS);
                if (std::string(name.GetString()) != "weight_shape") {
                    continue;
                }
                auto [shapeSource, shapePort] = node.GetInDataNodesAndPortIndexs(0);
                ASSERT_NE(shapeSource, nullptr);
                ASSERT_EQ(shapeSource->GetName(name), GRAPH_SUCCESS);
                EXPECT_EQ(std::string(name.GetString()), "dataWeight");
            }
        }
    }
}

std::string TestName(const testing::TestParamInfo<FusionCase>& info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

INSTANTIATE_TEST_SUITE_P(GroupedMatmulSwigluQuantTransposeFusionPass, GroupedMatmulSwigluQuantTransposeFusionPassTest,
                         testing::ValuesIn(LoadCases()), TestName);
} // namespace

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_PREPARE_KV_OFFSETS_ARCH35_MXFP4_HPP
#define GBSA_BLOCK_EPILOGUE_PREPARE_KV_OFFSETS_ARCH35_MXFP4_HPP

#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../attn_infra/epilogue/gbsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/block/gbsa_block_epilogue_arch35_utils.hpp"
#include "../../../arch35/gbsa_layout_utils.hpp"
#include "../../gemm/tile_common/gbsa_build_kv_records_mxfp4.hpp"

namespace NpuArch::Epilogue {
// Packed MXFP4 offsets are byte offsets. They are not an element-size
// extension of the shared non-quantized/FP8 descriptor policy.
template <bool Paged, bool Residual>
struct EpilogueAtlasA5PrepareKvOffsetsMxfp4 {
    using ArchTag = Arch::AtlasA5;
    static constexpr bool PAGED_CACHE_FLAG = Paged;
    static constexpr bool RESIDUAL_BLOCK_FLAG = Residual;
    static constexpr uint32_t COPY_ALIGNMENT = 16U;
};
} // namespace NpuArch::Epilogue

namespace NpuArch::Epilogue::Block {

template <bool PAGED_CACHE_FLAG_, bool RESIDUAL_BLOCK_FLAG_>
class BlockEpilogue<EpilogueAtlasA5PrepareKvOffsetsMxfp4<PAGED_CACHE_FLAG_, RESIDUAL_BLOCK_FLAG_>> {
public:
    using DispatchPolicy = EpilogueAtlasA5PrepareKvOffsetsMxfp4<PAGED_CACHE_FLAG_, RESIDUAL_BLOCK_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    static constexpr bool PAGED_CACHE_FLAG = DispatchPolicy::PAGED_CACHE_FLAG;
    static constexpr bool RESIDUAL_BLOCK_FLAG = DispatchPolicy::RESIDUAL_BLOCK_FLAG;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, UBufTileHelper& uBufTileHelper,
                                    uint32_t tileSize, uint32_t l1TileN, uint32_t blockY, uint32_t pageSize,
                                    uint64_t tokenStride, uint64_t kPageStride, uint32_t runtimeTokenBytes)
        : tileSize_(tileSize),
          l1TileN_(l1TileN),
          blockY_(blockY),
          pageSize_(pageSize),
          tokenStride_(tokenStride),
          kPageStride_(kPageStride),
          tokenBytes_(runtimeTokenBytes)
    {
        // The mask reservation is unused by the current sparse softmax path.
        groupWords_ = GbsaKernelArch35::CalcKvOffsetGroupWords<PAGED_CACHE_FLAG>(l1TileN_, blockY_);
        recordsBufMask_ = uBufTileHelper.kvOffsetUbBufNum - 1U;
        for (uint32_t i = 0; i < uBufTileHelper.kvOffsetUbBufNum; ++i) {
            recordsBuf_[i] = resource.ubBuf.template GetBufferByByte<uint64_t>(uBufTileHelper.maskStartOffset +
                                                                               i * groupWords_ * sizeof(uint64_t));
        }
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<uint64_t> info,
                                      const __gm__ int32_t* __restrict__ sparseIdx,
                                      const __gm__ int32_t* __restrict__ blockTable, uint32_t tileIdx,
                                      uint32_t tileTokens, uint32_t explicitCount, uint32_t residualId)
    {
        Gemm::Tile::KvRecordsOnly consumer;
        (*this)(info, sparseIdx, blockTable, tileIdx, tileTokens, explicitCount, residualId, consumer);
    }

    // Resolve each selected span directly against the independent PA page pitches.
    // K/V records share the destination order, while each has its own DMA stride.
    template <class Scale, bool PrecomputeKScale>
    struct PagePitchWriter {
        static constexpr bool PRECOMPUTE_KSCALE = PrecomputeKScale;
        struct Address {
            uint64_t key, value, scale;
        };

        Scale& scale;
        uint64_t keyPage, valuePage, scalePage;
        uint64_t tokenStride;
        __ubuf__ uint64_t* keyRecords = nullptr;
        uint32_t count = 0;
        uint32_t pendingRows = 0;
        uint32_t previousPage = 0;
        uint32_t previousEnd = 0;
        uint64_t logicalPage = ~0ULL;
        uint32_t physicalPage = 0;
        Address pageBase{}, pending{};
        typename Scale::FullBlockCopy fullBlockCopy{};

        __aicore__ inline void BeginGroup(uint32_t group, uint32_t rows, __ubuf__ uint64_t* records)
        {
            scale.BeginGroup(group, rows);
            keyRecords = records;
            count = 0;
            pendingRows = 0;
            logicalPage = ~0ULL;
        }

        __aicore__ inline void ResolvePage(uint64_t logical, const __gm__ int32_t* pages)
        {
            if (logical == logicalPage) {
                return;
            }
            logicalPage = logical;
            physicalPage = static_cast<uint32_t>(pages[logical]);
            const uint64_t page = physicalPage;
            pageBase = {page * keyPage, page * valuePage, page * scalePage};
        }

        template <uint32_t BlockRows>
        __aicore__ inline void RecordScale(uint64_t offset, uint32_t rows, uint64_t stride, uint32_t matrices)
        {
            if constexpr (BlockRows == 0U) {
                scale.Record(offset, rows, stride, matrices);
            } else if constexpr (PrecomputeKScale) {
                scale.template RecordFullBlocks<BlockRows>(offset, rows, stride, matrices, fullBlockCopy);
            } else {
                scale.template Record<BlockRows>(offset, rows, stride, matrices);
            }
        }

        __aicore__ inline void Write(const Address& source, uint32_t rows, uint64_t keyStride, uint64_t valueStride,
                                     uint32_t matrices)
        {
            const uint32_t word = 8U + 4U * count++;
            const uint64_t meta = (static_cast<uint64_t>(rows) << 1U) | (matrices - 1U);
            keyRecords[word] = source.key;
            keyRecords[word + 1U] = (keyStride << 32U) | meta;
            keyRecords[word + 2U] = source.value;
            keyRecords[word + 3U] = (valueStride << 32U) | meta;
        }

        template <uint32_t BlockRows>
        __aicore__ inline void EmitPending()
        {
            Write(pending, pendingRows, 0U, 0U, 1U);
            RecordScale<BlockRows>(pending.scale, pendingRows, 0U, 1U);
            pendingRows = 0;
        }

        template <uint32_t BlockRows = 0U>
        __aicore__ inline void Append(uint32_t inner, uint32_t rows, uint32_t blockY)
        {
            if (pendingRows != 0U && physicalPage == previousPage && inner == previousEnd) {
                pendingRows += rows;
                previousEnd = inner + rows;
                return;
            }
            const uint64_t offset = inner * tokenStride;
            const Address next{pageBase.key + offset, pageBase.value + offset, pageBase.scale + offset};
            if (pendingRows == blockY && rows == blockY) {
                if constexpr (PrecomputeKScale) {
                    const bool reverse = next.key < pending.key;
                    const Address& first = reverse ? next : pending;
                    const Address& second = reverse ? pending : next;
                    const uint64_t keyStride = second.key - first.key;
                    const uint64_t valueStride = second.value - first.value;
                    const uint64_t scaleStride = second.scale - first.scale;
                    const uint64_t span = rows * tokenStride;
                    if (keyStride >= span && keyStride <= 0xFFFFFFFFULL && valueStride >= span &&
                        valueStride <= 0xFFFFFFFFULL &&
                        ((keyStride | valueStride) % DispatchPolicy::COPY_ALIGNMENT) == 0U) {
                        Write(first, rows, keyStride, valueStride, 2U);
                        // Scale has an independent pitch. Fall back to two copies
                        // when its source gap cannot use the two-span DMA format.
                        if (scaleStride >= span && scaleStride <= 0xFFFFFFFFULL && scaleStride % 16U == 0U) {
                            RecordScale<BlockRows>(first.scale, rows, scaleStride, 2U);
                        } else {
                            RecordScale<BlockRows>(first.scale, rows, 0U, 1U);
                            RecordScale<BlockRows>(second.scale, rows, 0U, 1U);
                        }
                        pendingRows = 0;
                        return;
                    }
                } else {
                    const bool reverse = next.key < pending.key;
                    const uint64_t keyStride = reverse ? pending.key - next.key : next.key - pending.key;
                    const uint64_t valueStride = reverse ? pending.value - next.value : next.value - pending.value;
                    const uint64_t span = rows * tokenStride;
                    // Validated page pitches are 16-byte aligned and cover a full page.
                    // Two spans in different pages cannot overlap. In one page
                    // the K, V and packed-scale offsets have the same difference.
                    if (keyStride >= span && keyStride <= 0xFFFFFFFFULL && valueStride <= 0xFFFFFFFFULL) {
                        const Address first{reverse ? next.key : pending.key, reverse ? next.value : pending.value,
                                            reverse ? next.scale : pending.scale};
                        Write(first, rows, keyStride, valueStride, 2U);
                        const uint64_t secondScale = reverse ? pending.scale : next.scale;
                        const uint64_t scaleStride = secondScale - first.scale;
                        if (scaleStride <= 0xFFFFFFFFULL) {
                            RecordScale<BlockRows>(first.scale, rows, scaleStride, 2U);
                        } else {
                            RecordScale<BlockRows>(first.scale, rows, 0U, 1U);
                            RecordScale<BlockRows>(secondScale, rows, 0U, 1U);
                        }
                        pendingRows = 0;
                        return;
                    }
                }
            }
            if (pendingRows != 0U) {
                EmitPending<BlockRows>();
            }
            pending = next;
            pendingRows = rows;
            previousPage = physicalPage;
            previousEnd = inner + rows;
        }

        template <uint32_t BlockRows = 0U>
        __aicore__ inline void Finish()
        {
            if (pendingRows != 0U) {
                EmitPending<BlockRows>();
            }
            scale.EndGroup(keyRecords, count, true);
        }
    };

    template <uint32_t BlockRows, class Writer>
    __aicore__ inline void BuildFullStridedPagedGroup(const __gm__ int32_t* indices, const __gm__ int32_t* pages,
                                                      uint32_t blocks, Writer& writer)
    {
        constexpr uint32_t blocksPerPage = 128U / BlockRows;
        if (blocks == blocksPerPage) {
            const uint32_t first = static_cast<uint32_t>(indices[0]);
            if (first % blocksPerPage == 0U &&
                static_cast<uint32_t>(indices[blocksPerPage - 1U]) == first + blocksPerPage - 1U) {
                bool contiguous = true;
                for (uint32_t i = 1U; i + 1U < blocksPerPage; ++i) {
                    if (static_cast<uint32_t>(indices[i]) != first + i) {
                        contiguous = false;
                        break;
                    }
                }
                if (contiguous) {
                    writer.ResolvePage(first / blocksPerPage, pages);
                    if constexpr (Writer::PRECOMPUTE_KSCALE) {
                        writer.Write(writer.pageBase, 128U, 0U, 0U, 1U);
                        writer.scale.template Record<BlockRows>(writer.pageBase.scale, 128U, 0U, 1U);
                        writer.scale.EndGroup(writer.keyRecords, writer.count, true);
                    } else {
                        writer.template Append<BlockRows>(0U, 128U, BlockRows);
                        writer.template Finish<BlockRows>();
                    }
                    return;
                }
            }
        }
        if constexpr (Writer::PRECOMPUTE_KSCALE) {
            writer.fullBlockCopy = writer.scale.template CalcFullBlockCopy<BlockRows>();
        }
        for (uint32_t i = 0; i < blocks; ++i) {
            const uint64_t token = static_cast<uint64_t>(indices[i]) * BlockRows;
            writer.ResolvePage(token / 128U, pages);
            writer.template Append<BlockRows>(token % 128U, BlockRows, BlockRows);
        }
        writer.template Finish<BlockRows>();
    }

    template <class Writer>
    __aicore__ inline uint32_t BuildStridedPagedGroup(const __gm__ int32_t* indices, const __gm__ int32_t* pages,
                                                      uint32_t start, uint32_t tokens, uint32_t explicitCount,
                                                      uint32_t residualId, Writer& writer)
    {
        const uint32_t first = start / blockY_;
        uint32_t inner = start % blockY_;
        const uint32_t blocks = tokens / blockY_;
        if (pageSize_ == 128U && tokenStride_ <= 512U && inner == 0U && blocks * blockY_ == tokens &&
            first + blocks <= explicitCount) {
            if (blockY_ == 4U) {
                BuildFullStridedPagedGroup<4U>(indices + first, pages, tokens / 4U, writer);
                return writer.count;
            }
            if (blockY_ == 8U) {
                BuildFullStridedPagedGroup<8U>(indices + first, pages, tokens / 8U, writer);
                return writer.count;
            }
            if (blockY_ == 16U) {
                BuildFullStridedPagedGroup<16U>(indices + first, pages, tokens / 16U, writer);
                return writer.count;
            }
        }
        uint32_t dealt = 0;
        for (uint32_t index = first; dealt < tokens; ++index) {
            const uint32_t rows = min(blockY_ - inner, tokens - dealt);
            uint32_t blockId;
            if constexpr (RESIDUAL_BLOCK_FLAG) {
                blockId = index == explicitCount ? residualId : indices[index];
            } else {
                blockId = indices[index];
            }
            const uint64_t source = static_cast<uint64_t>(blockId) * blockY_ + inner;
            uint64_t page = source / pageSize_;
            uint32_t pageInner = source % pageSize_;
            for (uint32_t remaining = rows; remaining != 0U; ++page, pageInner = 0U) {
                const uint32_t span = min(remaining, pageSize_ - pageInner);
                writer.ResolvePage(page, pages);
                writer.Append(pageInner, span, blockY_);
                remaining -= span;
            }
            dealt += rows;
            inner = 0;
        }
        writer.Finish();
        return writer.count;
    }

    template <bool SeparatePages = false, class Consumer>
    __aicore__ inline void operator()(AscendC::GlobalTensor<uint64_t> info,
                                      const __gm__ int32_t* __restrict__ sparseIdx,
                                      const __gm__ int32_t* __restrict__ blockTable, uint32_t tileIdx,
                                      uint32_t tileTokens, uint32_t explicitCount, uint32_t residualId,
                                      Consumer& consumer)
    {
        // Meta packs (sourceStride << 32) | (length << 1) | (ndNum - 1).
        // Compact records share [offset, meta]. Strided PA records use
        // [K offset, K meta, V offset, V meta] under the same group header.
        constexpr uint32_t recordWords = SeparatePages ? 4U : 2U;
        const uint32_t groupNum = (tileTokens + l1TileN_ - 1U) / l1TileN_;
        for (uint32_t groupIdx = AscendC::GetSubBlockIdx(); groupIdx < groupNum; groupIdx += 2U) {
            // Alternate across actual groups, including groups in successive tiles/tasks.
            // Only wait when reusing this half; the other half may still be copying.
            const auto eventId = static_cast<event_t>(EVENT_ID5 + recordsBufIdx_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
            auto records_ = recordsBuf_[recordsBufIdx_];
            // This UB region is private to V0. Events 5/6 protect each half
            // explicitly; direct stores avoid a conservative tensor-wide
            // auto-sync dependency on the other half's outstanding MTE3 copy.
            auto records = reinterpret_cast<__ubuf__ uint64_t*>(records_.GetPhyAddr());
            auto groupInfo = info[groupIdx * groupWords_];
            const uint32_t groupBegin = groupIdx * l1TileN_;
            const uint32_t groupTokens = min(l1TileN_, tileTokens - groupBegin);
            const uint32_t compactBegin = tileIdx * tileSize_ + groupBegin;
            uint32_t count;
            if constexpr (SeparatePages) {
                consumer.BeginGroup(groupIdx, groupTokens, records);
                count = BuildStridedPagedGroup(sparseIdx, blockTable, compactBegin, groupTokens, explicitCount,
                                               residualId, consumer);
            } else {
                consumer.BeginGroup(groupIdx, groupTokens);
                bool fullRecords = false;
                uint32_t inner = compactBegin % blockY_;
                count = 0;
                uint32_t dealtLenAccum = 0;
                uint32_t previousLen = 0;
                uint32_t previousPage = 0;
                uint32_t previousPageEnd = 0;
                uint64_t previousOffset = 0;
                if constexpr (!PAGED_CACHE_FLAG) {
                    const uint32_t firstBlock = compactBegin / blockY_;
                    const uint32_t blocks = groupTokens / blockY_;
                    if (tokenStride_ == tokenBytes_ && inner == 0 && blocks * blockY_ == groupTokens &&
                        firstBlock + blocks <= explicitCount) {
                        if (blockY_ == 4U) {
                            count = Gemm::Tile::BuildFullKvRecordsRuntime<4U>(
                                sparseIdx + firstBlock, records, groupTokens / 4U, 4U, consumer, tokenBytes_);
                        } else if (blockY_ == 8U) {
                            count = Gemm::Tile::BuildFullKvRecordsRuntime<8U>(
                                sparseIdx + firstBlock, records, groupTokens / 8U, 8U, consumer, tokenBytes_);
                        } else if (blockY_ == 16U) {
                            count = Gemm::Tile::BuildFullKvRecordsRuntime<16U>(
                                sparseIdx + firstBlock, records, groupTokens / 16U, 16U, consumer, tokenBytes_);
                        } else {
                            count = Gemm::Tile::BuildFullKvRecordsRuntime<>(sparseIdx + firstBlock, records, blocks,
                                                                            blockY_, consumer, tokenBytes_);
                        }
                        fullRecords = true;
                        dealtLenAccum = groupTokens;
                    } else if (tokenStride_ != tokenBytes_ && inner == 0U && blocks != 0U &&
                               blocks * blockY_ == groupTokens && firstBlock + blocks <= explicitCount &&
                               (blockY_ == 4U || blockY_ == 8U)) {
                        count = Gemm::Tile::BuildFullKvRecordsRuntime<>(sparseIdx + firstBlock, records, blocks,
                                                                        blockY_, consumer, tokenStride_);
                        dealtLenAccum = groupTokens;
                        fullRecords = true;
                    }
                }
                if (PAGED_CACHE_FLAG && (tokenBytes_ == 96U || tokenBytes_ == 128U)) {
                    const uint32_t firstBlock = compactBegin / blockY_;
                    const uint32_t blocks = groupTokens / blockY_;
                    if (pageSize_ == 128U && (blockY_ == 2U || blockY_ == 4U || blockY_ == 8U || blockY_ == 16U) &&
                        inner == 0U && blocks * blockY_ == groupTokens && firstBlock + blocks <= explicitCount) {
                        if (tokenStride_ / 16U <= 32U && blockY_ == 4U) {
                            count = Gemm::Tile::BuildFullPagedKvRecords<128U, 4U>(sparseIdx + firstBlock, blockTable,
                                                                                  records, groupTokens / 4U, 4U,
                                                                                  tokenStride_, kPageStride_, consumer);
                        } else if (tokenStride_ / 16U <= 32U && blockY_ == 8U) {
                            count = Gemm::Tile::BuildFullPagedKvRecords<128U, 8U>(sparseIdx + firstBlock, blockTable,
                                                                                  records, groupTokens / 8U, 8U,
                                                                                  tokenStride_, kPageStride_, consumer);
                        } else if (tokenStride_ / 16U <= 32U && blockY_ == 16U) {
                            count = Gemm::Tile::BuildFullPagedKvRecords<128U, 16U>(
                                sparseIdx + firstBlock, blockTable, records, groupTokens / 16U, 16U, tokenStride_,
                                kPageStride_, consumer);
                        } else {
                            count = Gemm::Tile::BuildFullPagedKvRecords<128U>(sparseIdx + firstBlock, blockTable,
                                                                              records, blocks, blockY_, tokenStride_,
                                                                              kPageStride_, consumer);
                        }
                        fullRecords = true;
                        dealtLenAccum = groupTokens;
                    }
                }
                for (uint32_t blockIdx = compactBegin / blockY_; dealtLenAccum < groupTokens; ++blockIdx) {
                    const uint32_t rows = min(blockY_ - inner, groupTokens - dealtLenAccum);
                    // These read-only inputs never alias the descriptor workspace.
                    int32_t blockId;
                    if constexpr (RESIDUAL_BLOCK_FLAG) {
                        blockId = blockIdx == explicitCount ? static_cast<int32_t>(residualId) : sparseIdx[blockIdx];
                    } else {
                        blockId = sparseIdx[blockIdx];
                    }
                    const uint64_t sourceToken = static_cast<uint64_t>(blockId) * blockY_ + inner;
                    // A page split produces ordinary entries; Cube never parses pages.
                    for (uint32_t copied = 0; copied < rows;) {
                        uint32_t len = rows - copied;
                        uint64_t token = sourceToken + copied;
                        uint64_t kOffset = 0;
                        uint32_t pageInner = 0;
                        uint32_t page = 0;
                        if constexpr (PAGED_CACHE_FLAG) {
                            pageInner = token % pageSize_;
                            page = blockTable[token / pageSize_];
                            len = min(len, pageSize_ - pageInner);
                            kOffset = static_cast<uint64_t>(page) * kPageStride_;
                            token = pageInner;
                        }
                        kOffset += token * tokenStride_;
                        const uint32_t word = 8U + count * recordWords;
                        if constexpr (PAGED_CACHE_FLAG) {
                            const bool reverse = kOffset < previousOffset;
                            const uint64_t kStride = reverse ? previousOffset - kOffset : kOffset - previousOffset;
                            if (previousLen != 0U && page == previousPage && pageInner == previousPageEnd) {
                                // Within one physical page both K and V use tokenStride_.
                                // Extend an unpaired run without changing valid token order.
                                previousLen += len;
                                records[word - 1U] = static_cast<uint64_t>(previousLen) << 1U;
                            } else if (previousLen == blockY_ && len == blockY_ && kStride <= 0xFFFFFFFFULL &&
                                       kStride % DispatchPolicy::COPY_ALIGNMENT == 0U) {
                                // Reorder K and V together, only within this group. Short page/tail
                                // fragments stay single, so valid tokens never move into padded columns.
                                if (reverse) {
                                    records[word - recordWords] = kOffset;
                                }
                                records[word - 1U] = (kStride << 32U) | (static_cast<uint64_t>(len) << 1U) | 1U;
                                consumer.Record(reverse ? kOffset : previousOffset, len, kStride, 2U);
                                previousLen = 0;
                            } else {
                                if (previousLen != 0U) {
                                    consumer.Record(previousOffset, previousLen, 0U, 1U);
                                }
                                records[word] = kOffset;
                                records[word + 1U] = static_cast<uint64_t>(len) << 1U;
                                ++count;
                                previousLen = len;
                                previousOffset = kOffset;
                            }
                            previousPage = page;
                            previousPageEnd = pageInner + len;
                        } else {
                            const uint64_t stride = kOffset - previousOffset;
                            if (previousLen != 0U && kOffset == previousOffset + previousLen * tokenStride_) {
                                // Adjacent blocks share the token stride and need only one copy.
                                previousLen += len;
                                records[word - 1U] = static_cast<uint64_t>(previousLen) << 1U;
                            } else if (previousLen == blockY_ && len == blockY_ && stride <= 0xFFFFFFFFULL &&
                                       stride % DispatchPolicy::COPY_ALIGNMENT == 0U) {
                                records[word - 1U] = (stride << 32U) | (static_cast<uint64_t>(len) << 1U) | 1U;
                                previousLen = 0;
                            } else {
                                records[word] = kOffset;
                                records[word + 1U] = static_cast<uint64_t>(len) << 1U;
                                ++count;
                                previousLen = len;
                                previousOffset = kOffset;
                            }
                        }
                        copied += len;
                    }
                    dealtLenAccum += rows;
                    inner = 0;
                }
                if constexpr (PAGED_CACHE_FLAG) {
                    if (previousLen != 0U) {
                        consumer.Record(previousOffset, previousLen, 0U, 1U);
                    }
                    fullRecords = true; // Every finalized K record has already issued its scale copy.
                }
                consumer.EndGroup(records, count, fullRecords);
            }
            records[0] = count;
            const uint32_t copyWords = (8U + count * recordWords + 3U) / 4U * 4U;
            if constexpr (PAGED_CACHE_FLAG) {
                if (tileSize_ <= 2U * l1TileN_ || (tileIdx == 0U && groupIdx < 2U)) {
                    AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
            AscendC::DataCopy(groupInfo, records_, copyWords);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
            recordsBufIdx_ ^= recordsBufMask_;
        }
    }

    // The most recently produced local group stays valid until this producer
    // reuses its slot. Consumers may read it while MTE3 publishes the same data.
    __aicore__ inline AscendC::LocalTensor<uint64_t> GetLocalRecords(uint32_t previous = 0) const
    {
        return recordsBuf_[(recordsBufIdx_ ^ recordsBufMask_) ^ (previous & recordsBufMask_)];
    }

private:
    AscendC::LocalTensor<uint64_t> recordsBuf_[2];
    uint32_t recordsBufIdx_ = 0;
    uint32_t recordsBufMask_;
    uint32_t tileSize_;
    uint32_t l1TileN_;
    uint32_t blockY_;
    uint32_t pageSize_;
    uint32_t groupWords_;
    uint64_t tokenStride_;
    uint64_t kPageStride_;
    uint32_t tokenBytes_;
};

} // namespace NpuArch::Epilogue::Block

#endif // GBSA_BLOCK_EPILOGUE_PREPARE_KV_OFFSETS_ARCH35_MXFP4_HPP

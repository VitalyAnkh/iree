// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"

namespace mlir::iree_compiler::IREE::GPU {

static std::optional<SmallVector<int64_t>> getIntegerVector(ArrayAttr array) {
  if (!array || !llvm::all_of(array.getValue(), llvm::IsaPred<IntegerAttr>)) {
    return std::nullopt;
  }
  return llvm::map_to_vector(array.getValue(), [](Attribute s) -> int64_t {
    return cast<IntegerAttr>(s).getInt();
  });
}

constexpr StringLiteral kMmaKindName = "mma_kind";

IREE::Codegen::InnerTileDescAttrInterface
getMmaKind(LoweringConfigAttr config) {
  return config.getAttributes()
      .getAs<IREE::Codegen::InnerTileDescAttrInterface>(kMmaKindName);
}

void setMmaKind(MLIRContext *context, SmallVectorImpl<NamedAttribute> &attrs,
                IREE::Codegen::InnerTileDescAttrInterface kind) {
  attrs.emplace_back(kMmaKindName, kind);
}

// TODO: Merge subgroup counts functionality into subgroup tiling level
//       lowering, when we have it implemented.
constexpr StringLiteral kSubgroupMCountName = "subgroup_m_count";
constexpr StringLiteral kSubgroupNCountName = "subgroup_n_count";

std::optional<int64_t> getSubgroupMCount(LoweringConfigAttr config) {
  auto subgroup_m_count_attr =
      config.getAttributes().getAs<IntegerAttr>(kSubgroupMCountName);
  if (!subgroup_m_count_attr) {
    return std::nullopt;
  }
  return subgroup_m_count_attr.getInt();
}

std::optional<int64_t> getSubgroupNCount(LoweringConfigAttr config) {
  auto subgroup_n_count_attr =
      config.getAttributes().getAs<IntegerAttr>(kSubgroupNCountName);
  if (!subgroup_n_count_attr) {
    return std::nullopt;
  }
  return subgroup_n_count_attr.getInt();
}

void setSubgroupMCount(MLIRContext *context,
                       SmallVectorImpl<NamedAttribute> &attrs,
                       int64_t subgroup_m_count) {
  attrs.emplace_back(
      kSubgroupMCountName,
      IntegerAttr::get(IntegerType::get(context, 64), subgroup_m_count));
}

void setSubgroupNCount(MLIRContext *context,
                       SmallVectorImpl<NamedAttribute> &attrs,
                       int64_t subgroup_n_count) {
  attrs.emplace_back(
      kSubgroupNCountName,
      IntegerAttr::get(IntegerType::get(context, 64), subgroup_n_count));
}

const StringLiteral kSubgroupBasisName = "subgroup_basis";
const StringLiteral kLaneBasisName = "lane_basis";

static StringLiteral getBasisLevelName(IREE::GPU::TilingLevel level) {
  switch (level) {
  case GPU::TilingLevel::Thread:
    // We use the term 'lane_basis' here in the context of thread distribution
    // because of the strict nesting of 'lane_basis' within 'subgroup_basis'.
    return kLaneBasisName;
  case GPU::TilingLevel::Subgroup:
    return kSubgroupBasisName;
  default:
    assert(false && "Unknown tiling level for distribution");
    return "";
  }
}

void setBasis(MLIRContext *context, SmallVector<NamedAttribute> &attrs,
              IREE::GPU::TilingLevel level, const Basis &basis) {
  Builder b(context);
  ArrayAttr basisAttr = b.getArrayAttr(
      {b.getI64ArrayAttr(basis.counts), b.getI64ArrayAttr(basis.mapping)});
  attrs.emplace_back(getBasisLevelName(level), basisAttr);
}

FailureOr<Basis> getBasis(IREE::GPU::LoweringConfigAttr config,
                          IREE::GPU::TilingLevel level) {
  auto basisAttr = dyn_cast_or_null<ArrayAttr>(
      config.getAttributes().get(getBasisLevelName(level)));
  if (!basisAttr) {
    return failure();
  }

  ArrayRef<Attribute> attrs = basisAttr.getValue();
  if (attrs.size() != 2) {
    return failure();
  }

  std::optional<SmallVector<int64_t>> maybeCounts =
      getIntegerVector(dyn_cast_or_null<ArrayAttr>(attrs[0]));
  std::optional<SmallVector<int64_t>> maybeMapping =
      getIntegerVector(dyn_cast_or_null<ArrayAttr>(attrs[1]));

  if (!maybeCounts.has_value() || !maybeMapping.has_value()) {
    return failure();
  }

  return Basis{maybeCounts.value(), maybeMapping.value()};
}

constexpr StringLiteral kPromoteOperandsName = "promote_operands";
constexpr StringLiteral kPromotionTypesName = "promotion_types";
std::optional<SmallVector<int64_t>>
getPromotedOperandList(LoweringConfigAttr config) {
  auto array = config.getAttributes().getAs<ArrayAttr>(kPromoteOperandsName);
  if (!array) {
    return std::nullopt;
  }
  return getIntegerVector(array);
}

std::optional<ArrayRef<Attribute>>
getPromotionTypesList(LoweringConfigAttr config) {
  auto array = config.getAttributes().getAs<ArrayAttr>(kPromotionTypesName);
  if (!array) {
    return std::nullopt;
  }
  return array.getValue();
}

void appendPromotedOperandsList(MLIRContext *context,
                                SmallVectorImpl<NamedAttribute> &attrs,
                                ArrayRef<int64_t> operands,
                                ArrayRef<Attribute> promotionTypes) {
  Builder b(context);
  attrs.emplace_back(kPromoteOperandsName, b.getI64ArrayAttr(operands));
  if (!promotionTypes.empty()) {
    assert(promotionTypes.size() == operands.size() &&
           "Promotion types size must match promoted operands size");
    attrs.emplace_back(kPromotionTypesName, b.getArrayAttr(promotionTypes));
  }
}
IREE::GPU::LoweringConfigAttr
setPromotedOperandsList(MLIRContext *context,
                        IREE::GPU::LoweringConfigAttr currAttr,
                        ArrayRef<int64_t> operands,
                        std::optional<ArrayRef<Attribute>> promotionTypes) {
  Builder b(context);
  DictionaryAttr currAttributes = currAttr.getAttributes();
  NamedAttrList attributes(currAttributes);
  std::optional<SmallVector<int64_t>> currPromotedOperandsList =
      getPromotedOperandList(currAttr);
  if (currPromotedOperandsList &&
      currPromotedOperandsList.value() == operands) {
    return currAttr;
  }

  Attribute newPromotedOperandsListAttr = b.getI64ArrayAttr(operands);

  attributes.set(kPromoteOperandsName, newPromotedOperandsListAttr);

  if (promotionTypes) {
    attributes.set(kPromotionTypesName, b.getArrayAttr(promotionTypes.value()));
  }
  return IREE::GPU::LoweringConfigAttr::get(context,
                                            attributes.getDictionary(context));
}

constexpr StringLiteral kPaddingName = "padding";
constexpr StringLiteral kPaddingConvName = "padding_conv";

std::optional<SmallVector<int64_t>> getPaddingList(LoweringConfigAttr config,
                                                   bool paddingConv) {
  auto attrName = paddingConv ? kPaddingConvName : kPaddingName;
  auto array = config.getAttributes().getAs<ArrayAttr>(attrName);
  if (!array) {
    return std::nullopt;
  }
  return getIntegerVector(array);
}

IREE::GPU::UKernelConfigAttr
getUkernelSpec(IREE::GPU::LoweringConfigAttr config) {
  return config.getAttributes().getAs<IREE::GPU::UKernelConfigAttr>("ukernel");
}

} // namespace mlir::iree_compiler::IREE::GPU

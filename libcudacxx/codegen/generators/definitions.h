//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <map>
#include <string>
#include <type_traits>

#include <fmt/format.h>

//----------------------------------------------------------------------------------------------------------------------
// PTX Mapped Memory I/O

enum class Mmio
{
  Disabled,
  Enabled,
};

inline std::string mmio(Mmio m)
{
  static const char* mmio_map[]{
    "",
    ".mmio",
  };
  return mmio_map[std::underlying_type_t<Mmio>(m)];
}

inline std::string mmio_tag(Mmio m)
{
  static const char* mmio_map[]{
    "__atomic_cuda_mmio_disable",
    "__atomic_cuda_mmio_enable",
  };
  return mmio_map[std::underlying_type_t<Mmio>(m)];
}

//----------------------------------------------------------------------------------------------------------------------
// PTX operand types

enum class Operand
{
  Floating,
  Unsigned,
  Signed,
  Bit,
};

inline std::string operand(Operand op)
{
  static std::map op_map = {
    std::pair{Operand::Floating, "f"},
    std::pair{Operand::Unsigned, "u"},
    std::pair{Operand::Signed, "s"},
    std::pair{Operand::Bit, "b"},
  };
  return op_map[op];
}

inline std::string operand_proxy_type(Operand op, size_t sz)
{
  if (op == Operand::Floating)
  {
    if (sz == 32)
    {
      return {"float"};
    }
    else
    {
      return {"double"};
    }
  }
  else if (op == Operand::Signed)
  {
    return fmt::format("int{}_t", sz);
  }
  // Binary and unsigned can be the same proxy_type
  return fmt::format("uint{}_t", sz);
}

//----------------------------------------------------------------------------------------------------------------------
// PTX binding constraints

inline std::string constraints(Operand op, size_t sz)
{
  static std::map constraint_map = {
    std::pair{32,
              std::map{
                std::pair{Operand::Bit, "r"},
                std::pair{Operand::Unsigned, "r"},
                std::pair{Operand::Signed, "r"},
                std::pair{Operand::Floating, "f"},
              }},
    std::pair{64,
              std::map{
                std::pair{Operand::Bit, "l"},
                std::pair{Operand::Unsigned, "l"},
                std::pair{Operand::Signed, "l"},
                std::pair{Operand::Floating, "d"},
              }},
    std::pair{128,
              std::map{
                std::pair{Operand::Bit, "l"},
                std::pair{Operand::Unsigned, "l"},
                std::pair{Operand::Signed, "l"},
                std::pair{Operand::Floating, "d"},
              }},
  };

  if (sz == 16)
  {
    return {"h"};
  }
  else
  {
    return constraint_map[sz][op];
  }
}

//----------------------------------------------------------------------------------------------------------------------
// PTX Memory Semantics

enum class Semantic
{
  Weak,
  Relaxed,
  Release,
  Acquire,
  Acq_Rel,
  Seq_Cst,
  Volatile,
};

inline std::string semantic(Semantic sem)
{
  static std::map sem_map = {
    std::pair{Semantic::Weak, ".weak"},
    std::pair{Semantic::Relaxed, ".relaxed"},
    std::pair{Semantic::Release, ".release"},
    std::pair{Semantic::Acquire, ".acquire"},
    std::pair{Semantic::Acq_Rel, ".acq_rel"},
    std::pair{Semantic::Seq_Cst, ".sc"},
    std::pair{Semantic::Volatile, ""},
  };
  return sem_map[sem];
}

inline std::string semantic_tag(Semantic sem)
{
  static std::map sem_map = {
    std::pair{Semantic::Weak, "__cuda_weak"},
    std::pair{Semantic::Relaxed, "__atomic_cuda_relaxed"},
    std::pair{Semantic::Release, "__atomic_cuda_release"},
    std::pair{Semantic::Acquire, "__atomic_cuda_acquire"},
    std::pair{Semantic::Acq_Rel, "__atomic_cuda_acq_rel"},
    std::pair{Semantic::Seq_Cst, "__atomic_cuda_seq_cst"},
    std::pair{Semantic::Volatile, "__atomic_cuda_volatile"},
  };
  return sem_map[sem];
}

//----------------------------------------------------------------------------------------------------------------------
// PTX Thread Hierarchy Scopes

enum class Scope
{
  Thread,
  Warp,
  CTA,
  Cluster,
  GPU,
  System,
};

inline std::string scope(Scope sco)
{
  static std::map sco_map = {
    std::pair{Scope::Thread, ""},
    std::pair{Scope::Warp, ""},
    std::pair{Scope::CTA, ".cta"},
    std::pair{Scope::Cluster, ".cluster"},
    std::pair{Scope::GPU, ".gpu"},
    std::pair{Scope::System, ".sys"},
  };
  return sco_map[sco];
}

inline std::string scope_tag(Scope sco)
{
  static std::map sco_map = {
    std::pair{Scope::Thread, "__thread_scope_thread_tag"},
    std::pair{Scope::Warp, ""},
    std::pair{Scope::CTA, "__thread_scope_block_tag"},
    std::pair{Scope::Cluster, "__thread_scope_cluster_tag"},
    std::pair{Scope::GPU, "__thread_scope_device_tag"},
    std::pair{Scope::System, "__thread_scope_system_tag"},
  };
  return sco_map[sco];
}

//----------------------------------------------------------------------------------------------------------------------
// PTX Memory Space Scopes

enum class Space
{
  Global,
  Cluster,
  Shared,
};

inline std::string space(Space space)
{
  static std::map map = {std::pair{Space::Global, ".global"},
                         std::pair{Space::Cluster, ".shared::cluster"},
                         std::pair{Space::Shared, ".shared::cta"}};
  return map[space];
}

//----------------------------------------------------------------------------------------------------------------------
// PTX Cache Operators

enum class LoadCacheOperator
{
  None,
  CacheAll,
  CacheGlobal,
  CacheStreaming,
  LastUse,
  Volatile
};

inline std::string cache_operator(LoadCacheOperator space)
{
  static std::map map = {
    std::pair{LoadCacheOperator::None, ""},
    std::pair{LoadCacheOperator::CacheAll, ".ca"},
    std::pair{LoadCacheOperator::CacheGlobal, ".cg"},
    std::pair{LoadCacheOperator::CacheStreaming, ".cs"},
    std::pair{LoadCacheOperator::LastUse, ".lu"},
    std::pair{LoadCacheOperator::Volatile, ".cv"}};
  return map[space];
}

enum class StoreCacheOperator
{
  None,
  CacheWriteBack,
  CacheGlobal,
  CacheStreaming,
  CacheWriteThrough
};

inline std::string cache_operator(StoreCacheOperator space)
{
  static std::map map = {
    std::pair{StoreCacheOperator::None, ""},
    std::pair{StoreCacheOperator::CacheWriteBack, ".wb"},
    std::pair{StoreCacheOperator::CacheGlobal, ".cg"},
    std::pair{StoreCacheOperator::CacheStreaming, ".cs"},
    std::pair{StoreCacheOperator::CacheWriteThrough, ".wt"},
  };
  return map[space];
}

//----------------------------------------------------------------------------------------------------------------------
// PTX Eviction Priority

enum class EvictionPriority
{
  None,
  Normal,
  Unchanged,
  First,
  Last,
  NoAllocate
};

inline std::string eviction_priority(EvictionPriority space)
{
  static std::map map = {
    std::pair{EvictionPriority::None, ""},
    std::pair{EvictionPriority::Normal, ".L1::evict_normal"},
    std::pair{EvictionPriority::Unchanged, ".L1::evict_unchanged"},
    std::pair{EvictionPriority::First, ".L1::evict_first"},
    std::pair{EvictionPriority::Last, ".L1::evict_last"},
    std::pair{EvictionPriority::NoAllocate, ".L1::no_allocate"}};
  return map[space];
}

//----------------------------------------------------------------------------------------------------------------------
// PTX Prefetch Size

enum class PrefetchSize
{
  None,
  L2_64B,
  L2_128B,
  L2_256B
};

inline std::string prefetch_size(PrefetchSize space)
{
  static std::map map = {
    std::pair{PrefetchSize::None, ""},
    std::pair{PrefetchSize::L2_64B, ".L2::64B"},
    std::pair{PrefetchSize::L2_128B, ".L2::128B"},
    std::pair{PrefetchSize::L2_256B, ".L2::256B"}};
  return map[space];
}

#endif // DEFINITIONS_H

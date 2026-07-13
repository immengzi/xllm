/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/request/request.h"

namespace xllm {

enum class ShortRequestFirstDispatch : int8_t {
  IMMEDIATE = 0,
  SHORT = 1,
  LONG = 2,
  AGED_LONG = 3,
};

class PdPrefillShortRequestFirstQueue final {
 public:
  using NowMsFn = std::function<int64_t()>;

  PdPrefillShortRequestFirstQueue(int32_t threshold,
                                  double long_max_wait_ms,
                                  NowMsFn now_ms = nullptr);

  void push(const std::shared_ptr<Request>& request);
  void push_front(const std::shared_ptr<Request>& request);
  void requeue_deferred(
      const std::vector<std::shared_ptr<Request>>& deferred_requests);
  void erase_request_tracking(const std::shared_ptr<Request>& request);

  std::shared_ptr<Request> pop_next(ShortRequestFirstDispatch* dispatch);

  std::vector<std::shared_ptr<Request>> snapshot() const;

  bool empty() const;
  size_t size() const;

  size_t immediate_size() const;
  size_t short_size() const;
  size_t long_size() const;

 private:
  enum class Lane : int8_t {
    IMMEDIATE = 0,
    SHORT = 1,
    LONG = 2,
  };

  Lane classify_request(const std::shared_ptr<Request>& request) const;
  ShortRequestFirstDispatch select_dispatch() const;
  bool should_promote_aged_long() const;
  void push_impl(const std::shared_ptr<Request>& request, bool push_front);

  int32_t threshold_;
  double long_max_wait_ms_;
  NowMsFn now_ms_;
  std::deque<std::shared_ptr<Request>> immediate_queue_;
  std::deque<std::shared_ptr<Request>> short_queue_;
  std::deque<std::shared_ptr<Request>> long_queue_;
  std::unordered_map<std::string, int64_t> long_enqueue_time_ms_;
};

}  // namespace xllm

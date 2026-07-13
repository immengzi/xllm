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

#include "scheduler/pd_prefill_short_request_first_queue.h"

#include <chrono>

#include "glog/logging.h"

namespace xllm {
namespace {

int64_t default_now_ms() {
  const std::chrono::steady_clock::time_point now =
      std::chrono::steady_clock::now();
  const std::chrono::milliseconds elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          now.time_since_epoch());
  return elapsed_ms.count();
}

}  // namespace

PdPrefillShortRequestFirstQueue::PdPrefillShortRequestFirstQueue(
    int32_t threshold,
    double long_max_wait_ms,
    NowMsFn now_ms)
    : threshold_(threshold),
      long_max_wait_ms_(long_max_wait_ms),
      now_ms_(now_ms) {
  if (!now_ms_) {
    now_ms_ = default_now_ms;
  }
}

void PdPrefillShortRequestFirstQueue::push(
    const std::shared_ptr<Request>& request) {
  push_impl(request, false);
}

void PdPrefillShortRequestFirstQueue::push_front(
    const std::shared_ptr<Request>& request) {
  push_impl(request, true);
}

void PdPrefillShortRequestFirstQueue::requeue_deferred(
    const std::vector<std::shared_ptr<Request>>& deferred_requests) {
  for (auto it = deferred_requests.rbegin(); it != deferred_requests.rend();
       ++it) {
    push_front(*it);
  }
}

void PdPrefillShortRequestFirstQueue::erase_request_tracking(
    const std::shared_ptr<Request>& request) {
  if (request == nullptr) {
    return;
  }
  long_enqueue_time_ms_.erase(request->request_id());
}

std::shared_ptr<Request> PdPrefillShortRequestFirstQueue::pop_next(
    ShortRequestFirstDispatch* dispatch) {
  CHECK(!empty()) << "ShortRequestFirst queue is empty";
  const ShortRequestFirstDispatch selected_dispatch = select_dispatch();
  if (dispatch != nullptr) {
    *dispatch = selected_dispatch;
  }

  if (selected_dispatch == ShortRequestFirstDispatch::IMMEDIATE) {
    std::shared_ptr<Request> request = immediate_queue_.front();
    immediate_queue_.pop_front();
    return request;
  }
  if (selected_dispatch == ShortRequestFirstDispatch::SHORT) {
    std::shared_ptr<Request> request = short_queue_.front();
    short_queue_.pop_front();
    return request;
  }

  std::shared_ptr<Request> request = long_queue_.front();
  long_queue_.pop_front();
  return request;
}

std::vector<std::shared_ptr<Request>>
PdPrefillShortRequestFirstQueue::snapshot() const {
  PdPrefillShortRequestFirstQueue copied(*this);
  std::vector<std::shared_ptr<Request>> requests;
  requests.reserve(copied.size());
  while (!copied.empty()) {
    requests.emplace_back(copied.pop_next(/*dispatch=*/nullptr));
  }
  return requests;
}

bool PdPrefillShortRequestFirstQueue::empty() const {
  return immediate_queue_.empty() && short_queue_.empty() &&
         long_queue_.empty();
}

size_t PdPrefillShortRequestFirstQueue::size() const {
  return immediate_queue_.size() + short_queue_.size() + long_queue_.size();
}

size_t PdPrefillShortRequestFirstQueue::immediate_size() const {
  return immediate_queue_.size();
}

size_t PdPrefillShortRequestFirstQueue::short_size() const {
  return short_queue_.size();
}

size_t PdPrefillShortRequestFirstQueue::long_size() const {
  return long_queue_.size();
}

PdPrefillShortRequestFirstQueue::Lane
PdPrefillShortRequestFirstQueue::classify_request(
    const std::shared_ptr<Request>& request) const {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());
  Sequence* sequence = request->sequences()[0].get();
  CHECK(sequence != nullptr);

  if (request->preempted() ||
      sequence->kv_state().num_blocks(BlockType::KV) > 0 ||
      sequence->kv_cache_tokens_num() > 0) {
    return Lane::IMMEDIATE;
  }
  if (sequence->num_prompt_tokens() <= static_cast<size_t>(threshold_)) {
    return Lane::SHORT;
  }
  return Lane::LONG;
}

ShortRequestFirstDispatch PdPrefillShortRequestFirstQueue::select_dispatch()
    const {
  if (!immediate_queue_.empty()) {
    return ShortRequestFirstDispatch::IMMEDIATE;
  }
  if (should_promote_aged_long()) {
    return ShortRequestFirstDispatch::AGED_LONG;
  }
  if (!short_queue_.empty()) {
    return ShortRequestFirstDispatch::SHORT;
  }
  return ShortRequestFirstDispatch::LONG;
}

bool PdPrefillShortRequestFirstQueue::should_promote_aged_long() const {
  if (long_max_wait_ms_ <= 0.0 || short_queue_.empty() || long_queue_.empty()) {
    return false;
  }
  const std::shared_ptr<Request>& head = long_queue_.front();
  const auto it = long_enqueue_time_ms_.find(head->request_id());
  if (it == long_enqueue_time_ms_.end()) {
    return false;
  }
  const int64_t waited_ms = now_ms_() - it->second;
  return static_cast<double>(waited_ms) >= long_max_wait_ms_;
}

void PdPrefillShortRequestFirstQueue::push_impl(
    const std::shared_ptr<Request>& request,
    bool push_front) {
  const Lane lane = classify_request(request);
  if (lane == Lane::IMMEDIATE) {
    if (push_front) {
      immediate_queue_.push_front(request);
    } else {
      immediate_queue_.push_back(request);
    }
    return;
  }

  if (lane == Lane::SHORT) {
    if (push_front) {
      short_queue_.push_front(request);
    } else {
      short_queue_.push_back(request);
    }
    return;
  }

  if (push_front) {
    long_queue_.push_front(request);
  } else {
    long_queue_.push_back(request);
  }
  long_enqueue_time_ms_.emplace(request->request_id(), now_ms_());
}

}  // namespace xllm

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

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "framework/request/request.h"
#include "framework/request/request_state.h"

namespace xllm {
namespace {

std::shared_ptr<Request> make_request(const std::string& request_id,
                                      size_t prompt_tokens) {
  std::vector<int32_t> prompt_token_ids(prompt_tokens, 1);
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  stopping_checker.set_max_context_len(4096);
  stopping_checker.set_ignore_eos(true);

  RequestState state("prompt",
                     prompt_token_ids,
                     sampling_param,
                     scheduler_param,
                     stopping_checker,
                     prompt_token_ids.size() + 8,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*logprobs=*/false,
                     /*skip_special_tokens=*/false,
                     /*include_usage=*/false,
                     /*mm_data=*/nullptr,
                     /*service_request_id=*/nullptr);
  return std::make_shared<Request>(
      request_id, "x-request-id", "x-request-time", state, "service-request");
}

TEST(PdPrefillShortRequestFirstQueueTest, SplitsFreshRequestsByThreshold) {
  PdPrefillShortRequestFirstQueue queue(
      /*threshold=*/256, /*long_max_wait_ms=*/0.0);

  queue.push(make_request("short", /*prompt_tokens=*/64));
  queue.push(make_request("long", /*prompt_tokens=*/512));

  EXPECT_EQ(queue.immediate_size(), 0u);
  EXPECT_EQ(queue.short_size(), 1u);
  EXPECT_EQ(queue.long_size(), 1u);
}

TEST(PdPrefillShortRequestFirstQueueTest, RoutesPreemptedToImmediateLane) {
  PdPrefillShortRequestFirstQueue queue(
      /*threshold=*/256, /*long_max_wait_ms=*/0.0);

  std::shared_ptr<Request> request =
      make_request("preempted", /*prompt_tokens=*/1024);
  request->set_preempted();
  queue.push(request);

  EXPECT_EQ(queue.immediate_size(), 1u);
  EXPECT_EQ(queue.short_size(), 0u);
  EXPECT_EQ(queue.long_size(), 0u);
}

TEST(PdPrefillShortRequestFirstQueueTest, RoutesHeldKvRequestsToImmediateLane) {
  PdPrefillShortRequestFirstQueue queue(
      /*threshold=*/256, /*long_max_wait_ms=*/0.0);

  std::shared_ptr<Request> request =
      make_request("held-kv", /*prompt_tokens=*/1024);
  request->sequences()[0]->kv_state().set_kv_cache_tokens_num(1);
  queue.push(request);

  EXPECT_EQ(queue.immediate_size(), 1u);
  EXPECT_EQ(queue.short_size(), 0u);
  EXPECT_EQ(queue.long_size(), 0u);
}

TEST(PdPrefillShortRequestFirstQueueTest,
     DispatchesImmediateThenShortThenLong) {
  PdPrefillShortRequestFirstQueue queue(
      /*threshold=*/256, /*long_max_wait_ms=*/0.0);

  std::shared_ptr<Request> immediate =
      make_request("immediate", /*prompt_tokens=*/512);
  immediate->set_preempted();
  queue.push(make_request("long", /*prompt_tokens=*/512));
  queue.push(make_request("short", /*prompt_tokens=*/64));
  queue.push(immediate);

  ShortRequestFirstDispatch dispatch = ShortRequestFirstDispatch::LONG;
  EXPECT_EQ(queue.pop_next(&dispatch)->request_id(), "immediate");
  EXPECT_EQ(dispatch, ShortRequestFirstDispatch::IMMEDIATE);
  EXPECT_EQ(queue.pop_next(&dispatch)->request_id(), "short");
  EXPECT_EQ(dispatch, ShortRequestFirstDispatch::SHORT);
  EXPECT_EQ(queue.pop_next(&dispatch)->request_id(), "long");
  EXPECT_EQ(dispatch, ShortRequestFirstDispatch::LONG);
}

TEST(PdPrefillShortRequestFirstQueueTest, LongDoesNotBypassShortWhenAgingOff) {
  int64_t now_ms = 0;
  PdPrefillShortRequestFirstQueue queue(
      /*threshold=*/256,
      /*long_max_wait_ms=*/0.0,
      [&now_ms]() -> int64_t { return now_ms; });

  queue.push(make_request("long", /*prompt_tokens=*/512));
  queue.push(make_request("short", /*prompt_tokens=*/64));
  now_ms = 10000;

  ShortRequestFirstDispatch dispatch = ShortRequestFirstDispatch::LONG;
  EXPECT_EQ(queue.pop_next(&dispatch)->request_id(), "short");
  EXPECT_EQ(dispatch, ShortRequestFirstDispatch::SHORT);
}

TEST(PdPrefillShortRequestFirstQueueTest, LongAtWaitLimitCanBypassShort) {
  int64_t now_ms = 0;
  PdPrefillShortRequestFirstQueue queue(
      /*threshold=*/256,
      /*long_max_wait_ms=*/100.0,
      [&now_ms]() -> int64_t { return now_ms; });

  queue.push(make_request("long", /*prompt_tokens=*/512));
  queue.push(make_request("short", /*prompt_tokens=*/64));
  now_ms = 100;

  ShortRequestFirstDispatch dispatch = ShortRequestFirstDispatch::LONG;
  EXPECT_EQ(queue.pop_next(&dispatch)->request_id(), "long");
  EXPECT_EQ(dispatch, ShortRequestFirstDispatch::AGED_LONG);
}

TEST(PdPrefillShortRequestFirstQueueTest, AgingOnlyPromotesTheCurrentLongHead) {
  int64_t now_ms = 0;
  PdPrefillShortRequestFirstQueue queue(
      /*threshold=*/256,
      /*long_max_wait_ms=*/100.0,
      [&now_ms]() -> int64_t { return now_ms; });

  queue.push(make_request("long-0", /*prompt_tokens=*/512));
  now_ms = 50;
  queue.push(make_request("long-1", /*prompt_tokens=*/768));
  queue.push(make_request("short", /*prompt_tokens=*/64));
  now_ms = 149;

  ShortRequestFirstDispatch dispatch = ShortRequestFirstDispatch::LONG;
  EXPECT_EQ(queue.pop_next(&dispatch)->request_id(), "long-0");
  EXPECT_EQ(dispatch, ShortRequestFirstDispatch::AGED_LONG);
  EXPECT_EQ(queue.pop_next(&dispatch)->request_id(), "short");
  EXPECT_EQ(dispatch, ShortRequestFirstDispatch::SHORT);
  EXPECT_EQ(queue.pop_next(&dispatch)->request_id(), "long-1");
  EXPECT_EQ(dispatch, ShortRequestFirstDispatch::LONG);
}

TEST(PdPrefillShortRequestFirstQueueTest, RequeueKeepsOriginalClassification) {
  PdPrefillShortRequestFirstQueue queue(
      /*threshold=*/256, /*long_max_wait_ms=*/0.0);

  std::shared_ptr<Request> request =
      make_request("long", /*prompt_tokens=*/512);
  queue.push(request);

  ShortRequestFirstDispatch dispatch = ShortRequestFirstDispatch::LONG;
  std::shared_ptr<Request> popped = queue.pop_next(&dispatch);
  ASSERT_EQ(popped->request_id(), "long");
  queue.requeue_deferred({popped});

  EXPECT_EQ(queue.immediate_size(), 0u);
  EXPECT_EQ(queue.short_size(), 0u);
  EXPECT_EQ(queue.long_size(), 1u);
}

}  // namespace
}  // namespace xllm

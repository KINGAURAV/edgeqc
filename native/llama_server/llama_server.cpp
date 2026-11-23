/*
 * EdgeQC llama_server: a resident ExecuTorch Llama runner for the advisory pipeline.
 *
 * Same runner as ExecuTorch's examples/models/llama/main.cpp (llama_main), but the .pte and tokenizer
 * are loaded ONCE. Prompts then arrive on stdin, so each advice call skips the model load.
 *
 * Protocol (stdin → stdout, binary-safe, one request at a time):
 *   request : "GEN <prompt_bytes> <max_new_tokens> <temperature> [stream] [segment_bytes]\n" + <prompt_bytes> raw bytes
 *   response: "OK <reply_bytes> <prompt_tokens> <generated_tokens> <ttft_ms> <total_ms> <cached_tokens>\n"
 *             + reply bytes, or "ERR <message>\n"
 *             With stream=1, each generated piece is sent first as "TOK <bytes>\n" + bytes (for live display);
 *             the final OK frame still carries the full reply.
 *   request : "PREFIX <bytes>\n" + bytes   (optional) registers a static prompt prefix → "OK\n"
 *   "QUIT\n" exits. The server prints "READY <load_ms>\n" once the model is loaded.
 *
 * Prefix cache: the advice prompt starts with a fixed system prompt + worked example (~75% of its tokens).
 * When a GEN prompt starts with the registered prefix, the prefix's KV cache (prefilled once) is reused:
 * the runner is rewound to the end of the prefix and only the rest is prefilled. On a Pi 4 (~10 tok/s
 * prefill) this removes most of the time to first token. Any other prompt does a full reset and
 * invalidates the cached prefix, which is then re-prefilled on the next matching request.
 *
 * Second cache level (segment): GEN may give segment_bytes, the length of a longer reusable prompt head that starts
 * with the prefix (the advisor sends prefix + the defect class's reference notes). The server keeps the KV position
 * after the last segment; a request with the same segment rewinds there and prefills only the per-frame findings.
 * A different segment rewinds to the prefix, prefills the new segment and keeps it. Consecutive defects of the same
 * class (the usual case on a line) then prefill ~40-60 tokens instead of ~120-200.
 * Needs the two TextLLMRunner hooks edgeqc_pos()/edgeqc_rewind() that scripts/setup_executorch.sh adds.
 * The runner's own token echo and logs are sent to /dev/null and stderr; replies use a duplicated stdout fd.
 *
 * Built by scripts/setup_executorch.sh (it adds a target next to llama_main in the ExecuTorch tree).
 */

#include <executorch/examples/models/llama/runner/runner.h>
#include <gflags/gflags.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <unistd.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(model_path, "", "Llama .pte");
DEFINE_string(tokenizer_path, "", "tokenizer.model");
DEFINE_int32(cpu_threads, -1, "threads (-1 = number of performance cores)");
DEFINE_int32(num_bos, 1, "BOS tokens prepended to each prompt");

namespace llm = ::executorch::extension::llm;

static long now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Keep a private handle to the real stdout for protocol replies; silence everything else on stdout.
  FILE* out = fdopen(dup(fileno(stdout)), "w");
  if (!freopen("/dev/null", "w", stdout)) {
    fprintf(out, "ERR cannot redirect stdout\n");
    return 1;
  }

#if defined(ET_USE_THREADPOOL)
  uint32_t threads = FLAGS_cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(FLAGS_cpu_threads);
  if (threads > 0) {
    ::executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(threads);
  }
#endif

  long t0 = now_ms();
  std::unique_ptr<llm::TextLLMRunner> runner =
      example::create_llama_runner(FLAGS_model_path, FLAGS_tokenizer_path, std::vector<std::string>{});
  if (runner == nullptr || runner->load() != executorch::runtime::Error::Ok) {
    fprintf(out, "ERR failed to load model\n");
    fflush(out);
    return 1;
  }
  fprintf(out, "READY %ld\n", now_ms() - t0);
  fflush(out);

  std::string prefix;           // registered static prompt prefix
  int64_t prefix_pos = -1;      // KV-cache position after the prefix, -1 = not (or no longer) cached
  std::string seg_text;         // last cached segment (prefix + class notes)
  int64_t seg_pos = -1;         // KV-cache position after it, -1 = not cached

  std::string header;
  while (std::getline(std::cin, header)) {
    if (header == "QUIT") break;
    size_t plen = 0;
    if (sscanf(header.c_str(), "PREFIX %zu", &plen) == 1) {
      prefix.assign(plen, '\0');
      std::cin.read(&prefix[0], static_cast<std::streamsize>(plen));
      prefix_pos = -1;          // prefilled lazily on the first matching request
      seg_pos = -1;
      fprintf(out, "OK\n");
      fflush(out);
      continue;
    }
    size_t nbytes = 0;
    int max_new = 140;
    float temperature = 0.f;
    int stream = 0;
    size_t seg_len = 0;
    if (sscanf(header.c_str(), "GEN %zu %d %f %d %zu", &nbytes, &max_new, &temperature, &stream, &seg_len) < 3) {
      fprintf(out, "ERR bad header\n");
      fflush(out);
      continue;
    }
    std::string prompt(nbytes, '\0');
    std::cin.read(&prompt[0], static_cast<std::streamsize>(nbytes));

    llm::GenerationConfig config{};
    config.echo = false;
    config.max_new_tokens = max_new;
    config.temperature = temperature;
    config.num_bos = FLAGS_num_bos;
    config.num_eos = 0;

    std::string reply;
    long long n_prompt = 0, n_gen = 0;
    long ttft = -1;
    long start = now_ms();
    std::string rest = prompt;
    long long cached = 0;
    bool failed = false;
    bool use_prefix = !prefix.empty() && prompt.size() > prefix.size() &&
                      prompt.compare(0, prefix.size(), prefix) == 0;
    if (use_prefix) {
      if (prefix_pos < 0) {     // (re)build the prefix KV cache
        runner->reset();
        auto r = runner->prefill(prefix, FLAGS_num_bos, 0);
        if (!r.ok()) {
          fprintf(out, "ERR prefix prefill failed (%d)\n", static_cast<int>(r.error()));
          fflush(out);
          continue;
        }
        prefix_pos = runner->edgeqc_pos();
        seg_pos = -1;
      } else {
        cached = prefix_pos;    // reused, nothing to prefill for the prefix
      }
      config.num_bos = 0;       // BOS is already in the cached prefix
      bool use_seg = seg_len > prefix.size() && seg_len < prompt.size();
      if (use_seg && seg_pos >= 0 && seg_text.size() == seg_len && prompt.compare(0, seg_len, seg_text) == 0) {
        runner->edgeqc_rewind(seg_pos);    // same class notes as the cached segment: reuse them too
        cached = seg_pos;
        rest = prompt.substr(seg_len);
      } else {
        runner->edgeqc_rewind(prefix_pos); // drop the previous request's tokens, keep the prefix
        rest = prompt.substr(prefix.size());
        if (use_seg) {                     // prefill and keep the new segment
          auto r = runner->prefill(prompt.substr(prefix.size(), seg_len - prefix.size()), 0, 0);
          if (r.ok()) {
            seg_text = prompt.substr(0, seg_len);
            seg_pos = runner->edgeqc_pos();
            rest = prompt.substr(seg_len);
          } else {
            seg_pos = -1;
            failed = true;
          }
        }
      }
    } else {
      runner->reset();          // fresh KV cache; overwrites the cached prefix
      prefix_pos = -1;
      seg_pos = -1;
    }
    if (failed) {
      prefix_pos = -1;
      fprintf(out, "ERR segment prefill failed\n");
      fflush(out);
      continue;
    }
    const int64_t base_pos = use_prefix ? runner->edgeqc_pos() : 0;   // tokens before `rest`
    auto err = runner->generate(
        rest, config,
        [&](const std::string& piece) {
          if (ttft < 0) ttft = now_ms() - start;   // includes a (re)built prefix prefill, if any
          reply += piece;
          if (stream) {
            fprintf(out, "TOK %zu\n", piece.size());
            fwrite(piece.data(), 1, piece.size(), out);
            fflush(out);
          }
        },
        [&](const llm::Stats& s) {  // Stats isn't copyable; keep the numbers we report
          n_prompt = s.num_prompt_tokens;
          n_gen = s.num_generated_tokens;
        });
    long total = now_ms() - start;
    if (err != executorch::runtime::Error::Ok) {
      prefix_pos = -1;
      seg_pos = -1;
      fprintf(out, "ERR generate failed (%d)\n", static_cast<int>(err));
      fflush(out);
      continue;
    }
    fprintf(out, "OK %zu %lld %lld %ld %ld %lld\n", reply.size(), n_prompt + base_pos, n_gen, ttft, total, cached);
    fwrite(reply.data(), 1, reply.size(), out);
    fflush(out);
  }
  return 0;
}

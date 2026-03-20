#include "text_decoder.h"
#include <cstdio>
#include <vector>
#include <algorithm>

int main() {
    printf("=== Test Decoder Last Position (5 tokens) ===\n\n");
    
    qwen3_asr::TextDecoder decoder;
    if (!decoder.load_model("models/qwen3-asr-0.6b-f16.gguf")) {
        fprintf(stderr, "Failed to load model: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    if (!decoder.init_kv_cache(512)) {
        fprintf(stderr, "Failed to init KV cache: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    std::vector<int32_t> tokens = {151669, 151676, 151676, 151676, 151670};
    printf("Token sequence length: %zu\n", tokens.size());
    
    std::vector<float> logits;
    if (!decoder.forward(tokens.data(), tokens.size(), 0, logits)) {
        fprintf(stderr, "Forward pass failed: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    const auto & cfg = decoder.get_config();
    int32_t vocab_size = cfg.vocab_size;
    int32_t n_tokens = tokens.size();
    
    // Check LAST position
    const float * last_logits = logits.data() + (n_tokens - 1) * vocab_size;
    
    std::vector<std::pair<float, int32_t>> top;
    for (int32_t i = 0; i < vocab_size; ++i) {
        top.push_back({last_logits[i], i});
    }
    std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    printf("\nLAST position top 5 logits:\n");
    for (int i = 0; i < 5; ++i) {
        printf("  [%d] token=%d logit=%f\n", i, top[i].second, top[i].first);
    }
    
    printf("\nLast position first 10 logits:\n");
    for (int i = 0; i < 10; ++i) {
        printf("  logits[%d] = %f\n", i, last_logits[i]);
    }
    
    // Also check position 0
    const float * pos0_logits = logits.data();
    int32_t argmax0 = 0;
    float max_val0 = pos0_logits[0];
    for (int32_t i = 1; i < vocab_size; ++i) {
        if (pos0_logits[i] > max_val0) {
            max_val0 = pos0_logits[i];
            argmax0 = i;
        }
    }
    printf("\nPosition 0 argmax: %d (logit=%f)\n", argmax0, max_val0);
    
    // Expected: Last position argmax should be 198 (newline)
    if (top[0].second == 198) {
        printf("\nTEST PASSED: Last position argmax is 198 (newline)\n");
    } else {
        printf("\nTEST RESULT: Last position argmax is %d (expected 198)\n", top[0].second);
    }
    
    return 0;
}

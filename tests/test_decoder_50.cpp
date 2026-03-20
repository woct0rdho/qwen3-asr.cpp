#include "text_decoder.h"
#include <cstdio>
#include <vector>
#include <algorithm>

int main() {
    printf("=== Test Decoder 50 tokens ===\n\n");
    
    qwen3_asr::TextDecoder decoder;
    if (!decoder.load_model("models/qwen3-asr-0.6b-f16.gguf")) {
        fprintf(stderr, "Failed to load model: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    if (!decoder.init_kv_cache(512)) {
        fprintf(stderr, "Failed to init KV cache: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    // 50 tokens: audio_start + 48 audio_pad + audio_end
    std::vector<int32_t> tokens;
    tokens.push_back(151669);  // audio_start
    for (int i = 0; i < 48; ++i) {
        tokens.push_back(151676);  // audio_pad
    }
    tokens.push_back(151670);  // audio_end
    
    printf("Token sequence length: %zu\n", tokens.size());
    
    std::vector<float> logits;
    if (!decoder.forward(tokens.data(), tokens.size(), 0, logits)) {
        fprintf(stderr, "Forward pass failed: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    const auto & cfg = decoder.get_config();
    int32_t vocab_size = cfg.vocab_size;
    int32_t n_tokens = tokens.size();
    
    const float * last_logits = logits.data() + (n_tokens - 1) * vocab_size;
    
    std::vector<std::pair<float, int32_t>> top;
    for (int32_t i = 0; i < vocab_size; ++i) {
        top.push_back({last_logits[i], i});
    }
    std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    printf("\nC++ - Last position top 5 logits:\n");
    for (int i = 0; i < 5; ++i) {
        printf("  [%d] token=%d logit=%f\n", i, top[i].second, top[i].first);
    }
    
    // Expected: token 659 should be top
    if (top[0].second == 659) {
        printf("\nTEST PASSED: Top token is 659\n");
    } else {
        printf("\nTEST RESULT: Top token is %d (expected 659)\n", top[0].second);
    }
    
    return 0;
}

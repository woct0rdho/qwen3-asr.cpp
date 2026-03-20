#include "text_decoder.h"
#include <cstdio>
#include <vector>
#include <algorithm>

int main() {
    printf("=== Test Decoder WITHOUT Audio (405 tokens) ===\n\n");
    
    qwen3_asr::TextDecoder decoder;
    if (!decoder.load_model("models/qwen3-asr-0.6b-f16.gguf")) {
        fprintf(stderr, "Failed to load model: %s\n", decoder.get_error().c_str());
        return 1;
    }
    printf("Model loaded\n");
    
    if (!decoder.init_kv_cache(2048)) {
        fprintf(stderr, "Failed to init KV cache: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    int32_t n_audio_frames = 390;
    
    const int32_t im_start = 151644;
    const int32_t im_end = 151645;
    const int32_t system_token = 8948;
    const int32_t user_token = 872;
    const int32_t assistant_token = 77091;
    const int32_t newline = 198;
    const int32_t audio_start = 151669;
    const int32_t audio_pad = 151676;
    const int32_t audio_end = 151670;
    
    std::vector<int32_t> tokens;
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    tokens.push_back(audio_start);
    for (int i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(audio_pad);
    }
    tokens.push_back(audio_end);
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    printf("Token sequence length: %zu\n", tokens.size());
    
    std::vector<float> logits;
    
    // NO audio injection - just token embeddings
    if (!decoder.forward(tokens.data(), tokens.size(), 0, logits)) {
        fprintf(stderr, "Forward pass failed: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    printf("Logits size: %zu\n", logits.size());
    
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
    
    printf("\nTop 5 logits at last position:\n");
    for (int i = 0; i < 5; ++i) {
        printf("  [%d] token=%d logit=%f\n", i, top[i].second, top[i].first);
    }
    
    printf("\nToken 198 (newline) logit: %f\n", last_logits[198]);
    printf("Token 151645 (EOS) logit: %f\n", last_logits[151645]);
    printf("Token 11528 (language) logit: %f\n", last_logits[11528]);
    
    if (top[0].second == 11528) {
        printf("\nTEST PASSED: Top token is 11528 (language)\n");
    } else {
        printf("\nTEST RESULT: Top token is %d (expected 11528)\n", top[0].second);
    }
    
    return 0;
}

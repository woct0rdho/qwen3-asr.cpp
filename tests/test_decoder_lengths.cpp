#include "text_decoder.h"
#include <cstdio>
#include <vector>
#include <algorithm>

int main() {
    qwen3_asr::TextDecoder decoder;
    if (!decoder.load_model("models/qwen3-asr-0.6b-f16.gguf")) {
        fprintf(stderr, "Failed to load model: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    if (!decoder.init_kv_cache(512)) {
        fprintf(stderr, "Failed to init KV cache: %s\n", decoder.get_error().c_str());
        return 1;
    }
    
    int lengths[] = {5, 10, 20, 50, 100};
    
    for (int n_tokens : lengths) {
        decoder.clear_kv_cache();
        
        std::vector<int32_t> tokens;
        tokens.push_back(151669);  // audio_start
        for (int i = 0; i < n_tokens - 2; ++i) {
            tokens.push_back(151676);  // audio_pad
        }
        tokens.push_back(151670);  // audio_end
        
        std::vector<float> logits;
        if (!decoder.forward(tokens.data(), tokens.size(), 0, logits)) {
            fprintf(stderr, "Forward pass failed: %s\n", decoder.get_error().c_str());
            return 1;
        }
        
        const auto & cfg = decoder.get_config();
        int32_t vocab_size = cfg.vocab_size;
        
        const float * last_logits = logits.data() + (n_tokens - 1) * vocab_size;
        
        int32_t argmax = 0;
        float max_val = last_logits[0];
        for (int32_t i = 1; i < vocab_size; ++i) {
            if (last_logits[i] > max_val) {
                max_val = last_logits[i];
                argmax = i;
            }
        }
        
        printf("n_tokens=%d: argmax=%d (logit=%.2f)\n", n_tokens, argmax, max_val);
    }
    
    return 0;
}
